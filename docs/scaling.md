# Scaling: sparse interactions & lazy similarity

`grouprec` runs on full-size benchmarks (ML-25M, ML-32M, the complete `ml-latest`) on an
ordinary laptop. Two structures used to make that impossible, and both are now avoided by
default — you don't opt in.

## The two walls

For a dataset with `U` users and `I` items:

| structure | size at ML-32M (≈200k × 84k) | size at full `ml-latest` (331k × 83k) |
|---|---|---|
| dense user–item matrix | ~135 GB | **220 GB** |
| dense user–user similarity | ~320 GB | **876 GB** |

Interaction data is >99% sparse, so the first is pure waste. The second is subtler: it is
genuinely dense, but **you almost never need all of it**.

## Sparse interactions

`Dataset.user_item_csr()` is the memoised CSR view every internal consumer now uses —
`EASE`, `ItemKNN`, `ProfileGroupRecommender`, `GroupIM`, and group formation. The full
`ml-latest` interaction matrix is **407 MB** as CSR against 220 GB dense.

```python
X = data.user_item_csr(value="binary")   # scipy.sparse.csr_matrix, cached
```

`user_item_matrix()` still returns the dense array for small data, tests, and callers that
genuinely need one — it just is not on any hot path.

**Torch interop.** Deep models hold the CSR through `models._sparse.CsrRows`, which
densifies **only the rows a batch touches** and returns a `torch.Tensor`. It mirrors
ndarray indexing (`M[i]`, `M[a:b]`, `M[[i,j]]`, and N-D index arrays), so model code is
unchanged. `CsrRows.to_torch_csr()` gives a `torch.sparse_csr_tensor` for ops that take
sparse operands directly.

!!! note "What is still dense, and why"
    EASE's Gram matrix is `(n_items, n_items)` and is dense *by nature* — it is the thing
    being inverted. That, not the interactions, is EASE's real size limit: ~4.3 GB at 23k
    items. Reduce the item space (`k_core`) rather than the user space if you hit it.

## Lazy user–user similarity

The group builders only ever read **whole rows** — `sim[member]` — and only a handful:
growing one group of size *k* touches ~*k* rows. So `LazySimilarity` computes rows on
demand and keeps an LRU cache.

```python
sim = gr.groups.similarity_matrix(data, "pearson")     # lazy="auto" (default)
sim[42]                                                # one row, computed & cached
sim.cache_stats()                                      # {'rows_computed': 1, ...}
```

`lazy="auto"` returns a `LazySimilarity` only when the dense matrix would exceed
`max_dense_gib` (default 2 GiB), so small data keeps the fast vectorised path. Force it
either way with `lazy=True` / `lazy=False`, and tune the cache with `cache_rows`.

**Rows are exact, not approximated.** For rows centred over all `n` items,

```
cov(i,j) = (1/n)·(xᵢ·xⱼ) − μᵢ·μⱼ          sd(i)² = (1/n)·(xᵢ·xᵢ) − μᵢ²
```

so a Pearson row is one sparse product `X @ X[i].T` plus two precomputed length-`U`
vectors — the centring never materialises. Cosine and Jaccard factor the same way. Lazy
rows match the dense matrix to ~1e-13 (Pearson) and exactly (cosine/Jaccard), and
`synthetic()` returns identical groups on either path.

`similarity_matrix` is a `LazySimilarity`, not an ndarray, when lazy — it supports
`.shape` and `sim[i]` only. Pass `lazy=False` if you need to slice or broadcast over the
whole matrix. Callable and precomputed metrics are dense by definition.

## Measured

Full `ml-latest` (330,975 users × 83,239 items, 33.8M ratings), 16 GB laptop, via the
public API:

| step | result |
|---|---|
| `user_item_csr()` | 407 MB (vs 220 GB dense) |
| `similarity_matrix(...)` | → `LazySimilarity`, 0.9 s |
| one similarity row | ~68 ms |
| `groups.synthetic(kind=...)` | **0.6–0.7 s** per regime |
| peak process RSS | **2.9 GB** |

The 876 GB dense similarity matrix is never allocated; 12 rows are.
