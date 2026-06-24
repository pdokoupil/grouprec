# Datasets & licensing

`gr.datasets.load("ml-1m")` fetches + caches + parses. Every entry records a license,
citation, and **download policy**:

| Policy | Datasets | Behavior |
|---|---|---|
| `auto` | MovieLens 100K/1M/25M/32M | auto-fetched (GroupLens license permits it) |
| `auto_nc` | KGRec, Last.fm (Taste Profile) | non-commercial — needs `load(..., accept_license=True)` |
| `manual` | CAMRa2011, Mafengwo, Weeplaces, Yelp, Douban | prints where to download + where to drop files |

**Licensing:** the library is MIT (that's the *code*); datasets keep their own licenses.
These never conflict because we **never redistribute the data** — `auto` is fetched from
the canonical host on your machine; anything non-redistributable is `manual`.

## Group-structured datasets

```python
gd = gr.datasets.load_consrec("path/to/CAMRa2011")   # AGREE/ConsRec format
# -> GroupBenchmarkData(dataset, groups, group_interactions, test_instances)
```

Use `gd.test_instances` with `evaluate_sampled` / `benchmark(level="sampled")`.

## Bring your own

```python
gr.datasets.from_path("ratings.csv", user_col="u", item_col="i", rating_col="r")
gr.datasets.from_huggingface("some/repo", user_col="userId", item_col="movieId")
```

## Preprocessing

```python
data = gr.datasets.k_core(data, k=5)
data = gr.datasets.filter_min_interactions(data, min_per_user=5, min_per_item=5)
data = gr.datasets.binarize(data, threshold=4.0)
```
