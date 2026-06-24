# Concepts

## Two paradigms, one API

- **Results aggregation** (`paradigm="results"`): a base single-user recommender scores
  each member, an `Aggregator` combines those scores into a group list. Wrapped by
  `GroupRecommender(base, aggregator)`.
- **Profile aggregation** (`paradigm="profile"`): a model consumes member profiles /
  group interactions and emits a group list directly (the deep models in
  `grouprec.models`).

Both expose `fit(dataset)` + `recommend(members, k, *, exclude=None, candidates=None)`,
so they're interchangeable in evaluation and benchmarking — that's the *rift bridge*.

## Aggregators operate on a score matrix

Every aggregator takes a dense `(n_members, n_items)` matrix and returns item
**indices**. They are numpy-only and unit-testable in isolation; `GroupRecommender`
binds them to a base RS and maps indices back to item ids.

!!! note "Normalize for proportional fairness"
    GFAR / EP-FuzzDA / RLProp / LTP assume **commensurable** member scores. Use
    `GroupRecommender(..., normalize="minmax")` (per-member) to reproduce the
    literature — raw MF scores on different per-user scales make these degenerate.

## Two data regimes

| Regime | Groups | Ground truth | Who runs | Protocol |
|---|---|---|---|---|
| **member-level** | synthetic (ML/KGRec/Last.fm) | each member's held-out feedback | aggregators | coupled + decoupled |
| **group-level** | real/inferred (CAMRa/Mafengwo/Yelp/Douban/Weeplaces) | the group's held-out choice | deep **and** aggregators | coupled (sampled 1-vs-N) |

Deep models can't use synthetic groups (no group-level ground truth to train on); the
bridge therefore lives in the group-level regime.

## Custom group-similarity metrics

`gr.groups.synthetic(..., metric=...)` accepts, beyond the `"pearson"`/`"cosine"`/`"jaccard"`
built-ins, **a callable** `f(data) -> (n_users, n_users)` array (use it to define a custom
similarity or to compute it over side-information/feature embeddings) or a **precomputed**
`(n_users, n_users)` matrix aligned to `data.users`:

```python
from grouprec.groups import synthetic, similarity_matrix
S = my_feature_similarity(data)                      # your own (n_users, n_users) matrix
groups = synthetic(data, kind="similar", size=4, n=100, metric=S)
groups = synthetic(data, kind="divergent", size=4, metric=lambda d: similarity_matrix(d, "cosine"))
```
