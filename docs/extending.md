# Extending grouprec

We **especially want researchers**: add your method, or just open a `port-request`
issue with a link to your repo/paper and we'll port it. Students & practitioners
welcome — see `good-first-issue`. Full recipes are in
[CONTRIBUTING.md](https://github.com/pdokoupil/grouprec/blob/main/CONTRIBUTING.md); the
essentials:

## New aggregator

```python
from grouprec.aggregators.base import Aggregator, as_score_matrix, top_k_indices, available_mask

class MyAgg(Aggregator):
    name = "MyAgg"
    def aggregate(self, scores, k, *, exclude=None):
        rm = as_score_matrix(scores)
        return top_k_indices(rm.mean(axis=0), k, available_mask(rm.shape[1], exclude))
```
Register in `aggregators/__init__.py::_REGISTRY`, add a citation to `references.py`,
add a test vs a reference oracle.

## New deep model

Implement `fit(dataset)` + `recommend(members, k, *, exclude=None, candidates=None)`,
set `paradigm="profile"`, keep torch under `grouprec/models/`. Validate with
`evaluate_sampled` on CAMRa2011.

## New dataset

Add a `DatasetSpec` (license + citation + policy) + a loader. Never bundle data.

## New group kind / custom similarity

Similarity is pluggable: `gr.groups.synthetic(..., metric=...)` takes `"pearson"`/`"cosine"`/
`"jaccard"`, **a precomputed matrix**, or **a callable** `f(data) -> (n_users, n_users)` (use
this to compute similarity over a feature subset or side-information embeddings).

Group *kinds* are pluggable the same way: pass `kind=` a builder
`f(sim, size, rng) -> list[int] | None` (member indices, or `None` if it stalls). The public
helpers `build_predicate_group` and `build_outlier_group` are the building blocks. Example —
a **2+2** group (a size-4 group made of two similar pairs that are dissimilar across pairs):

```python
import numpy as np, grouprec as gr

def two_plus_two(sim, size, rng, hi=0.3, lo=0.0):
    n = sim.shape[0]
    for _ in range(200):
        a = int(rng.integers(n))
        far = np.flatnonzero(sim[a] <= lo)                 # an anchor dissimilar to a
        if not far.size:
            continue
        b = int(rng.choice(far))
        pa = [int(x) for x in np.flatnonzero(sim[a] >= hi) if x not in (a, b)][:1]  # partner near a
        pb = [int(x) for x in np.flatnonzero(sim[b] >= hi) if x not in {a, b, *pa}][:1]  # near b
        if pa and pb:
            return [a, *pa, b, *pb]
    return None

groups = gr.groups.synthetic(data, kind=two_plus_two, size=4, n=10, metric="pearson", seed=0)
# groups.metadata["kind"] == "two_plus_two"
```

## New metric

```python
gr.eval.register_metric("coverage", lambda rec, gains, relevant, k: len(set(rec[:k])) / k)
gr.eval.register_aggregation("p90", lambda v: float(np.percentile(v, 90)))
```

Always add the paper to `grouprec/references.py` so `gr.cite("MyMethod")` works — a test
(`tests/test_citations.py`) **fails CI** if a registered aggregator/model/dataset has no
citation. Citations are auto-collected from the objects used in a run via
`gr.collect_citations(...)` / `gr.Experiment(..., cite=[recommender, dataset])`.
