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

## New metric

```python
gr.eval.register_metric("coverage", lambda rec, gains, relevant, k: len(set(rec[:k])) / k)
gr.eval.register_aggregation("p90", lambda v: float(np.percentile(v, 90)))
```

Always add the paper to `grouprec/references.py` so `gr.cite("MyMethod")` works — a test
(`tests/test_citations.py`) **fails CI** if a registered aggregator/model/dataset has no
citation. Citations are auto-collected from the objects used in a run via
`gr.collect_citations(...)` / `gr.Experiment(..., cite=[recommender, dataset])`.
