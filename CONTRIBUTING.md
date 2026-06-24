# Contributing to grouprec

**We especially want researchers.** If you've published a group-recommendation
method, the highest-impact thing you can do is help it live in a maintained,
comparable library:

- **Add your algorithm** (a PR — see below), or
- **Just point us to your repo** by opening an issue tagged `port-request`, and
  we'll port it ourselves. A one-line link is enough to make things happen.

**Students & practitioners welcome too** — porting a baseline, adding a dataset
loader, writing a metric, improving docs, or filing a reproduction discrepancy are
all great first contributions. Look for `good-first-issue`.

Every method ships with its **citation** (`gr.cite("YourMethod")`) so your work is
credited wherever it's used. This is **enforced**: a test
(`tests/test_citations.py`) fails CI if any registered aggregator, deep model, or
dataset has no entry in `grouprec/references.py`. Citations are then **auto-collected**
from the objects actually used in a run — `gr.collect_citations(recommender, dataset)`
and `gr.Experiment(..., cite=[recommender, dataset])` resolve keys for you (wrapped
LensKit/RecBole/implicit backends cite the *framework*, since the transitive algorithm
isn't resolvable).

---

## Add a new aggregator

Subclass `Aggregator` (or `SequentialAggregator` for stateful/sequential ones). Work on a dense
`(n_members, n_items)` score matrix and return item **indices**.

```python
from grouprec.aggregators.base import Aggregator, as_score_matrix, top_k_indices, available_mask

class MyAgg(Aggregator):
    name = "MyAgg"
    def aggregate(self, scores, k, *, exclude=None):
        rm = as_score_matrix(scores)
        avail = available_mask(rm.shape[1], exclude)
        return top_k_indices(rm.mean(axis=0), k, avail)  # your rule here
```

Register it in `grouprec/aggregators/__init__.py::_REGISTRY`, add a citation in
`grouprec/references.py`, and add a test against a reference oracle.

## Add a new deep model

Implement `fit(dataset)` + `recommend(members, k, *, exclude=None, candidates=None)`
and set `paradigm = "profile"`. Put torch-only code under `grouprec/models/`. Validate
with `evaluate_sampled` on CAMRa2011 (`gr.datasets.load_consrec(...)`).

## Add a new dataset

Add a `DatasetSpec` to `grouprec/datasets/registry.py` with **license + citation +
download policy** (`auto` / `auto_nc` / `manual`) and a loader in `loaders.py`. Never
bundle data. For group-structured data, follow `datasets/consrec.py`.

## Add a new evaluation metric

```python
import grouprec as gr
gr.eval.register_metric("coverage", lambda rec, gains, relevant, k: len(set(rec[:k])) / k)
gr.eval.register_aggregation("p90", lambda v: float(np.percentile(v, 90)))
gr.evaluate(model, ..., metrics=["coverage"], group_aggregations=["p90"])
```

You can also profile non-accuracy costs (e.g. carbon) with `gr.track_emissions()`.

## Dev setup

```bash
pip install -e ".[dev,torch,implicit,lenskit]"
pytest -q
ruff check . && mypy grouprec
```
