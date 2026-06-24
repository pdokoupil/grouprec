# grouprec

Group recommender systems for Python — **results aggregation** and **profile
aggregation** as first-class citizens behind one API.

The README is the 30-second tour; these docs are the longer reference. Start with
[Concepts](concepts.md), then [Evaluation & the rift](evaluation.md). To contribute a
method, jump to [Extending](extending.md).

## Install

```bash
pip install grouprec                 # light numpy/scipy core
pip install grouprec[torch]          # deep group models
pip install grouprec[implicit,lenskit,recbole]   # base-RS backends
pip install grouprec[full]           # everything
```

## 60-second example

```python
import grouprec as gr
from grouprec import GroupRecommender, evaluate
from grouprec.backends import EASE

data   = gr.make_blobs_dataset(seed=0)
groups = gr.groups.synthetic(data, kind="similar", size=4, n=100)
folds  = gr.split.crossval(data, k=5, seed=0)
rec    = GroupRecommender(EASE(), gr.aggregators.get("GFAR"), normalize="minmax")

report = evaluate(rec, data, groups, folds, protocol=["coupled", "decoupled"],
                  metrics=["ndcg@10"], group_aggregations=["mean", "min", "minmax"])
print(report.pivot())
```
