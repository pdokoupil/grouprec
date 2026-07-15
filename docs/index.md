<div class="hero" markdown>

# grouprec

**Group recommender systems for Python.** Deep-learning models *and* classic,
fairness-oriented aggregators behind **one API** — with license-aware datasets,
coupled/decoupled evaluation, and an interactive inspector that shows every
recommendation being made.

[Open the inspector&nbsp;↗](group_rec_inspector.html){ .md-button .md-button--primary }
[Quickstart](#quickstart){ .md-button }
[GitHub](https://github.com/pdokoupil/grouprec){ .md-button }

</div>

## See a group recommendation being made

The **inspector** is the fastest way to understand what the library does: pick a
group, move each member's influence slider, and watch the ranking change. *Every*
number is a real `grouprec` call — aggregators through `GroupRecommender`, the deep
model through `GroupIM.group_scores` — computed offline and baked into one page.

<iframe class="inspector-embed" src="group_rec_inspector.html" title="grouprec interactive inspector" loading="lazy"></iframe>

[Open the full inspector&nbsp;↗](group_rec_inspector.html){ .md-button .md-button--primary }
[How it's built](INSPECTOR.md){ .md-button }

## Why grouprec

<div class="grid cards" markdown>

-   **One API, two paradigms**

    Swap a fairness aggregator (GFAR, EP-FuzzDA, RLProp/LTP) for a deep model
    (GroupIM, ConsRec, AGREE, …) without touching the rest of your pipeline.

-   **Evaluation across the rift**

    Score the *same* recommender under **coupled** vs **decoupled** protocols, with
    per-member fairness lenses (`min` / `minmax` / Jain) — the comparison the two
    group-rec communities rarely make.

-   **Reproducible by default**

    `gr.Experiment` captures seed + environment + git SHA + citations; datasets are
    pinned by checksum; every method ships its BibTeX via `gr.cite(...)`.

-   **Bring your own everything**

    Adapters for LensKit / implicit / RecBole, license-aware dataset loaders, and
    hooks to register custom metrics, aggregators, and group kinds.

</div>

## Quickstart { #quickstart }

```bash
pip install grouprec                 # light numpy/scipy core
pip install grouprec[torch]          # deep group models + the inspector generator
pip install grouprec[full]           # everything
```

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

Regenerate the inspector yourself (needs the `[torch]` extra):

```bash
grouprec-build-inspector --out group_rec_inspector.html
```

## Backed by research

A demo-track paper describing this toolkit is **under review at RecSys 2026**:

> Patrik Dokoupil, Ludovico Boratto, and Ladislav Peska.
> *GroupRec: A Unified Toolkit for Reproducible and Inspectable Group Recommendation Research.*

See the [demo-paper landing page](papers/demo.md) for the reviewer-facing tour.

## Explore the docs

- [Design (the rift)](design.md) — why group recommendation split into two communities
- [Concepts](concepts.md) · [Evaluation](evaluation.md) · [Reproducibility](reproducibility.md)
- [Extending](extending.md) — add an algorithm, dataset, metric, or group kind
- [Integration](integration.md) — LensKit / RecBole / implicit backends
- [README on GitHub](https://github.com/pdokoupil/grouprec#readme) · [Citation](https://github.com/pdokoupil/grouprec/blob/main/CITATION.cff)
