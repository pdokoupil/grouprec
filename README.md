# grouprec

> **Group recommender systems for Python** — results aggregation and profile
> aggregation as first-class citizens behind one API.

[![Python versions](https://img.shields.io/pypi/pyversions/grouprec.svg)](https://pypi.org/project/grouprec/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/pdokoupil/grouprec?style=social)](https://github.com/pdokoupil/grouprec/stargazers)
[![CI](https://github.com/pdokoupil/grouprec/actions/workflows/ci.yml/badge.svg)](https://github.com/pdokoupil/grouprec/actions)
[![codecov](https://codecov.io/gh/pdokoupil/grouprec/branch/main/graph/badge.svg)](https://codecov.io/gh/pdokoupil/grouprec)
[![docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://pdokoupil.github.io/grouprec)
[![code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)

Keywords: **group recommendation · group recommender systems · preference
aggregation · collaborative filtering · fairness · Python**.

## Why this exists

**One library for both kinds of group recommender** — deep-learning SOTA models
(AGREE, GroupIM, ConsRec, …) *and* classic, fairness-oriented aggregators (GFAR,
EP-FuzzDA, RLProp/LTP, …) — behind **one API**, so you compare them head-to-head on the
same data, splits, and metrics.

Group recommendation has split into two communities that rarely compare against each
other (Peska et al., *Bridging the Rift*, UMAP 2025); general-purpose toolkits (LensKit,
RecBole, Cornac, …) are single-user. `grouprec` bridges that — swap a results-aggregator
for a deep model, score the *same* recommender under **coupled** vs **decoupled**, and
put both in one leaderboard. See [the design & the rift](docs/design.md).

## Install

```bash
pip install -e ".[full]"
```

## Quickstart

```python
import grouprec as gr
from grouprec import GroupRecommender, evaluate
from grouprec.backends import EASE          # or implicit_als(), Popularity(), lenskit(...)

data   = gr.make_blobs_dataset(seed=0)       # or gr.datasets.load("ml-1m") / your own Dataset
groups = gr.groups.synthetic(data, kind="similar", size=4, n=1000, metric="pearson")
folds  = gr.split.crossval(data, k=5, seed=0)

rec = GroupRecommender(EASE(), gr.aggregators.get("GFAR"), normalize="minmax")

report = evaluate(rec, data, groups, folds,
                  protocol=["coupled", "decoupled"],            # score both at once
                  metrics=["ndcg@10", "recall@10"],
                  group_aggregations=["mean", "min", "minmax"])  # the fairness lens
print(report.pivot())
```

Aggregators are plain numpy and usable standalone on a `(n_members, n_items)` matrix:

```python
import numpy as np
from grouprec.aggregators import get
get("LTP").aggregate(np.array([[5., 3., 1.], [2., 3., 4.]]), k=2)
```

## Bring your own recommender (wide algorithm coverage via adapters)

We **don't reimplement single-user recommenders** — we wrap the established
frameworks so you inherit their entire model zoos, plus a few dependency-free
built-ins. Any object with `fit(dataset)` + `score(users, items=None)` is a backend.

| Backend | How | Algorithms |
|---|---|---|
| **Built-in** (no extra deps) | `Popularity`, `EASE`, `ItemKNN`, `Random` | popularity, EASE^R, item-kNN |
| **implicit** `[implicit]` | `gr.backends.implicit_als(factors=64)`, `implicit_bpr(...)` | ALS, BPR |
| **LensKit** `[lenskit]` | `gr.backends.lenskit(ImplicitMFScorer(...))` | ImplicitMF, BiasedMF, ItemKNN, UserKNN, EASE, SLIM, … |
| **RecBole** `[recbole]` | `gr.backends.recbole(model, dataset)` *(experimental)* | the full RecBole zoo |

## Deep group models (`[torch]`)

Lazily imported, so `import grouprec` never pulls torch:

```python
from grouprec.models import AGREE, GroupIM, ConsRec, make_synthetic_group_data
gd = make_synthetic_group_data(seed=0)
model = ConsRec(gd.groups, gd.group_interactions).fit(gd.dataset)
model.recommend(gd.groups[0], k=10)          # paradigm="profile" -> coupled group-level
```

Model zoo: **`NCFGroup`**, **`AGREE`**, **`GroupIM`** (InfoMax SSL), **`ConsRec`**
(overlap/hypergraph/LightGCN consensus), **`HyperGroup`** (HGNN), **`AlignGroup`**
(InfoNCE member/group alignment) — each reviewed against its original repo. They plug
into `benchmark(..., level="sampled")` and share the **same coupled leaderboard** as
results-aggregators. Reproduced on CAMRa2011 (HR@5): GroupIM 0.62, ConsRec 0.62,
AGREE/AlignGroup/HyperGroup ~0.59, EASE+GFAR 0.58.

There's also a **profile-first** path (`ProfileGroupRecommender`):
*aggregate-then-recommend* — merge member profiles (`average`/`union`/`sum`) into a
pseudo-user, query the base RS once.

## Datasets (license-aware)

```python
data = gr.datasets.load("ml-1m")             # auto-fetched, cached, parsed
data = gr.datasets.k_core(data, k=5)         # k-core / binarize / min-count preprocessing
print(gr.datasets.info("kgrec").license)     # every entry carries license + citation
gd   = gr.datasets.load_consrec("path/CAMRa2011")          # explicit-group benchmark
gd   = gr.datasets.load_yin(gr.datasets.fetch_yin(accept_license=True), "yelp")
```

| Policy | Datasets | Behavior |
|---|---|---|
| `auto` | MovieLens 100K/1M/25M/32M/latest-small | fetched for your own use (GroupLens permits research download; *redistribution* terms differ by release — 100K/1M forbid it, 25M/latest allow it under same terms) |
| `auto_nc` | KGRec, Last.fm, Yelp-LA, Douban-SH | non-commercial — fetched only after `accept_license=True`, citations surfaced |
| `manual` | CAMRa2011, Mafengwo, Weeplaces | redistribution unclear — `load` prints where to download |

**Licensing:** The library code is released under **MIT**. Datasets are not relicensed by the library and remain subject to their upstream terms.
The library does not bundle dataset files; for datasets with access restrictions or non-commercial terms, loading is enabled only when the user accepts the dataset-specific license or follows the upstream download instructions. Plus
`from_huggingface(...)` / `from_path(...)` for anything else.

## Benchmark, leaderboard & the rift

```python
res = gr.benchmark(recs, tasks, protocols=["coupled", "decoupled"], metrics=["ndcg@10"])
res.to_csv("leaderboard.csv")                # tidy long-format
res.leaderboard("ndcg", k=10, protocol="coupled")     # ranking flips vs "decoupled"
```

`scripts/run_leaderboard.py` regenerates a CSV + a static HTML page (GitHub-Pages
hostable) + an accumulating `LeaderboardStore`. A live Streamlit browser is in
`examples/leaderboard_app.py` (`pip install grouprec[demo]`). Plot the
relevance–fairness trade-off with `gr.bench.viz.plot_pareto(...)`.

`scripts/build_showcase.py` produces the headline two-panel board
([`docs/leaderboard.html`](docs/leaderboard.html), refreshed by CI): (A) deep models vs
`EASE+GFAR`/`EASE+AVG` under coupled sampled HR/NDCG on CAMRa2011 + Mafengwo; (B)
aggregators decoupled on MovieLens, where **LTP/RLProp lead the fairness–utility
trade-off** (`ndcg.min` 0.27 vs AVG 0.22; AVG wins raw utility).

## What's inside

- **Aggregators** (numpy core): `ADD AVG LMS MUL MPL AVGNM BDC FAI`,
  `GFAR GreedyLM PAR SPGreedy EPFuzzDA` (fairness), `RLProp LTP PeriodicFAI
  EPFuzzDAWeighted SDAA SIAA` (sequential).
- **Group formation**: `gr.groups.synthetic(kind="random|similar|divergent|outlier")`.
- **Evaluation**: `coupled` / `decoupled` / `sampled` protocols; metrics at three
  levels — per-member (`ndcg/recall/hr/ar/…` × `mean/min/minmax/jain/zero`), per-list
  (`novelty`, `list_coverage`, `register_list_metric`), per-run (carbon via
  `gr.track_emissions`). Long-term fairness: `dMAE`, `groupSatO`, `groupDisO`.
- **Reproducibility**: `gr.Experiment` (seed + env + git SHA/dirty/diff + citations),
  `gr.set_seed`, `gr.cite("ConsRec")`.

## Extras at a glance

```python
gr.cite("ConsRec")                            # BibTeX for any implemented method
gr.collect_citations(rec, dataset)            # auto-collect cites for what a run used
with gr.Experiment("run1", seed=42, cite=[rec, dataset]) as exp:  # cites auto-resolved
    ...                                       # writes runs/run1-<ts>/ on exit
with gr.track_emissions() as em: ...          # carbon cost of a run
gr.eval.register_metric("coverage", fn)       # custom per-member metric
gr.eval.register_list_metric("ild", fn)       # custom per-list metric
```

## Contributing — researchers especially welcome

Published a group-rec method? **Add it** (small PR — [CONTRIBUTING.md](CONTRIBUTING.md)
has copy-paste recipes for an aggregator / deep model / dataset / metric), or just
**link your repo** in a `port-request` issue and we'll port it. Every method
ships with its citation, so your work is credited wherever the library is used.
**Students & practitioners** welcome too (`good-first-issue`).

## Citation

If you use `grouprec`, please cite it (`CITATION.cff`) and the relevant method
(`gr.cite(...)`). Docs: <https://pdokoupil.github.io/grouprec>.

## License

MIT. Datasets retain their own licenses (see Datasets above).
