# Design: the rift, the families, and the bridge

## The one-sentence pitch

`grouprec` puts **deep-learning SOTA group models** and **classic, fairness-oriented
preference aggregators** behind **one API**, so you can compare them head-to-head on the
same data, splits, and metrics.

## The rift

Group recommendation research has split into two communities that rarely meet
(Peska et al., *Bridging the Rift*, UMAP 2025):

- **Results aggregation** — run a single-user recommender, score each member, then
  combine those scores/lists into one group list. This line is where the **fairness**
  work lives (least-misery, GFAR, EP-FuzzDA, RLProp/LTP, …): given per-member utilities,
  how do we pick a list that's good *and* fair to everyone? Evaluated on synthetic
  groups over MovieLens-style data with coupled/decoupled protocols.
- **Profile aggregation** — learn a group representation directly (attention, mutual
  information, hypergraphs, consensus graphs) and predict for the group as a unit. This
  line is where the **deep SOTA** lives (AGREE, GroupIM, ConsRec, …). Evaluated on
  datasets with real/inferred groups (CAMRa2011, Mafengwo, Yelp, Douban, Weeplaces) with
  group-level leave-one-out HR/nDCG.

They drifted apart into **disjoint datasets, baselines, metrics, and protocols** — so a
"new" method in one community is rarely compared against the other. That's the rift.

## Why we keep the two families distinct (but unified)

They genuinely consume different things and answer different questions, so we don't
pretend they're identical — we make them **interoperable**:

| | Results aggregation | Profile aggregation |
|---|---|---|
| `paradigm` | `"results"` | `"profile"` |
| input | base-RS per-member scores | group/member interactions |
| strength | fairness, interpretability, zero training | accuracy/SOTA |
| API | `GroupRecommender(base, aggregator)` | `grouprec.models.*` |

Both expose the same contract — `fit(dataset)` + `recommend(members, k, …)` — so the
evaluator and `benchmark` drive them identically.

## Two evaluation regimes

- **Member-level** (synthetic groups; ML/KGRec/Last.fm): ground truth is each member's
  held-out feedback. Supports **coupled** (vs feedback) and **decoupled** (vs the base
  RS's predicted `r̂`, isolating the aggregator). This is the aggregator/fairness arena.
- **Group-level** (real/inferred groups; CAMRa/Mafengwo/Yelp/Douban/Weeplaces): ground
  truth is the *group's* held-out choice (sampled 1-vs-N HR/nDCG). Deep models can only
  be trained/evaluated here.

Deep models cannot run on synthetic groups (no group-level signal to learn from), so the
two regimes are real, not cosmetic.

## The bridge

In the **group-level, coupled** regime, *both* families emit a group top-k list scored
against the same group choice — so a results-aggregator (`GFAR`) and a deep model
(`AGREE`) land in the **same leaderboard**, comparable for the first time. Conversely, in
the member-level regime the **same** recommender scored under coupled vs decoupled can
*flip rank*, and the relevance–fairness trade-off becomes a Pareto front
(`gr.bench.viz.plot_pareto`). Making both of these one-liners is the point of the library.

See [Concepts](concepts.md) for the API surface and [Evaluation](evaluation.md) for the
protocol/metric details.
