# Evaluation & the rift

## Protocols

- **coupled** — score the group list against held-out **feedback** (RS + aggregator
  judged together). Works for any recommender.
- **decoupled** — treat each member's predicted scores `r̂` as ground truth, isolating
  the aggregator. Requires a base RS (`.base.score`); deep/profile models are
  **coupled-only** and `benchmark` drops decoupled for them with a warning.

```python
evaluate(rec, data, groups, folds, protocol=["coupled", "decoupled"],
         metrics=["ndcg@10", "recall@10"], group_aggregations=["mean", "min", "minmax"])
```

The **same recommender** scored under both protocols, with the ranking changing between
them, is the rift made tangible.

## Metrics × aggregations

Base metrics: `ndcg dcg ar recall brecall precision hr dfh mrr`. The **group
aggregation is the fairness lens**: `mean` (welfare), `min` (least misery), `minmax`
(balance), `std`, `jain`, `zero` (→ zRecall on recall). Register your own with
`gr.eval.register_metric` / `register_aggregation`.

## Sequential / long-term

`gr.evaluate_sequential(...)` runs N rounds and adds **dMAE** (KAIS, discounted
cumulative-utility imbalance), **groupSatO**, **groupDisO** (Stratigi).

## Sampled group-level (deep-model protocol)

`gr.evaluate_sampled(rec, data, groups, test_instances, ks=(5,10))` ranks the held-out
positive against ~99 negatives (HR@k / nDCG@k) — the protocol AGREE/GroupIM/ConsRec
report, and the one where aggregators and deep models share a leaderboard.
