# GroupRec — RecSys 2026 demo submission

> **Patrik Dokoupil, Ludovico Boratto, and Ladislav Peska.**
> *GroupRec: A Unified Toolkit for Reproducible and Inspectable Group Recommendation Research.*
> Demo track — **under review at RecSys 2026.**

This page is a reviewer-facing tour of the demo. The star of the demonstration is the
**interactive inspector**: a live, in-browser view of a group recommendation being
produced, where every displayed number is a genuine call into the library.

[Open the inspector&nbsp;↗](../group_rec_inspector.html){ .md-button .md-button--primary }
[How it's built](../INSPECTOR.md){ .md-button }
[GitHub](https://github.com/pdokoupil/grouprec){ .md-button }

<iframe class="inspector-embed" src="../../group_rec_inspector.html" title="grouprec interactive inspector" loading="lazy"></iframe>

## What the demo shows

- **Real, not mocked.** Rankings come from `GroupRecommender` (aggregators) and
  `GroupIM.group_scores` (deep model). Groups are sampled with `gr.groups.synthetic`
  and their interactions **derived** — not simulated — via
  `gr.groups.derive_group_interactions`.
- **Steerable & inspectable.** Per-member influence sliders reweight the aggregation /
  the model's attention pooling; a Top-K SAE explanation layer surfaces each member's
  most *distinctive* latent concepts.
- **Self-contained.** The whole thing is computed offline and baked into a single HTML
  file — no server, no API calls at view time.

## Reproduce it

```bash
pip install grouprec[torch]
grouprec-build-inspector --out group_rec_inspector.html
```

Full write-up of every framework call and every displayed subset:
[Interactive inspector — how it's built](../INSPECTOR.md).

## The library behind the demo

- [Design (the rift)](../design.md) — the two-community split the toolkit bridges
- [Evaluation](../evaluation.md) — coupled / decoupled / sampled protocols
- [Reproducibility](../reproducibility.md) — pinned datasets, `gr.Experiment`, citations
- [README](https://github.com/pdokoupil/grouprec#readme)
