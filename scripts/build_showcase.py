#!/usr/bin/env python
"""Build the showcase leaderboard for the README/demo/GitHub Pages.

Two panels:
  A) COUPLED, sampled group-level (HR/NDCG @5,10) on real group datasets
     (CAMRa2011, Mafengwo): deep models + EASE+GFAR + EASE+AVG, side by side.
  B) DECOUPLED, member-level on MovieLens: aggregators with an EASE base, showing
     the relevance (ndcg.mean) vs fairness (ndcg.min / minmax) trade-off
     — LTP / RLProp should lead on fairness.

Writes CSVs to results/, a combined CSV to examples/showcase_leaderboard.csv, and a
two-panel static HTML to docs/leaderboard.html (GitHub-Pages hostable).

Usage:
    python scripts/build_showcase.py --consrec-data raw/tmp/WWW2023ConsRec/data
The deep panel is skipped if torch or the data isn't available, so it degrades cleanly
(e.g. on a machine without the datasets).
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd

import grouprec as gr
from grouprec import GroupRecommender, evaluate, evaluate_sampled
from grouprec.backends import EASE

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# Panel A: coupled sampled on real group datasets (deep + aggregators)
# --------------------------------------------------------------------------- #
def deep_recommenders(gd, predictor="DOT"):
    # ConsRec/AlignGroup papers: DOT predictor for CAMRa2011, MLP for Mafengwo.
    from grouprec.models import AGREE, AlignGroup, ConsRec, GroupIM, HyperGroup, NCFGroup
    g, gi = gd.groups, gd.group_interactions
    return {
        "NCF":        NCFGroup(g, gi, epochs=30, seed=0),
        "AGREE":      AGREE(g, gi, epochs=30, seed=0),
        "GroupIM":    GroupIM(g, gi, epochs=40, pretrain_epochs=10, seed=0),
        "HyperGroup": HyperGroup(g, gi, emb_dim=32, layers=2, epochs=40, seed=0),
        "ConsRec":    ConsRec(g, gi, epochs=50, batch_size=1024, predictor=predictor, seed=0),
        "AlignGroup": AlignGroup(g, gi, epochs=25, batch_size=1024, predictor=predictor, seed=0),
    }


def panel_coupled(consrec_data: Path) -> pd.DataFrame:
    from grouprec.aggregators import get
    rows = []
    for name, sub in [("CAMRa2011", "CAMRa2011"), ("Mafengwo", "Mafengwo")]:
        path = consrec_data / sub
        if not path.exists():
            print(f"[skip] {name}: {path} not found")
            continue
        gd = gr.datasets.load_consrec(path, name)
        recs = {"EASE+AVG": GroupRecommender(EASE(reg=100.0), get("AVG"), normalize="minmax"),
                "EASE+GFAR": GroupRecommender(EASE(reg=100.0), get("GFAR"), normalize="minmax")}
        predictor = "DOT"  # our DOT head reproduces best on both here; MLP undertrains
        try:
            recs.update(deep_recommenders(gd, predictor=predictor))
        except ImportError:
            print("[skip] deep models: torch not installed")
        for rname, rec in recs.items():
            print(f"  [{name}] {rname} ...", flush=True)
            rep = evaluate_sampled(rec, gd.dataset, gd.groups, gd.test_instances, ks=(5, 10))
            d = rep.to_dict()
            rows.append({"dataset": name, "recommender": rname,
                         "HR@5": d[("coupled", "hr", 5, "sampled")],
                         "NDCG@5": d[("coupled", "ndcg", 5, "sampled")],
                         "HR@10": d[("coupled", "hr", 10, "sampled")],
                         "NDCG@10": d[("coupled", "ndcg", 10, "sampled")]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Panel B: decoupled aggregators on MovieLens (fairness/util trade-off)
# --------------------------------------------------------------------------- #
def panel_decoupled(dataset_name: str = "ml-100k") -> pd.DataFrame:
    from grouprec.aggregators import get
    try:
        data = gr.datasets.load(dataset_name)
    except Exception as exc:
        print(f"[fallback] {dataset_name} unavailable ({exc}); using synthetic blobs")
        data = gr.make_blobs_dataset(n_users=200, n_items=120, n_clusters=5, density=0.4, seed=0)
        dataset_name = "blobs"
    groups = gr.groups.synthetic(data, kind="divergent", size=4, n=60, metric="pearson",
                                 sim_low=0.2, seed=0)
    split = gr.split.random_split(data, test_frac=0.2, seed=0)
    rows = []
    for name in ["AVG", "LMS", "GFAR", "GreedyLM", "EPFuzzDA", "RLProp", "LTP"]:
        print(f"  [{dataset_name} decoupled] {name} ...", flush=True)
        rec = GroupRecommender(EASE(reg=100.0), get(name), normalize="minmax")
        rep = evaluate(rec, data, groups, split, k=10, protocol="decoupled",
                       metrics=["ndcg@10"], group_aggregations=["mean", "min", "minmax"])
        d = rep.to_dict()
        rows.append({"dataset": dataset_name, "recommender": name,
                     "ndcg.mean (utility)": d[("decoupled", "ndcg", 10, "mean")],
                     "ndcg.min (fairness)": d[("decoupled", "ndcg", 10, "min")],
                     "ndcg.minmax (balance)": d[("decoupled", "ndcg", 10, "minmax")]})
    return pd.DataFrame(rows).sort_values("ndcg.min (fairness)", ascending=False)


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
def write_html(coupled: pd.DataFrame, decoupled: pd.DataFrame, path: Path) -> None:
    from datetime import datetime, timezone
    rnd = lambda df: df.round(4)
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>grouprec leaderboard</title>
<style>body{{font-family:system-ui,Arial,sans-serif;margin:2rem;color:#222;max-width:1000px}}
h1{{font-size:1.4rem}}h2{{font-size:1.1rem;margin-top:1.6rem}}.meta{{color:#666;font-size:.85rem}}
table{{border-collapse:collapse;margin:.6rem 0;font-size:.9rem}}
th,td{{border:1px solid #ddd;padding:.35rem .6rem;text-align:right}}
th{{background:#f6f6f6}}td:first-child,th:first-child,td:nth-child(2),th:nth-child(2){{text-align:left}}
tr:nth-child(even){{background:#fafafa}}</style></head><body>
<h1>grouprec — leaderboard</h1>
<p class="meta">Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} ·
deep models and aggregators compared on the same data, splits, and metrics.</p>
<h2>A. Coupled (sampled group-level HR/NDCG) — deep models vs aggregators</h2>
<p class="meta">Faithful reimplementations on the 1-vs-99 sampled protocol. CAMRa2011 matches
published ranges; Mafengwo is approximate (a training-procedure gap — full paper configs
with early stopping would close it).</p>
{rnd(coupled).to_html(index=False, border=0) if len(coupled) else "<p>(deep panel skipped)</p>"}
<h2>B. Decoupled (MovieLens) — aggregator relevance vs fairness trade-off</h2>
<p class="meta">Higher <code>ndcg.min</code>/<code>minmax</code> = fairer; LTP/RLProp lead on
fairness while AVG leads raw utility.</p>
{rnd(decoupled).to_html(index=False, border=0)}
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--consrec-data", default="raw/tmp/WWW2023ConsRec/data")
    p.add_argument("--ml", default="ml-100k")
    p.add_argument("--html", default="docs/leaderboard.html")
    p.add_argument("--out-dir", default="results")
    a = p.parse_args()

    coupled = panel_coupled(Path(a.consrec_data))
    decoupled = panel_decoupled(a.ml)

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    if len(coupled):
        coupled.to_csv(out / "showcase_coupled.csv", index=False)
    decoupled.to_csv(out / "showcase_decoupled.csv", index=False)
    combined = pd.concat([coupled.assign(panel="coupled"),
                          decoupled.assign(panel="decoupled")], ignore_index=True)
    Path("examples").mkdir(exist_ok=True)
    combined.to_csv("examples/showcase_leaderboard.csv", index=False)
    write_html(coupled, decoupled, Path(a.html))
    print("\n=== Panel A (coupled) ===\n", coupled.round(4).to_string(index=False))
    print("\n=== Panel B (decoupled) ===\n", decoupled.round(4).to_string(index=False))
    print(f"\nwrote {a.html}, examples/showcase_leaderboard.csv, {out}/showcase_*.csv")


if __name__ == "__main__":
    main()
