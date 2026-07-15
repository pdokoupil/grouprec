#!/usr/bin/env python
"""Build the *real-data* group-recommendation inspector (self-contained HTML).

Dataset: MovieLens latest-small (real titles + genres -> interpretable items & concepts;
its license permits redistribution, so titles can be published).

Every *ranking* shown is produced by a genuine ``grouprec`` call -- the aggregators go
through ``GroupRecommender`` and the deep model through ``GroupIM.group_scores``:

    import grouprec as gr
    from grouprec import GroupRecommender
    from grouprec.backends import EASE
    from grouprec.aggregators import WeightedAverageAggregator, EPFuzzDAAggregator
    from grouprec.models import GroupIM

    data   = gr.datasets.load("ml-latest-small")
    groups = gr.groups.synthetic(data, kind="divergent", size=3, n=3, seed=0)
    gints  = gr.groups.derive_group_interactions(data, groups)   # per-group signal (majority, overridable)
    ease   = EASE(reg=200.0).fit(data)

    # results-aggregation (utilitarian / fairness), steered by per-member weights:
    rec = GroupRecommender(ease, EPFuzzDAAggregator(member_weights=w)); rec.dataset_ = data
    rec.recommend(members, k=5, candidates=cands)

    # profile-aggregation deep model, steered by per-member weights:
    gim = GroupIM(groups, gints).fit(data)
    gim.group_scores(members, cands, member_weights=w)        # or gim.recommend(..., member_weights=w)

This script is NOT four lines: around those real calls it adds (a) parsing MovieLens
``movies.csv`` for titles/genres, (b) candidate sampling, (c) a Top-K SAE (adapted from
umap2026, NOT part of grouprec) for the latent concepts, and (d) baking a 125-point
member-weight grid into one JSON blob -- ~400 lines total. The snippet above is the
*essence*; this file is the full generator.

Honest notes:
* Group membership: ``gr.groups.synthetic`` in three regimes (similar/divergent/outlier, 3 each),
  tagged with regime + measured mean pairwise rating correlation.
* Group interactions are DERIVED, not simulated, through the framework's
  ``gr.groups.derive_group_interactions`` (default majority rule: item liked by >=2 members,
  like = rating >=4; the rule is overridable via a predicate).
* Aggregator rankings come from the real ``GroupRecommender`` pipeline with the framework
  ``WeightedAverageAggregator`` / ``EPFuzzDAAggregator`` (member_weights); the slider snaps to the
  nearest grid point.
* The deep model is GroupIM (additive, attention-pooled -> cleanly steerable; AGREE dropped as its
  non-linear head steers weakly). Member steering is the framework's own
  ``GroupIM.group_scores(member_weights=...)`` (reweights its attention pooling) -- NOT SAE steering.
* SAE concepts: a Top-K SAE (adapted from umap2026, external to grouprec) on GroupIM's item
  embeddings, labelled by genre. This is an *explanation* layer, not a grouprec component.

Usage:
    python scripts/build_inspector.py --out docs/group_rec_inspector.html
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import grouprec as gr
from grouprec import GroupRecommender
from grouprec.backends import EASE
from grouprec.models import GroupIM
from grouprec.aggregators import WeightedAverageAggregator, EPFuzzDAAggregator
from grouprec.aggregators._normalize import normalize_mgains
from grouprec.datasets.cache import dataset_dir

DATASET = "ml-latest-small"   # redistribution-permitting MovieLens release (titles can be published)

# ---- config --------------------------------------------------------------- #
GROUP_PLAN = [("similar", 3), ("divergent", 3), ("outlier", 3)]   # regime -> count
GROUP_SIZE = 3
N_CAND = 50                 # candidate pool per group (1 consensus positive + negatives)
LIKE = 4                    # rating >= LIKE counts as "liked"
CONSENSUS = 2               # item is a group positive if >= CONSENSUS members liked it
MIN_CONSENSUS_ITEMS = 3     # keep groups with at least this many consensus items
N_HISTORY = 3               # member history items shown (top by rating)
N_SAE_FEATS = 3             # SAE concepts shown per member
N_SAE_EXEMPLARS = 3         # exemplar films shown per concept
GRID_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]     # per-member influence grid (5^3 = 125 combos)
SAE_HIDDEN, SAE_K, SAE_STEPS = 64, 6, 4000

MEMBER_STYLE = [
    {"color": "#5DCAA5", "bg": "#d0f0e4"},
    {"color": "#7F77DD", "bg": "#dddcf8"},
    {"color": "#D85A30", "bg": "#f7d8cc"},
]

ALGORITHMS = [
    {"key": "wAVG", "name": "Weighted average", "family": "AGG", "mode": "score",
     "desc": "Member-importance-weighted mean of members' real EASE scores via grouprec's GroupRecommender.group_scores + WeightedAverageAggregator; the 0–1 value is the group relevance. The influence weight tilts the mean."},
    {"key": "EPFuzzDA", "name": "EP-FuzzDA", "family": "AGG", "mode": "order",
     "desc": "Fairness aggregator (grouprec EPFuzzDAAggregator): greedily picks items to meet each member's target share. The influence weight sets each member's target importance."},
    {"key": "GroupIM", "name": "GroupIM", "family": "E2E", "mode": "score",
     "desc": "Deep model encoding each member's history, pooled by an attention aggregator; a learned predictor scores items. Influence reweights the aggregator (real forward pass)."},
]
SCORE_KEYS = [a["key"] for a in ALGORITHMS if a["mode"] == "score"]      # bar = 0..1 score
ORDER_KEYS = [a["key"] for a in ALGORITHMS if a["mode"] == "order"]      # bar = rank
E2E_KEYS = [a["key"] for a in ALGORITHMS if a["family"] == "E2E"]


def _minmax(v: np.ndarray) -> np.ndarray:
    lo, hi = float(v.min()), float(v.max())
    return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)


def _norm_combo(levels):
    w = np.array(levels, dtype=float)
    s = w.sum()
    return (np.ones_like(w) / len(w)) if s == 0 else (w / s)


# --------------------------------------------------------------------------- #
# MovieLens metadata
# --------------------------------------------------------------------------- #
def load_movielens_metadata():
    """Parse the ml-latest-small ``movies.csv`` (movieId, title, pipe-separated genres).
    Located via the framework cache so it honors GROUPREC_CACHE and the pinned snapshot."""
    mv = next(dataset_dir(DATASET).rglob("movies.csv"))
    titles, gsets, vocab = {}, {}, []
    with open(mv, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)                                            # header
        for mid, title, gstr in reader:
            mid = int(mid)
            titles[mid] = title
            gs = [g for g in gstr.split("|") if g and g != "(no genres listed)"]
            gsets[mid] = gs
            for g in gs:
                if g not in vocab:
                    vocab.append(g)
    genres = sorted(vocab)
    gidx = {g: i for i, g in enumerate(genres)}
    genre_vecs = {}
    for mid, gs in gsets.items():
        v = np.zeros(len(genres), dtype=float)
        for g in gs:
            v[gidx[g]] = 1.0
        genre_vecs[mid] = v
    return genres, titles, genre_vecs


# --------------------------------------------------------------------------- #
# Top-K SAE (adapted from umap2026/sae.py TopKSAE)
# --------------------------------------------------------------------------- #
class TopKSAE(nn.Module):
    def __init__(self, input_dim, hidden, k, l1=1e-3):
        super().__init__()
        self.k, self.l1 = k, l1
        self.enc_w = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(input_dim, hidden)))
        self.enc_b = nn.Parameter(torch.zeros(hidden))
        self.dec_w = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(hidden, input_dim)))
        self.dec_b = nn.Parameter(torch.zeros(input_dim))
        self._normalize_decoder()

    @torch.no_grad()
    def _normalize_decoder(self):
        self.dec_w.data = self.dec_w.data / (self.dec_w.data.norm(dim=-1, keepdim=True) + 1e-8)

    def _standardize(self, x):
        return (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-7)

    def encode(self, x):
        x = self._standardize(x)
        e = F.relu((x - self.dec_b) @ self.enc_w + self.enc_b)
        top = torch.topk(e, self.k, dim=-1)
        return torch.zeros_like(e).scatter(-1, top.indices, top.values)

    def forward(self, x):
        e = self.encode(x)
        return e @ self.dec_w + self.dec_b, e


def fit_topk_sae(item_emb, *, hidden=SAE_HIDDEN, k=SAE_K, steps=SAE_STEPS, seed=0):
    torch.manual_seed(seed)
    X = torch.as_tensor(item_emb, dtype=torch.float32)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    net = TopKSAE(Xs.shape[1], hidden, k)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(steps):
        recon, e = net(Xs)
        loss = (recon - net._standardize(Xs)).pow(2).mean() + net.l1 * e.abs().sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        net._normalize_decoder()
    with torch.no_grad():
        _, Z = net(Xs)
    return Z.numpy()


# --------------------------------------------------------------------------- #
# Real framework calls over the per-member influence grid
# --------------------------------------------------------------------------- #
def build_groupim_grid(model, members, cand):
    """GroupIM scores + per-combo pooling attention over the weight grid via
    ``GroupIM.group_scores(members, cand, member_weights=w, return_attention=True)``.
    Collecting attention per combo (not just at uniform weights) lets the UI derive
    attribution that is consistent with which grid point's items are being shown."""
    grid, att_grid, M = {}, {}, len(members)
    for combo in itertools.product(range(len(GRID_LEVELS)), repeat=M):
        key = "".join(str(c) for c in combo)
        w = list(_norm_combo([GRID_LEVELS[c] for c in combo]))
        s, att = model.group_scores(members, cand, member_weights=w, return_attention=True)
        grid[key] = [round(float(x), 4) for x in _minmax(np.asarray(s, dtype=float))]
        att_grid[key] = [round(float(x), 4) for x in att]
    return grid, att_grid


def build_agg_score_grid(rec, members, cand, agg_factory):
    """Per-item group scores from the real ``GroupRecommender.group_scores`` over the weight
    grid (for score-reduction aggregators like wAVG, which expose ``score_items``). Returns
    combo -> [minmax 0-1 scores], so the UI can show an actual group rating, not just a rank."""
    grid, M = {}, len(members)
    for combo in itertools.product(range(len(GRID_LEVELS)), repeat=M):
        w = _norm_combo([GRID_LEVELS[c] for c in combo])
        rec.aggregator = agg_factory(w)
        s = np.asarray(rec.group_scores(members, items=cand), dtype=float)
        grid["".join(str(c) for c in combo)] = [round(float(x), 4) for x in _minmax(s)]
    return grid


def build_agg_order_grid(rec, members, cand, agg_factory):
    """Candidate ordering from the real ``GroupRecommender`` pipeline over the weight grid
    (for selection-based aggregators like EP-FuzzDA, which return an ordering, not scores).
    ``rec`` carries a pre-fitted base; we only swap in the aggregator per combo."""
    pos = {c: i for i, c in enumerate(cand)}
    grid, M = {}, len(members)
    for combo in itertools.product(range(len(GRID_LEVELS)), repeat=M):
        w = _norm_combo([GRID_LEVELS[c] for c in combo])
        rec.aggregator = agg_factory(w)
        order_ids = rec.recommend(members, k=len(cand), candidates=cand)
        grid["".join(str(c) for c in combo)] = [pos[int(x)] for x in order_ids]
    return grid


# --------------------------------------------------------------------------- #
# main build
# --------------------------------------------------------------------------- #
def build(out: Path) -> None:
    data = gr.datasets.load(DATASET)                 # downloads+extracts (pinned snapshot)
    genres, titles, genre_vecs = load_movielens_metadata()
    ui, ii = data.user_index, data.item_index
    items_arr = data.items
    inter = data.interactions
    # SUBSET RULE (history): top-N_HISTORY items by rating, ties broken by ascending item id (deterministic).
    rated_top = {int(u): sub.sort_values(["rating", "item"], ascending=[False, True])["item"].astype(int).tolist()
                 for u, sub in inter.groupby("user")}
    R = data.user_item_matrix(value="rating")

    def cohesion(mset):
        rows = np.array([R[ui[m]] for m in mset], dtype=float)
        cors = [float(np.corrcoef(rows[a], rows[b])[0, 1])
                for a, b in itertools.combinations(range(len(mset)), 2)
                if rows[a].std() > 0 and rows[b].std() > 0]
        return float(np.mean(cors)) if cors else 0.0

    # ---- groups via the framework's synthetic sampler, one regime at a time ---- #
    print("[groups] generating via gr.groups.synthetic (similar / divergent / outlier) ...", flush=True)
    chosen, group_kinds, group_interactions = [], [], {}
    for kind, n_want in GROUP_PLAN:
        cand_groups = gr.groups.synthetic(data, kind=kind, size=GROUP_SIZE, n=max(60, n_want * 20), seed=0)
        # dedup candidate member-sets, preserving generation order
        uniq, seen = [], set()
        for members in cand_groups:
            key = tuple(sorted(int(x) for x in members))
            if key not in seen:
                seen.add(key); uniq.append(np.array(key, dtype=np.int64))
        # derive each candidate's consensus items through the FRAMEWORK: an item is a group
        # interaction if >= CONSENSUS members liked it (like = rating >= LIKE). Same call a user makes.
        cand_ints = gr.groups.derive_group_interactions(
            data, gr.Groups(uniq, metadata={"kind": kind}),
            like_threshold=LIKE, min_members=CONSENSUS)
        kept = 0
        for i, members in enumerate(uniq):
            cons = cand_ints[i]
            if len(cons) >= MIN_CONSENSUS_ITEMS:
                group_interactions[len(chosen)] = cons
                chosen.append([int(x) for x in members]); group_kinds.append(kind)
                kept += 1
            if kept >= n_want:
                break
        print(f"  [{kind}] kept {kept}/{n_want}")
    cohesions = [cohesion(m) for m in chosen]

    groups = gr.Groups([np.array(m, dtype=np.int64) for m in chosen],
                       metadata={"kind": "synthetic", "source": DATASET})

    # ---- fit recommenders (framework) ---- #
    print("[fit] EASE ...", flush=True)
    ease = EASE(reg=200.0).fit(data)
    # one GroupRecommender with the pre-fitted EASE base; we swap its weighted aggregator per combo
    rec = GroupRecommender(ease, WeightedAverageAggregator(), normalize="minmax")
    rec.dataset_ = data                              # reuse the fitted base (skip refit)
    print("[fit] GroupIM ...", flush=True)
    groupim = GroupIM(groups, group_interactions, epochs=40, pretrain_epochs=10, seed=0).fit(data)

    # ---- SAE on GroupIM item embeddings, labelled by genre ---- #
    # NB: we use the ENCODER's user_predictor rows, not group_predictor. The group_predictor is
    # trained on only |groups| target distributions, so its item space collapses onto popularity
    # (one PC explains ~66% of variance, corr ~-0.68 with item popularity) and every member ends
    # up with the same "concepts". The encoder head is pretrained over *all* users, so it actually
    # carries taste structure.
    print("[fit] Top-K SAE on GroupIM encoder item embeddings ...", flush=True)
    item_emb = groupim.net_.encoder.user_predictor.weight.detach().cpu().numpy()   # (n_items, d)
    Z = fit_topk_sae(item_emb)
    Z_global = Z.mean(0) + 1e-9        # for distinctiveness ("lift") based concept selection
    Gmat = np.array([genre_vecs.get(int(items_arr[idx]), np.zeros(len(genres))) for idx in range(len(items_arr))])

    def feature_label(f):
        # SUBSET RULE (exemplars): top-8 items by SAE activation (stable ties) -> genre label; show top-N_SAE_EXEMPLARS.
        top = np.argsort(-Z[:, f], kind="stable")[:8]
        gsum = Gmat[top].sum(0)
        items = [int(items_arr[t]) for t in top[:N_SAE_EXEMPLARS]]
        if gsum.sum() == 0:
            return "mixed", items
        names = [genres[j] for j in np.argsort(-gsum, kind="stable")[:2] if gsum[j] > 0]
        return " / ".join(names), items

    feat_cache = {f: feature_label(f) for f in range(Z.shape[1])}

    rng = np.random.default_rng(0)
    groups_out = []
    for gslot, mset in enumerate(chosen):
        cons = group_interactions[gslot]
        pos = int(cons[0])
        # SUBSET RULE (candidates): pos consensus item + uniformly-sampled negatives that no member rated (seed gslot).
        member_rated = set()
        for m in mset:
            member_rated.update(rated_top.get(m, ()))
        neg_pool = [int(items_arr[idx]) for idx in range(len(items_arr))
                    if int(items_arr[idx]) not in member_rated and int(items_arr[idx]) != pos]
        grng = np.random.default_rng(gslot)
        negs = list(grng.choice(neg_pool, size=min(N_CAND - 1, len(neg_pool)), replace=False))
        cand = [pos] + [int(x) for x in negs]

        # per-member normalised scores (framework EASE + framework normalizer) -- used for the
        # attribution dots and the per-member "recs received" view.
        scores = np.asarray(ease.score(mset, items=cand), dtype=float)        # (M, C)
        rm = normalize_mgains(scores, "minmax")

        # rankings: real framework calls (GroupRecommender pipeline for the aggregators,
        # GroupIM.group_scores for the deep model) over the 125-point member-weight grid.
        order_grid = {
            "EPFuzzDA": build_agg_order_grid(rec, mset, cand, lambda w: EPFuzzDAAggregator(member_weights=w)),
        }
        wavg_grid = build_agg_score_grid(rec, mset, cand, lambda w: WeightedAverageAggregator(member_weights=w))
        gim_grid, gim_att_grid = build_groupim_grid(groupim, mset, cand)
        score_grid = {"wAVG": wavg_grid, "GroupIM": gim_grid}

        members_out = []
        for mi, m in enumerate(mset):
            hist = [h for h in rated_top.get(m, []) if h in titles][:N_HISTORY]
            hidx = [ii[h] for h in rated_top.get(m, []) if h in ii]
            # SUBSET RULE (concepts): top-N_SAE_FEATS features by *lift* -- the member's mean
            # activation divided by the global mean, i.e. what is distinctive about this member
            # rather than what is globally strongest (stable ties).
            top_feats = (np.argsort(-(Z[hidx].mean(0) / Z_global), kind="stable")[:N_SAE_FEATS].tolist()
                         if hidx else [0, 1, 2])
            sae = [{"label": feat_cache[f][0], "items": feat_cache[f][1]} for f in top_feats]
            members_out.append({
                "uid": m, "label": f"User {m}", "initials": f"U{m}",
                "color": MEMBER_STYLE[mi]["color"], "bg": MEMBER_STYLE[mi]["bg"],
                "history": hist, "sae": sae,
            })

        kind = group_kinds[gslot]
        groups_out.append({
            "label": f"Group {gslot} ({kind}) · users {', '.join(str(m) for m in mset)}",
            "kind": kind, "cohesion": round(cohesions[gslot], 2),
            "members": members_out, "candidates": cand, "pos": pos,
            "agg": [r.round(4).tolist() for r in rm],            # per-member scores (attribution)
            "scoreGrid": score_grid,                              # wAVG + GroupIM: combo -> [C scores]
            "orderGrid": order_grid,                              # EPFuzzDA: combo -> [ordering]
            "attGrid": {"GroupIM": gim_att_grid},                 # GroupIM: combo -> [M att weights]
        })
        print(f"  [group {gslot + 1}/{len(chosen)}] {kind} users={mset} coh={cohesions[gslot]:.2f} consensus={len(cons)}")

    payload = {
        "dataset": "MovieLens latest-small",
        "algorithms": ALGORITHMS, "scoreKeys": SCORE_KEYS, "orderKeys": ORDER_KEYS, "e2eKeys": E2E_KEYS,
        "gridLevels": GRID_LEVELS,
        "titles": {str(it): titles.get(it, f"Item {it}")
                   for g in groups_out
                   for it in (g["candidates"]
                              + [x for mo in g["members"] for x in mo["history"]]
                              + [x for mo in g["members"] for f in mo["sae"] for x in f["items"]])},
        "groups": groups_out,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(payload))
    print(f"\nwrote {out}  ({len(groups_out)} groups, {len(ALGORITHMS)} algorithms, {out.stat().st_size/1e6:.2f} MB)")


def render_html(payload: dict) -> str:
    return _HTML_TEMPLATE.replace("/*__DATA__*/", json.dumps(payload, separators=(",", ":")))


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Group Recommendation Inspector — MovieLens (latest-small)</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  :root{--bg:#f8f7f3;--surface:#fff;--surface1:#f1efe8;--border:rgba(0,0,0,.12);
    --border-strong:rgba(0,0,0,.22);--text:#2c2c2a;--text-sec:#5f5e5a;--text-muted:#888780;
    --accent:#185FA5;--accent-bg:#E6F1FB;--agg:#2f8f5b;--agg-bg:#e2f4ea;--e2e:#7a52c9;--e2e-bg:#efe9fb;
    --bar:#7c8aa0;--pos:#c2891b;--radius:8px;}
  @media (prefers-color-scheme:dark){:root{--bg:#1e1e1c;--surface:#2c2c2a;--surface1:#242422;
    --border:rgba(255,255,255,.1);--border-strong:rgba(255,255,255,.18);--text:#e8e6dc;--text-sec:#b4b2a9;
    --text-muted:#6e6d67;--accent:#85B7EB;--accent-bg:#042C53;--agg:#69c993;--agg-bg:#10301f;--e2e:#b497ec;--e2e-bg:#241636;
    --bar:#8b97a8;--pos:#e0a93a;}}
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:var(--bg);color:var(--text);padding:24px;min-height:100vh}
  h1{font-size:17px;font-weight:600;margin-bottom:4px}
  .subtitle{font-size:12px;color:var(--text-muted);margin-bottom:18px}
  .controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:16px}
  select{font-size:13px;padding:7px 10px;border-radius:var(--radius);border:.5px solid var(--border-strong);background:var(--surface);color:var(--text);min-width:200px}
  .badge{font-size:11px;font-weight:600;padding:3px 9px;border-radius:20px;letter-spacing:.04em}
  .badge.AGG{background:var(--agg-bg);color:var(--agg)}.badge.E2E{background:var(--e2e-bg);color:var(--e2e)}
  .badge.coh{background:var(--surface1);color:var(--text-sec);border:.5px solid var(--border)}
  .btn{font-size:12px;padding:6px 12px;border-radius:var(--radius);border:.5px solid var(--border-strong);background:var(--surface);color:var(--text-sec);cursor:pointer}
  .btn:hover{border-color:var(--accent);color:var(--accent)}
  .algo-desc{font-size:12px;color:var(--text-sec);background:var(--surface1);border:.5px solid var(--border);border-radius:var(--radius);padding:8px 11px;margin-bottom:16px;line-height:1.5}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}
  @media (max-width:760px){.grid{grid-template-columns:1fr}}
  .section-label{font-size:11px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px}
  .tab-row{display:flex;gap:4px;margin-bottom:10px;flex-wrap:wrap}
  .tab-btn{font-size:12px;padding:4px 10px;border-radius:var(--radius);border:.5px solid var(--border-strong);background:transparent;color:var(--text-sec);cursor:pointer}
  .tab-btn.active{background:var(--accent-bg);color:var(--accent);border-color:var(--accent)}
  .note-box{font-size:12px;color:var(--text-sec);background:var(--surface1);border-radius:var(--radius);padding:8px 10px;margin-bottom:10px;border:.5px solid var(--border);line-height:1.5}
  .user-card{background:var(--surface);border:.5px solid var(--border);border-radius:12px;padding:12px 14px;margin-bottom:8px}
  .user-header{display:flex;align-items:center;gap:10px;margin-bottom:8px}
  .avatar{width:34px;height:34px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:600;flex-shrink:0}
  .slider-row{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--text-sec)}
  .slider-row input[type=range]{flex:1;accent-color:var(--accent)}
  .slider-row .val{min-width:34px;text-align:right;color:var(--text)}
  .recs-card{background:var(--surface);border:.5px solid var(--border);border-radius:12px;padding:12px 14px}
  .rec-item{display:flex;align-items:center;gap:10px;padding:9px 0;border-bottom:.5px solid var(--border)}
  .rec-item:last-child{border-bottom:none}
  .rank-badge{width:22px;height:22px;border-radius:50%;background:var(--surface1);border:.5px solid var(--border-strong);display:flex;align-items:center;justify-content:center;font-size:11px;color:var(--text-muted);flex-shrink:0}
  .rec-name{flex:1;min-width:0}
  .rec-name .title{font-size:13px;font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .rec-name .title .pos-tag{color:var(--pos);font-weight:600}
  .rec-name .genre{font-size:11px;color:var(--text-muted)}
  .scoreval{font-size:11px;color:var(--text-muted);min-width:30px;text-align:right;font-variant-numeric:tabular-nums}
  .bar-bg{width:54px;height:6px;background:var(--surface1);border-radius:3px;overflow:hidden}
  .bar-fill{height:100%;border-radius:3px;background:var(--bar);transition:width .35s ease}
  .dots{display:flex;gap:4px;align-items:center;width:46px;justify-content:flex-end}
  .dot{width:9px;height:9px;border-radius:50%;transition:transform .2s,opacity .2s;cursor:default}
  .attr-box{font-size:12px;color:var(--text-sec);background:var(--surface1);border-radius:var(--radius);padding:10px 11px;border:.5px solid var(--border);line-height:1.7}
  .attr-box b{color:var(--text)} .attr-box .mono{font-size:11px;color:var(--text-sec)}
  .attr-note{font-size:11px;color:var(--text-muted);margin-top:8px;line-height:1.45}
  .legend{display:flex;gap:12px;margin-top:10px;flex-wrap:wrap}
  .legend-item{display:flex;align-items:center;gap:5px;font-size:11px;color:var(--text-muted)}
  .legend-dot{width:10px;height:10px;border-radius:50%}
  .chart-wrap{margin-top:14px;background:var(--surface);border:.5px solid var(--border);border-radius:12px;padding:12px 14px}
  .tip{position:fixed;pointer-events:none;background:var(--text);color:var(--bg);font-size:11px;padding:4px 7px;border-radius:5px;opacity:0;transition:opacity .12s;z-index:10;white-space:nowrap}
  .viewtabs{display:flex;gap:6px;margin-bottom:16px}
  .viewtabs .tab-btn{font-size:13px;padding:6px 14px}
  .about{max-width:860px}
  .about h3{font-size:13px;margin:18px 0 6px;color:var(--text)}
  .about p,.about li{font-size:12.5px;color:var(--text-sec);line-height:1.6}
  .about ol,.about ul{margin:4px 0 4px 18px}
  .about code{background:var(--surface1);padding:1px 4px;border-radius:4px;font-size:11.5px}
  .about pre{background:var(--surface1);border:.5px solid var(--border);border-radius:8px;padding:10px 12px;overflow:auto;font-size:11px;line-height:1.5;margin:6px 0}
  .about table{border-collapse:collapse;font-size:12px;margin:6px 0}
  .about th,.about td{border:.5px solid var(--border);padding:5px 9px;text-align:left;vertical-align:top}
  .about th{background:var(--surface1)}
</style>
</head>
<body>
<h1>Group recommendation inspector</h1>
<p class="subtitle">Real MovieLens (latest-small) · groups from <code>gr.groups.synthetic</code> (similar / divergent / outlier) with consensus-derived interactions · each slider sets a member's <b>relative importance</b> in 5 discrete steps (each a precomputed output); the % shown is the resulting <b>share</b>, so it depends on the other members</p>

<div class="viewtabs">
  <button class="tab-btn active" id="vt-inspector" onclick="setView('inspector')">Inspector</button>
  <button class="tab-btn" id="vt-about" onclick="setView('about')">How it's built</button>
</div>

<div id="view-inspector">
<div class="controls">
  <select id="algo-select" onchange="onAlgo()"></select>
  <span id="algo-badge" class="badge"></span>
  <select id="group-select" onchange="onGroup()"></select>
  <span id="coh-badge" class="badge coh"></span>
  <button class="btn" onclick="resetWeights()">Reset weights</button>
</div>
<div class="algo-desc" id="algo-desc"></div>

<div class="grid">
  <div>
    <p class="section-label">Group members</p>
    <div class="note-box" id="method-note"></div>
    <div id="users-container"></div>
    <div style="margin-top:14px">
      <p class="section-label">Attribution view</p>
      <div class="tab-row" id="attr-tabs">
        <button class="tab-btn active" data-v="history" onclick="setAttrView('history')">Item history</button>
        <button class="tab-btn" data-v="received" onclick="setAttrView('received')">Recs received</button>
        <button class="tab-btn" data-v="latent" onclick="setAttrView('latent')">Latent concepts</button>
      </div>
      <div class="attr-box" id="attr-detail"></div>
      <div class="attr-note" id="attr-note"></div>
    </div>
    <div class="legend" id="legend"></div>
  </div>
  <div>
    <p class="section-label">Group recommendations (top 5 of <span id="cand-n"></span> candidates) · <span id="score-label" style="text-transform:none;color:var(--text-sec)"></span></p>
    <div class="recs-card" id="recs-container"></div>
    <p class="attr-note"><span id="score-legend"></span> · dots = each user's share of that item's relevance (hover for %) · <span style="color:var(--pos)">amber ✓</span> = the group's consensus item (rated ≥4 by ≥2 members)</p>
    <div class="chart-wrap">
      <p class="section-label">Aggregate user contribution across the top 5</p>
      <div style="position:relative;height:90px"><canvas id="contrib-chart"></canvas></div>
    </div>
  </div>
</div>
</div><!-- /view-inspector -->

<div id="view-about" class="about" style="display:none">
  <p>This page is a <b>case study</b> for the <b>grouprec</b> toolkit: a group recommendation pipeline
  built end-to-end from the library's public API. <b>Everything shown is real</b> — no mock score
  matrices — computed <b>offline</b> once and then <b>baked into this single self-contained file</b>
  (no server, no API calls at view time).</p>

  <h3>1 · The real framework calls</h3>
  <p>Every <em>ranking</em> you see is produced by a real <code>grouprec</code> call — the
  aggregators through <code>GroupRecommender</code>, the deep model through
  <code>GroupIM.group_scores</code> (both steered by the per-member weights):</p>
  <pre>data   = gr.datasets.load("ml-latest-small")
groups = gr.groups.synthetic(data, kind="divergent", size=3, n=3)
gints  = gr.groups.derive_group_interactions(data, groups)   # per-group signal (majority, overridable)
ease   = EASE(reg=200.0).fit(data)

# results-aggregation (utilitarian / fairness), steered by member weights:
rec = GroupRecommender(ease, EPFuzzDAAggregator(member_weights=w)); rec.dataset_ = data
rec.recommend(members, k=5, candidates=cands)

# profile-aggregation deep model, steered by member weights:
gim = GroupIM(groups, gints).fit(data)
gim.group_scores(members, cands, member_weights=w)</pre>
  <p><b>This is the core part, not the whole script.</b> The full generator
  (<code>scripts/build_inspector.py</code>, ~400 lines) wraps those calls with: parsing MovieLens
  <code>movies.csv</code> for titles/genres; candidate sampling; a Top-K sparse autoencoder for the
  latent concepts (adapted from an external repo — <em>not</em> part of grouprec); and baking a
  125-point member-weight grid into JSON. So the framework does the recommendation <em>and</em> the
  group-interaction derivation; the script does data prep, the SAE explanation, and packaging.</p>

  <h3>2 · How the data is shipped (offline → one file)</h3>
  <ol>
    <li>Fit the recommenders; for each group call the framework over a 125-point member-weight grid
        (<code>GroupRecommender.recommend</code> orderings for the aggregators, real
        <code>GroupIM.group_scores</code> forward passes for the deep model), plus the SAE concepts
        and the displayed subsets.</li>
    <li>Assemble one Python <code>dict</code> and serialise it with <code>json.dumps</code>.</li>
    <li>Substitute that JSON into a placeholder (<code>const DATA = /*…*/;</code>) in an HTML template
        → one self-contained file (~1&nbsp;MB; the JSON is a single long line, which is what makes the
        file large — the generation logic itself is ~400 lines).</li>
    <li>In your browser, the page reads <code>DATA</code> and renders/re-ranks entirely client-side;
        a slider just looks up the nearest precomputed grid point.</li>
  </ol>

  <h3>3 · Synthetic groups</h3>
  <p>Movielens latest small was chosen due to high familiarity of the items, however, no groups were available in the dataset => synthetic groups were generated.</p>
  <p>Membership comes from <code>gr.groups.synthetic</code> in three regimes — <b>similar</b>,
  <b>divergent</b>, <b>outlier</b> (three each), badged with the measured mean pairwise rating
  correlation. A group's interactions are <b>derived, not simulated</b>, by the framework call
  <code>gr.groups.derive_group_interactions</code>: by default an item is a group's
  <b>consensus item</b> if rated ≥4 by ≥2 of the 3 members — a deterministic function of the real
  ratings — and the rule is overridable with a predicate (e.g. unanimity, or "any member").</p>

  <h3>4 · Algorithms &amp; faithful interactivity</h3>
  <p>The slider is each member's importance weight (normalised to 100%). Outputs are precomputed on a
  <code>{0,¼,½,¾,1}³ = 125</code>-point grid and the slider snaps to the nearest point, so every
  ranking is a genuine output:</p>
  <ul>
    <li><b>Weighted average</b> — <code>GroupRecommender</code> + grouprec
        <code>WeightedAverageAggregator(member_weights=w)</code>.</li>
    <li><b>EP-FuzzDA</b> — <code>GroupRecommender</code> + grouprec
        <code>EPFuzzDAAggregator(member_weights=w)</code> (fairness; the weight is each member's target share).</li>
    <li><b>GroupIM</b> — the framework's <code>GroupIM.group_scores(members, items, member_weights=w)</code>,
        a real forward pass that reweights the model's attention pooling
        (<code>α'<sub>m</sub> ∝ w<sub>m</sub>·α<sub>m</sub></code>), reducing to the native model at equal
        weights. Steering is in the model's pooling — <b>not</b> via the SAE.</li>
  </ul>

  <h3>5 · Latent concepts (how the SAE was adopted)</h3>
  <p>We reuse a <b>Top-K sparse autoencoder</b> (standardised input, unit-norm decoder, ReLU encoder,
  hard top-K=6, MSE + small L1) and fit it on <b>GroupIM's item embeddings</b> — specifically the rows
  of its <code>encoder.user_predictor</code> weight, one vector per movie. Each latent feature is
  <b>named by the dominant genre</b> of its top-activating movies, turning opaque dimensions into
  readable concepts. A member is tagged with a concept because the films they watched activate it; the
  films listed are the concept's global <b>exemplars</b>, so they may differ from the member's own
  history. The SAE is only an explanation of members.</p>
  <p><b>Why the encoder head, and why "lift".</b> Fitting the SAE on <code>group_predictor</code>
  instead gave <em>every</em> member the same concepts: that head is trained on only as many target
  distributions as there are groups, so its item space collapses onto popularity (one direction
  explained ~66% of the variance, correlating ≈&minus;0.68 with item popularity) and the SAE simply
  decomposed <em>popularity</em>. The <code>encoder.user_predictor</code> head is pretrained over
  <b>all users</b>, so it carries real taste structure. Independently, ranking a member's concepts by
  raw mean activation returns whatever is globally strongest, so we rank by <b>lift</b> — the member's
  mean activation divided by the global mean, i.e. what is <em>distinctive</em> about that member.</p>

  <h3>6 · Every displayed subset is deterministic</h3>
  <table>
    <tr><th>Subset</th><th>Rule (ties: ascending index, <code>argsort kind="stable"</code>)</th></tr>
    <tr><td>Member history (3)</td><td>top-3 by rating; ties by item id</td></tr>
    <tr><td>Candidate pool (50)</td><td>consensus positive + 49 negatives sampled uniformly (per-group seed) from items no member rated</td></tr>
    <tr><td>Recs received (3)</td><td>member's top-3 candidates by EASE score</td></tr>
    <tr><td>Top-5 recs</td><td>top-5 by group score / selection order</td></tr>
    <tr><td>Latent concepts (3)</td><td>SAE features with the highest <em>lift</em> over the member's items (member mean activation ÷ global mean, i.e. most distinctive)</td></tr>
    <tr><td>Concept exemplars (3)</td><td>items with highest activation for the feature</td></tr>
  </table>
  <p style="margin-top:14px">Reproduce with <code>python scripts/build_inspector.py</code>. Full write-up:
  <code>docs/INSPECTOR.md</code> in the repository.</p>
</div>

<div class="tip" id="tip"></div>
<p style="font-size:11px;color:var(--text-muted);margin-top:20px;line-height:1.5">
  Built with <b>grouprec</b>. Data: the <a href="https://grouplens.org/datasets/movielens/" style="color:var(--accent)">MovieLens</a>
  latest-small dataset (F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets. ACM TiiS),
  courtesy of GroupLens, used under its usage license (redistribution permitted under the same terms).
</p>

<script>
const DATA = /*__DATA__*/;
const ALGOS = DATA.algorithms, LEVELS = DATA.gridLevels, TITLES = DATA.titles;
const SCOREK = DATA.scoreKeys, ORDERK = DATA.orderKeys;
// Outputs are precomputed only at the grid levels, so the sliders must move in those same
// steps -- otherwise the handle/label imply a granularity the results cannot reflect.
const STEP = 100 / (LEVELS.length - 1);
const tip = document.getElementById('tip');
let algoKey = ALGOS[0].key, groupIdx = 0, attrView = 'history', weights = [], chart = null;

function algo(){ return ALGOS.find(a => a.key === algoKey); }
function group(){ return DATA.groups[groupIdx]; }
function fam(){ return algo().family; }
function isE2E(){ return fam() === 'E2E'; }
function isOrder(){ return ORDERK.indexOf(algoKey) >= 0; }
function title(id){ return TITLES[id] || ('Item ' + id); }
function comboKey(){ return weights.map(w=>{let bi=0,bd=9;LEVELS.forEach((L,i)=>{const d=Math.abs(L-w);if(d<bd){bd=d;bi=i;}});return bi;}).join(''); }
function eqKey(){ return group().members.map(()=>{let bi=0,bd=9;LEVELS.forEach((L,i)=>{const d=Math.abs(L-0.5);if(d<bd){bd=d;bi=i;}});return bi;}).join(''); }

function initControls(){
  document.getElementById('algo-select').innerHTML = ALGOS.map(a=>`<option value="${a.key}">${a.name} [${a.family}]</option>`).join('');
  document.getElementById('group-select').innerHTML = DATA.groups.map((g,i)=>`<option value="${i}">${g.label}</option>`).join('');
}
function onAlgo(){ algoKey = document.getElementById('algo-select').value; renderAll(); }
function onGroup(){ groupIdx = +document.getElementById('group-select').value; resetWeights(); }
function resetWeights(){ weights = group().members.map(()=>0.5); renderAll(); }
function normWeights(){ const t = weights.reduce((a,b)=>a+b,0)||1; return weights.map(w=>w/t); }

// Attribution uses the same snapped effective weights as the displayed items, so dots and
// items are always consistent (both update at the same grid boundaries, not independently).
function effectiveWeights(){
  const ck=comboKey(), raw=Array.from(ck).map(c=>LEVELS[parseInt(c)]);
  const s=raw.reduce((a,b)=>a+b,0);
  return s===0 ? group().members.map(()=>1/group().members.length) : raw.map(w=>w/s);
}
function computeScores(){
  const g = group(), M = g.members.length, C = g.candidates.length, ew = effectiveWeights();
  let score = new Array(C).fill(0);
  let contrib = Array.from({length:M},()=>new Array(C).fill(0));
  const S = g.agg;                                  // M x C normalised per-member scores
  if(!isE2E()){
    for(let m=0;m<M;m++) for(let c=0;c<C;c++) contrib[m][c]=ew[m]*S[m][c];
  }
  if(SCOREK.indexOf(algoKey)>=0){                   // score-based (wAVG, GroupIM)
    const vec = g.scoreGrid[algoKey][comboKey()] || g.scoreGrid[algoKey][eqKey()];
    for(let c=0;c<C;c++) score[c]=vec[c];
    if(isE2E()){
      // attGrid stores per-combo pooling weights (accurate); fall back to ew*att if absent
      const ck=comboKey(), ag=g.attGrid||{}, attW=(ag[algoKey]||{})[ck]||(ag[algoKey]||{})[eqKey()]||null;
      if(attW){ for(let c=0;c<C;c++) for(let m=0;m<M;m++) contrib[m][c]=attW[m]; }
      else if(g.att){ const att=g.att[algoKey]; for(let c=0;c<C;c++) for(let m=0;m<M;m++) contrib[m][c]=ew[m]*att[m][c]; }
    }
  } else {                                          // order-based (EP-FuzzDA)
    const ord = g.orderGrid[algoKey][comboKey()] || g.orderGrid[algoKey][eqKey()];
    for(let r=0;r<ord.length;r++) score[ord[r]] = (ord.length - r);
  }
  const rows=[];
  for(let c=0;c<C;c++){
    const tot = contrib.reduce((a,r)=>a+Math.max(0,r[c]),0)||1;
    rows.push({c, item:g.candidates[c], score:score[c], shares:contrib.map(r=>Math.max(0,r[c])/tot)});
  }
  rows.sort((a,b)=>b.score-a.score);
  const top=rows.slice(0,5), mx=Math.max(...top.map(r=>r.score)), mn=Math.min(...top.map(r=>r.score));
  rows.forEach((r,idx)=>{ r.rank=idx; r.bar = mx>mn ? (r.score-mn)/(mx-mn) : 1; });
  return rows;
}

function renderControls(){
  document.getElementById('algo-select').value=algoKey;
  document.getElementById('group-select').value=groupIdx;
  const a=algo(), b=document.getElementById('algo-badge');
  b.className='badge '+a.family; b.textContent=a.family;
  document.getElementById('algo-desc').textContent=a.desc;
  document.getElementById('coh-badge').textContent=`members: ${group().kind} (r=${group().cohesion})`;
  document.getElementById('method-note').textContent = isE2E()
    ? "GroupIM (deep): the influence weight is injected into the model's own attention aggregator and precomputed on a grid — NOT SAE steering. Each step is a real forward pass; equal weights = the native model."
    : "Aggregation: the influence weight is each member's importance. Weighted-average = grouprec WeightedAverageAggregator; EP-FuzzDA is the real fairness aggregator run with these member weights. (Weight-agnostic rules like least-misery are excluded — they ignore weights.)";
  document.getElementById('cand-n').textContent=group().candidates.length;
  // persistent (not hover-only) description of what each item's number means
  const scoreLbl = isOrder()
    ? 'number = selection rank (#1 = picked first)'
    : 'number = group relevance score (0–1)';
  document.getElementById('score-label').textContent = scoreLbl;
  document.getElementById('score-legend').textContent = scoreLbl.charAt(0).toUpperCase() + scoreLbl.slice(1);
}
// Labels show the EFFECTIVE (grid-snapped) weights, so the percentage always matches the
// weights the displayed ranking was actually computed with.
function updateWeightLabels(){ const wn=effectiveWeights(); group().members.forEach((u,i)=>{const el=document.getElementById('wlab-'+i); if(el) el.textContent=Math.round(wn[i]*100)+'%';}); }
function renderUsers(){
  const g=group(), wn=effectiveWeights();
  document.getElementById('users-container').innerHTML=g.members.map((u,i)=>`
    <div class="user-card"><div class="user-header">
      <div class="avatar" style="background:${u.bg};color:${u.color}">${u.initials}</div>
      <div><div style="font-size:13px;font-weight:500">${u.label}</div>
           <div style="font-size:11px;color:var(--text-muted)">top: ${u.history.map(title).join(' · ')||'—'}</div></div>
      <div style="margin-left:auto;font-size:12px;color:var(--text-sec)" id="wlab-${i}">${Math.round(wn[i]*100)}%</div>
    </div><div class="slider-row"><span style="font-size:11px;color:var(--text-muted)">influence</span>
      <input type="range" min="0" max="100" value="${Math.round(weights[i]*100)}" step="${STEP}"
        title="Snaps to the ${LEVELS.length} precomputed levels (${LEVELS.join(', ')}) -- every position is a real, precomputed output"
        oninput="weights[${i}]=this.value/100;updateWeightLabels();renderRecs();renderChart()">
    </div></div>`).join('');
}
function renderRecs(){
  const g=group(), rows=computeScores().slice(0,5), ord=isOrder();
  document.getElementById('recs-container').innerHTML=rows.map((r,rank)=>{
    const dots=g.members.map((u,i)=>{const sh=r.shares[i],sc=.55+sh*1.0,op=.25+sh*.75;
      return `<div class="dot" style="background:${u.color};opacity:${op};transform:scale(${sc})"
        onmousemove="showTip(event,'${u.label}: '+Math.round(${sh}*100)+'%')" onmouseleave="hideTip()"></div>`;}).join('');
    const isPos=r.item===g.pos;
    const val = ord ? ('#'+(rank+1)) : r.score.toFixed(2);
    const posTip='Consensus item — rated ≥4 by ≥2 of the 3 members; the historical signal used to build this group, not a model prediction';
    const valTip = ord
      ? `Selection rank #${rank+1} — EP-FuzzDA balances members by selecting items, so it ranks rather than assigning a 0–1 score`
      : `Group relevance score: ${r.score.toFixed(3)} (higher = better, 0–1)`;
    const posAttr = isPos ? ` onmousemove="showTip(event,'${posTip}')" onmouseleave="hideTip()" style="cursor:help"` : '';
    const valAttr = ` onmousemove="showTip(event,'${valTip}')" onmouseleave="hideTip()" style="cursor:help"`;
    return `<div class="rec-item"><div class="rank-badge">${rank+1}</div>
      <div class="rec-name"><div class="title">${title(r.item)}${isPos?` <span class="pos-tag"${posAttr}>✓</span>`:''}</div>
        <div class="genre"${posAttr}>${isPos?'consensus item':'candidate'}</div></div>
      <span class="scoreval"${valAttr}>${val}</span>
      <div class="bar-bg"${valAttr}><div class="bar-fill" style="width:${Math.round(r.bar*100)}%"></div></div>
      <div class="dots">${dots}</div></div>`;
  }).join('');
}
function renderChart(){
  const g=group(), rows=computeScores().slice(0,5);
  const totals=g.members.map((_,i)=>rows.reduce((a,r)=>a+r.shares[i],0));
  const tot=totals.reduce((a,b)=>a+b,0)||1, data=totals.map(t=>+(100*t/tot).toFixed(1));
  const ds=g.members.map((u,i)=>({label:u.label,data:[data[i]],backgroundColor:u.color,borderWidth:0}));
  if(chart){ chart.data.datasets=ds; chart.update('none'); return; }
  chart=new Chart(document.getElementById('contrib-chart'),{type:'bar',
    data:{labels:['top-5 contribution'],datasets:ds},
    options:{indexAxis:'y',responsive:true,maintainAspectRatio:false,
      scales:{x:{stacked:true,max:100,ticks:{callback:v=>v+'%',color:'#888'},grid:{display:false}},y:{stacked:true,grid:{display:false},ticks:{display:false}}},
      plugins:{legend:{position:'bottom',labels:{boxWidth:10,font:{size:11}}},tooltip:{callbacks:{label:c=>c.dataset.label+': '+c.raw+'%'}}}}});
}
function setAttrView(v){ attrView=v; document.querySelectorAll('#attr-tabs .tab-btn').forEach(b=>b.classList.toggle('active',b.dataset.v===v)); renderAttr(); }
function renderAttr(){
  const g=group(), el=document.getElementById('attr-detail'), note=document.getElementById('attr-note');
  note.textContent='';
  if(attrView==='history'){
    el.innerHTML=g.members.map(u=>`<b>${u.label}:</b> <span class="mono">${u.history.map(title).join(', ')||'—'}</span>`).join('<br>');
    note.textContent='Each member\'s top-3 films by rating (ties broken by item id), from the real MovieLens ratings.';
  } else if(attrView==='received'){
    el.innerHTML=g.members.map((u,i)=>{
      const order=[...g.agg[i].keys()].sort((a,b)=>g.agg[i][b]-g.agg[i][a]);
      return `<b>${u.label}:</b> <span class="mono">${order.slice(0,3).map(c=>title(g.candidates[c])).join(', ')}</span>`;
    }).join('<br>');
    note.textContent='Each member\'s own top-3 candidates from the EASE base recommender, before any group merging.';
  } else {
    el.innerHTML=g.members.map(u=>{
      const f=u.sae.map(s=>`<b style="color:var(--agg)">${s.label}</b> <span class="mono">[${s.items.map(title).join(', ')}]</span>`).join('<br>&nbsp;&nbsp;');
      return `<b>${u.label}:</b><br>&nbsp;&nbsp;${f}`;
    }).join('<br>');
    note.textContent = isE2E()
      ? 'The 3 concepts most distinctive of each member (highest lift = their mean activation over their history / the global mean), from a Top-K SAE over GroupIM\'s encoder item embeddings. These explain member taste profiles — the SAE is not involved in the model\'s attention-based steering.'
      : 'The 3 concepts most distinctive of each member (highest lift = their mean activation over their history / the global mean), from a Top-K SAE over item embeddings, labelled by the genre of each concept\'s top items. The films listed are the concept\'s exemplars, so they may differ from the member\'s own history.';
  }
}
function renderLegend(){
  document.getElementById('legend').innerHTML=group().members.map(u=>
    `<div class="legend-item"><div class="legend-dot" style="background:${u.color}"></div>${u.label}</div>`).join('');
}
function showTip(e,t){ tip.textContent=t; tip.style.opacity=1; const w=tip.offsetWidth;
  let x=e.clientX+12; if(x+w>window.innerWidth-8) x=e.clientX-w-12;
  tip.style.left=Math.max(4,x)+'px'; tip.style.top=(e.clientY+12)+'px'; }
function hideTip(){ tip.style.opacity=0; }
function renderAll(){ renderControls(); renderUsers(); renderRecs(); renderChart(); renderAttr(); renderLegend(); }
function setView(v){
  document.getElementById('view-inspector').style.display = v==='inspector' ? '' : 'none';
  document.getElementById('view-about').style.display = v==='about' ? '' : 'none';
  document.getElementById('vt-inspector').classList.toggle('active', v==='inspector');
  document.getElementById('vt-about').classList.toggle('active', v==='about');
}
initControls(); resetWeights();
</script>
</body>
</html>
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="docs/group_rec_inspector.html")
    a = p.parse_args()
    build(Path(a.out))


if __name__ == "__main__":
    main()
