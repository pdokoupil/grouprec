"""Parser + fetcher for the Yin et al. group benchmark (Yelp-LA, Douban-SH).

Source: https://sites.google.com/view/hongzhi-yin/datasets (one Google-Drive zip with
``yelp_la/`` and ``douban_sh/``). No explicit license is given, so we treat it as
**non-commercial research use** (like KGRec): ``fetch_yin(accept_license=True)``
surfaces the terms and the required citations (Yin et al. ICDE'18 + ICDE'19).

Files per subset: ``groupid_users.dat`` (gid\\tu1,u2,...), ``groupid_events.dat``
(gid\\te1,e2,...), ``user_events.dat`` (uid\\te1,...). There is no pre-split, so we
hold out one event per group as the test positive and sample N negatives.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..data import Dataset
from .cache import dataset_dir, extract
from .consrec import GroupBenchmarkData

_GDRIVE_ID = "171SnzhoPR9CnCg36ZQZZHl2UVQP9hyoF"
_LICENSE = ("No explicit license; provided for research on group recommendation. "
            "Treat as non-commercial. Cite Yin et al. ICDE'18 (10.1109/ICDE.2018.00088) "
            "and ICDE'19 (10.1109/ICDE.2019.00057).")


def _read_adj(path: Path) -> dict:
    out: dict = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            key, _, rest = line.partition("\t")
            out[key] = [x for x in rest.split(",") if x != ""]
    return out


def fetch_yin(*, accept_license: bool = False, cache=None) -> Path:
    """Download + extract the Yin zip to the cache; returns the extracted dir.

    Requires ``accept_license=True`` (non-commercial; cite ICDE'18 + '19)."""
    if not accept_license:
        raise RuntimeError(
            "Yin Yelp/Douban: " + _LICENSE + "\nRe-run with accept_license=True to download.")
    try:
        import gdown
    except ImportError as exc:
        raise ImportError("fetching the Yin datasets needs gdown: pip install gdown") from exc
    ddir = Path(cache) if cache else dataset_dir("yin")
    zip_path = ddir / "yin.zip"
    if not (ddir / "yelp_la").exists():
        if not zip_path.exists():
            gdown.download(id=_GDRIVE_ID, output=str(zip_path), quiet=True)
        extract(zip_path, ddir)
    print(f"[grouprec] Yin datasets: {_LICENSE}")
    return ddir


def load_yin(path, which: str = "yelp", *, n_negatives: int = 100, seed: int | None = 0,
             min_group_events: int = 2) -> GroupBenchmarkData:
    """Load ``yelp_la`` (``which='yelp'``) or ``douban_sh`` (``which='douban'``).

    ``path`` is the extracted Yin dir (containing ``yelp_la/`` & ``douban_sh/``) or a
    subset dir directly.
    """
    root = Path(path)
    sub = "yelp_la" if which == "yelp" else "douban_sh"
    base = root / sub if (root / sub).exists() else root

    g_users = _read_adj(base / "groupid_users.dat")
    g_events = _read_adj(base / "groupid_events.dat")
    u_events = _read_adj(base / "user_events.dat")

    gids = sorted(g_users, key=lambda x: int(x) if x.isdigit() else x)
    members = [g_users[g] for g in gids]

    # user-item interactions DataFrame (string ids preserved)
    import pandas as pd
    rows = [(u, it) for u, items in u_events.items() for it in items]
    ui_df = pd.DataFrame(rows, columns=["user", "item"])
    all_items = set(ui_df["item"]) | {it for items in g_events.values() for it in items}
    all_users = set(ui_df["user"]) | {u for m in members for u in m}
    dataset = Dataset(ui_df, name=which, users=sorted(all_users), items=sorted(all_items))

    rng = np.random.default_rng(seed)
    items_arr = np.array(sorted(all_items), dtype=object)
    group_interactions: dict[int, list] = {}
    test_instances: list[tuple[int, object, list]] = []
    for gi, gid in enumerate(gids):
        events = g_events.get(gid, [])
        if len(events) < min_group_events:
            group_interactions[gi] = list(events)
            continue
        held = events[rng.integers(0, len(events))]
        group_interactions[gi] = [e for e in events if e != held]
        forbidden = set(events)
        target = min(n_negatives, items_arr.size - len(forbidden))
        negs, attempts = [], 0
        while len(negs) < target and attempts < 50 * max(target, 1):
            cand = items_arr[rng.integers(0, items_arr.size)]
            attempts += 1
            if cand not in forbidden:
                negs.append(cand); forbidden.add(cand)
        test_instances.append((gi, held, negs))

    from .consrec import GroupBenchmarkData as _GBD
    from ..data import Groups
    return _GBD(dataset, Groups(members, metadata={"kind": "inferred", "source": which}),
                group_interactions, test_instances, which)


__all__ = ["load_yin", "fetch_yin"]
