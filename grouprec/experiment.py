"""Reproducibility: an Experiment that records config + environment + code state and
emits a copy-pastable reproduction snippet.

    with gr.Experiment("camra-bridge", seed=42) as exp:
        res = gr.benchmark(...)          # results + citations recorded automatically
    # on exit -> runs/camra-bridge-<ts>/ with config/env/citations/patch/results

What it captures (so a run is reconstructable):
* **seed** + user **params** (whatever you pass). Entering the block seeds Python /
  NumPy / torch, so a run inside ``with`` is seeded whether or not you remember to.
* **results**: a :func:`~grouprec.bench.benchmark` call inside the block attaches its
  leaderboard and the citations for what it ran. Use ``exp.log(hr=0.62)`` /
  ``exp.attach("name", frame)`` for anything computed some other way.
* **environment**: python/platform, CPU count, hostname, and versions of
  grouprec/numpy/scipy/pandas/torch/implicit/lenskit.
* **code state** (git): commit SHA, branch, dirty flag, and the **full** working-tree
  diff (a real patch via :func:`save_patch`) so an uncommitted run reproduces exactly.
* **citations**: pass ``cite=[...]`` and ``exp.citations()`` returns the BibTeX.
"""

from __future__ import annotations

import json
import os
import platform
import random
import re
import subprocess
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path


#: The Experiment whose ``with`` block we are inside, if any. Set by ``__enter__`` so
#: gr.benchmark can attach its results without the caller wiring them up by hand.
_ACTIVE: ContextVar = ContextVar("grouprec_active_experiment", default=None)


def active() -> "Experiment | None":
    """The innermost :class:`Experiment` currently open, or ``None``."""
    return _ACTIVE.get()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and (if present) torch for reproducible runs."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def environment() -> dict:
    """Versions + machine info for the reproduction record."""
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": platform.node(),
        "cpu_count": os.cpu_count(),
    }
    for mod in ("grouprec", "numpy", "scipy", "pandas", "torch", "implicit", "lenskit"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            pass
    return env


def git_info() -> dict:
    """Commit SHA, branch, dirty flag, and the **full** working-tree diff.

    The diff is the complete ``git diff HEAD`` (a real patch), never truncated, so a
    run reproduces exactly via ``git checkout <sha> && git apply <patch>``. Save it to
    a ``.patch`` file with :func:`save_patch` rather than inlining huge text elsewhere.
    """
    def _run(args):
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL,
                                       text=True)
    try:
        sha = _run(["rev-parse", "HEAD"]).strip()
    except Exception:
        return {"available": False}
    info = {"available": True, "sha": sha}
    try:
        info["branch"] = _run(["rev-parse", "--abbrev-ref", "HEAD"]).strip()
        diff = _run(["diff", "HEAD"])              # full patch, untruncated
        info["dirty"] = bool(diff.strip())
        if info["dirty"]:
            info["diff"] = diff
    except Exception:
        pass
    return info


def save_patch(path, info: dict | None = None) -> bool:
    """Write the captured working-tree diff to ``path`` as a ``.patch``. Returns
    ``True`` if a patch was written (i.e. the tree was dirty)."""
    info = info if info is not None else git_info()
    diff = info.get("diff")
    if not diff:
        return False
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(diff)
    return True


def _slug(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "-", name).strip("-") or "run"


class Experiment:
    """Records an experiment's parameters, environment, and code state, and (as a
    context manager) writes a self-contained per-run folder.

    Manual::

        exp = gr.Experiment("camra-bridge", seed=42, cite=["GFAR"])
        gr.set_seed(exp.seed)
        ...
        exp.finalize()                       # writes runs/camra-bridge-<ts>/

    Context manager (the ergonomic path) -- finalizes on exit::

        with gr.Experiment("camra-bridge", seed=42, cite=["GFAR"]) as exp:
            gr.set_seed(exp.seed)
            res = gr.benchmark(...)
            exp.attach("leaderboard", res)   # CSV; or exp.log(hr=0.62)

    Output folder (default ``runs/<name>-<timestamp>/``, override with ``dir=``):
    ``config.json``, ``env.json``, ``citations.bib``, ``results.json``, attached
    ``*.csv``, and ``code.patch`` (the full uncommitted diff) when in a git repo.
    Works both in a clone of this repo (captures git SHA + patch) and when installed
    from PyPI (no git -- records ``grouprec`` version, entry script, and cwd instead).
    """

    def __init__(self, name: str, *, seed: int = 0, dir=None, cite=None, **params) -> None:
        self.name = name
        self.seed = seed
        self._dir = Path(dir) if dir is not None else None
        # cite items may be citation **keys** (str) or **objects** actually used
        # (aggregator / recommender / model / dataset), auto-resolved to keys.
        self._cite_items = list(cite) if cite else []
        self.params = params
        self.created = datetime.now(timezone.utc).isoformat()
        self.env = environment()
        self.git = git_info()
        self.entry_script = sys.argv[0] if sys.argv and sys.argv[0] else None
        self.cwd = os.getcwd()
        self.results: dict = {}
        self._frames: dict = {}

    @property
    def cite(self) -> list:
        """Resolved citation **keys** (order-stable, deduped) from keys + objects."""
        from .references import citation_keys_for, has
        out: list = []
        for item in self._cite_items:
            keys = [item] if isinstance(item, str) else sorted(citation_keys_for(item))
            for k in keys:
                if k not in out and (isinstance(item, str) or has(k)):
                    out.append(k)
        return out

    def add_citations(self, *objs) -> "Experiment":
        """Auto-collect citations from objects used in the run (e.g.
        ``exp.add_citations(recommender, group_data)``); manual keys still work."""
        self._cite_items.extend(objs)
        return self

    # -- context manager ---------------------------------------------------- #
    def __enter__(self) -> "Experiment":
        set_seed(self.seed)          # so "seeded" is a property of the block, not of discipline
        self._token = _ACTIVE.set(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        _ACTIVE.reset(self._token)
        if exc is not None:
            self.results.setdefault("error", f"{exc_type.__name__}: {exc}")
        self.finalize()
        return False  # never suppress exceptions

    # -- attach results ----------------------------------------------------- #
    def log(self, **kv) -> "Experiment":
        """Record scalar results / notes (merged into ``results.json``)."""
        self.results.update(kv)
        return self

    def attach(self, name: str, obj) -> "Experiment":
        """Attach a table (a DataFrame or anything with ``to_frame()``, e.g. a
        ``BenchmarkResult``/``Report``) -- written as ``<name>.csv`` on finalize."""
        frame = obj.to_frame() if hasattr(obj, "to_frame") else obj
        self._frames[name] = frame
        return self

    # -- record ------------------------------------------------------------- #
    def citations(self) -> dict:
        from .references import cite as _cite, has
        return {k: _cite(k) for k in self.cite if has(k)}

    @property
    def run_dir(self) -> Path:
        if self._dir is not None:
            return self._dir
        ts = re.sub(r"[:.]", "-", self.created)
        return Path.cwd() / "runs" / f"{_slug(self.name)}-{ts}"

    def to_dict(self) -> dict:
        return {"name": self.name, "seed": self.seed, "params": self.params,
                "cite": self.cite, "created": self.created, "entry_script": self.entry_script,
                "cwd": self.cwd, "env": self.env,
                "git": {k: v for k, v in self.git.items() if k != "diff"}}

    def save(self, path) -> None:
        """Write just the JSON record to ``path`` (config + env + git, no patch/results)."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def finalize(self, dir=None) -> Path:
        """Write the full per-run folder; returns its path."""
        d = Path(dir) if dir is not None else self.run_dir
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "config.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        with open(d / "env.json", "w") as f:
            json.dump(self.env, f, indent=2, default=str)
        if self.results:
            with open(d / "results.json", "w") as f:
                json.dump(self.results, f, indent=2, default=str)
        cites = self.citations()
        if cites:
            (d / "citations.bib").write_text("\n\n".join(cites.values()) + "\n")
        save_patch(d / "code.patch", self.git)  # writes only if the tree was dirty
        for fname, frame in self._frames.items():
            try:
                frame.to_csv(d / f"{_slug(fname)}.csv", index=False)
            except Exception:
                (d / f"{_slug(fname)}.json").write_text(json.dumps(frame, default=str))
        return d

    @classmethod
    def load(cls, path) -> "Experiment":
        """Load from a config JSON file or a run-folder path."""
        p = Path(path)
        if p.is_dir():
            p = p / "config.json"
        with open(p) as f:
            d = json.load(f)
        exp = cls(d["name"], seed=d.get("seed", 0), cite=d.get("cite"), **d.get("params", {}))
        exp.created = d.get("created", exp.created)
        exp.env, exp.git = d.get("env", {}), d.get("git", {})
        exp.entry_script, exp.cwd = d.get("entry_script"), d.get("cwd", exp.cwd)
        return exp

    def snippet(self) -> str:
        """A copy-pastable Python reproduction header."""
        g = self.git or {}
        sha = g.get("sha", "n/a")
        dirty = " (DIRTY working tree — see git.diff in the saved record!)" if g.get("dirty") else ""
        lines = [
            f"# reproduction of experiment {self.name!r} (created {self.created})",
            f"# code: git {sha}{dirty}",
            f"# env: {self.env}",
            "import grouprec as gr",
            f"gr.set_seed({self.seed})",
        ]
        for key, val in self.params.items():
            lines.append(f"{key} = {val!r}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Experiment({self.name!r}, seed={self.seed}, params={self.params})"


__all__ = ["Experiment", "set_seed", "environment", "git_info", "save_patch"]
