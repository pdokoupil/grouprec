"""Dynamic, persistent leaderboard store.

Append benchmark results (with metadata) to a CSV that grows across runs, then query
the current best per dataset/metric -- a living leaderboard for the project/community.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


class LeaderboardStore:
    """A CSV-backed accumulating leaderboard."""

    COLUMNS = ["timestamp", "dataset", "recommender", "paradigm", "protocol",
               "metric", "k", "aggregation", "value", "tag"]

    def __init__(self, path) -> None:
        self.path = Path(path)

    def add(self, result, *, tag: str = "") -> "LeaderboardStore":
        """Append a :class:`~grouprec.bench.BenchmarkResult` (or its DataFrame)."""
        df = result.to_frame() if hasattr(result, "to_frame") else result.copy()
        df = df.copy()
        df["timestamp"] = datetime.now(timezone.utc).isoformat()
        df["tag"] = tag
        df = df[self.COLUMNS]
        header = not self.path.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.path, mode="a", header=header, index=False)
        return self

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.path) if self.path.exists() else pd.DataFrame(columns=self.COLUMNS)

    def best(self, metric: str, *, dataset: str | None = None, k: int | None = None,
             aggregation: str = "mean", protocol: str = "coupled",
             ascending: bool = False) -> pd.DataFrame:
        """Best recommender per dataset for one (metric, k, aggregation, protocol)."""
        df = self.load()
        sel = (df["metric"] == metric) & (df["aggregation"] == aggregation) & (df["protocol"] == protocol)
        if dataset is not None:
            sel &= df["dataset"] == dataset
        if k is not None:
            sel &= df["k"] == k
        df = df[sel].sort_values("value", ascending=ascending)
        return df.groupby("dataset", as_index=False).first()


_HTML_TMPL = """<!doctype html><html><head><meta charset="utf-8">
<title>grouprec leaderboard</title>
<style>
body{{font-family:system-ui,Arial,sans-serif;margin:2rem;color:#222}}
h1{{font-size:1.4rem}} .meta{{color:#666;font-size:.85rem}}
table{{border-collapse:collapse;margin-top:1rem;font-size:.9rem}}
th,td{{border:1px solid #ddd;padding:.35rem .6rem;text-align:right}}
th{{background:#f6f6f6;cursor:pointer}} td:first-child,th:first-child{{text-align:left}}
tr:nth-child(even){{background:#fafafa}}
</style></head><body>
<h1>grouprec leaderboard</h1>
<p class="meta">{meta}</p>
{table}
<script>
document.querySelectorAll('th').forEach((th,i)=>th.onclick=()=>{{
 const tb=th.closest('table').tBodies[0];
 const rows=[...tb.rows];const asc=th.dataset.asc=th.dataset.asc==='1'?'0':'1';
 rows.sort((a,b)=>{{const x=a.cells[i].innerText,y=b.cells[i].innerText;
  const nx=parseFloat(x),ny=parseFloat(y);
  const c=(!isNaN(nx)&&!isNaN(ny))?nx-ny:x.localeCompare(y);return asc==='1'?c:-c;}});
 rows.forEach(r=>tb.appendChild(r));}});
</script></body></html>"""


def render_html(result, path, *, metric: str = "ndcg", k: int | None = None,
                aggregation: str = "mean", protocol: str = "coupled",
                meta: str = "") -> str:
    """Write a **static, sortable** HTML leaderboard page (host it on GitHub Pages).

    GitHub Pages serves static files only, so it cannot run the Streamlit app (that
    needs a Python server — use Streamlit Community Cloud for that). This page is
    plain HTML/JS and commits straight to a ``gh-pages`` branch / ``docs/`` folder.
    """
    import pandas as pd
    df = result.load() if hasattr(result, "load") else (
        result.to_frame() if hasattr(result, "to_frame") else result)
    sel = (df["metric"] == metric) & (df["aggregation"] == aggregation) & (df["protocol"] == protocol)
    if k is not None:
        sel &= df["k"] == k
    table = df[sel].pivot_table(index="recommender", columns="dataset", values="value").round(4)
    html = _HTML_TMPL.format(
        meta=meta or f"{metric}@{k} · {aggregation} · {protocol}",
        table=table.to_html(border=0))
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(html)
    return html


__all__ = ["LeaderboardStore", "render_html"]
