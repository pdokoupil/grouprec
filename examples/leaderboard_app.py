"""Lightweight Streamlit browser for a grouprec benchmark CSV.

Not part of the library core or the conference demo by default -- a convenience for
exploring results visually. Run with the ``[demo]`` extra:

    pip install grouprec[demo]
    streamlit run examples/leaderboard_app.py -- --csv leaderboard.csv

The CSV is the output of ``BenchmarkResult.to_csv(...)`` (columns: dataset,
recommender, paradigm, protocol, metric, k, aggregation, value).
"""

from __future__ import annotations

import argparse

import pandas as pd
import streamlit as st


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="leaderboard.csv")
    args, _ = parser.parse_known_args()

    st.set_page_config(page_title="grouprec leaderboard", layout="wide")
    st.title("grouprec — leaderboard browser")

    df = pd.read_csv(args.csv)

    c1, c2, c3, c4 = st.columns(4)
    protocol = c1.selectbox("protocol", sorted(df["protocol"].unique()))
    metric = c2.selectbox("metric", sorted(df["metric"].unique()))
    ks = sorted(df["k"].unique())
    k = c3.selectbox("k", ks, index=len(ks) - 1)
    aggregation = c4.selectbox("aggregation", sorted(df["aggregation"].unique()))

    sel = df[(df.protocol == protocol) & (df.metric == metric)
             & (df.k == k) & (df.aggregation == aggregation)]
    table = sel.pivot_table(index="recommender", columns="dataset", values="value")
    st.subheader(f"{metric}@{k} · {aggregation} · {protocol}")
    st.dataframe(table.style.highlight_max(axis=0))
    st.bar_chart(table)

    st.caption("Toggle protocol coupled↔decoupled to watch the aggregator ranking flip "
               "— the rift, made tangible.")


if __name__ == "__main__":
    main()
