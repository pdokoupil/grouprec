# Leaderboard

## Persistent store

```python
store = gr.LeaderboardStore("results/leaderboard.csv")
store.add(result, tag="v0.1")          # appends; accumulates across runs
store.best("ndcg", k=10, protocol="coupled")
```

## Hosting on GitHub Pages

GitHub Pages serves **static files only** — it cannot run the Streamlit app (that needs
a Python server). Two options:

1. **Static HTML (Pages-native):** generate a self-contained sortable page and commit it
   to `docs/` or a `gh-pages` branch:

    ```python
    from grouprec.bench.leaderboard import render_html
    render_html(store, "docs/leaderboard.html", metric="ndcg", k=10, protocol="coupled")
    ```

    Regenerate automatically with a GitHub Action on each push (see
    `.github/workflows/docs.yml`).

2. **Streamlit Community Cloud** (free) for the interactive browser
   (`examples/leaderboard_app.py`, `pip install grouprec[demo]`) — point it at your repo;
   it runs the Python server for you. Not GitHub Pages.

The interactive app's headline interaction — toggling **coupled ↔ decoupled** and
watching the aggregator ranking flip — is the rift made tangible for a live demo.
