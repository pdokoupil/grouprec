# Reproducibility

## Seeding

```python
import grouprec as gr
gr.set_seed(42)        # seeds python, numpy, and torch (if installed)
```

Inside an `Experiment` block you don't need this call — entering the block seeds from
`exp.seed`, so a run under `with` is seeded whether or not you remember to.

## The Experiment record

`gr.Experiment` captures everything needed to reconstruct a run and, as a **context
manager**, writes a self-contained per-run folder on exit:

```python
with gr.Experiment("camra-bridge", seed=42) as exp:
    res = gr.benchmark(...)              # leaderboard + citations recorded for you
    exp.log(note="divergent groups")     # anything else: scalar results / notes
# -> runs/camra-bridge-<timestamp>/
print(exp.snippet())                     # copy-pastable reproduction header
```

A `benchmark` call inside the block attaches its results as `benchmark.csv` and
collects the citations for the recommenders and datasets it ran, so the common case
needs no bookkeeping. For results computed some other way, `exp.attach("name", obj)`
takes a DataFrame / `BenchmarkResult` / `Report` and writes it as `<name>.csv`;
pass `cite=[...]` to add references by hand.

Manual use still works (`exp = gr.Experiment(...)`, then `exp.finalize()` or
`exp.save("exp.json")`), but only the `with` form seeds and records automatically.
Override the folder with `dir=...`.

### The run folder

`config.json` · `env.json` · `citations.bib` · `results.json` · attached `*.csv` ·
`code.patch` (only when in a git repo with uncommitted changes).

### What it stores

- **seed** and your **params** (anything you pass as kwargs).
- **environment** (`gr.environment()`): python version, platform, machine, hostname,
  CPU count, and versions of grouprec/numpy/scipy/pandas/torch/implicit/lenskit.
- **code state**: in a clone of this repo, the commit **SHA**, branch, dirty flag, and
  the **full** working-tree diff written as `code.patch` (so
  `git checkout <sha> && git apply code.patch` reproduces exactly). When installed from
  PyPI (no git), it records the `grouprec` version, the **entry script** (`sys.argv[0]`),
  and the working directory instead.
- **citations** → `citations.bib`. `cite=[...]` accepts citation **keys** *or* the
  objects used in the run (aggregator / recommender / model / dataset), auto-resolved
  to keys; `exp.add_citations(rec, dataset)` and `gr.collect_citations(...)` do the same.
  Wrapped LensKit/RecBole/implicit backends cite the *framework* (the transitive
  algorithm isn't resolvable). A test enforces that every registered algorithm/dataset
  has a citation.

`Experiment.load(path)` accepts a `config.json` file **or** a run-folder path.

## Carbon / energy

```python
with gr.track_emissions() as em:
    rec.fit(data)
print(em.kg_co2e, em.seconds, em.backend)   # codecarbon if installed, else estimate
```

Or per-recommender in a benchmark: `gr.benchmark(..., track_carbon=True)` adds a
`metric="carbon_kg", aggregation="run"` row. Note carbon is **run-level** (training +
eval cost), not a per-recommendation metric, and is sensitive to the base RS and eval
protocol — not just whether a model is "deep".
