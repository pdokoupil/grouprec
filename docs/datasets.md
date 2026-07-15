# Datasets & licensing

`gr.datasets.load("ml-1m")` fetches + caches + parses. Every entry records a license,
citation, and **download policy**:

| Policy | Datasets | Behavior |
|---|---|---|
| `auto` | MovieLens 100K/1M/25M/32M/latest/latest-small | auto-fetched for your own use (GroupLens permits research download) |
| `auto_nc` | KGRec, Last.fm (Taste Profile) | non-commercial — needs `load(..., accept_license=True)` |
| `manual` | CAMRa2011, Mafengwo, Weeplaces, Yelp, Douban | prints where to download + where to drop files |

**Licensing:** the library is MIT (that's the *code*); datasets keep their own licenses.
These never conflict because we **never redistribute the data** — `auto` fetches from the
canonical host onto your machine; anything non-redistributable is `manual`. Note that
*fetching for your own use* is not *redistribution*: the **MovieLens** terms even differ by
release — the older **100K/1M** READMEs forbid redistribution without permission, while
**25M/32M/latest** permit it under the same terms (the registry records the right one per
dataset). `ml-latest` and `ml-latest-small` are rolling releases, so we pin each one's exact
snapshot by sha256 for reproducibility.

!!! warning "`ml-latest` / `ml-latest-small` are development datasets"
    GroupLens states both are *development* datasets that "may change over time and [are] not
    an appropriate dataset for shared research results". Use **`ml-25m` / `ml-32m`** for
    anything you report; the registry records this in each spec's `notes`.

## Group-structured datasets

```python
gd = gr.datasets.load_consrec("path/to/CAMRa2011")   # AGREE/ConsRec format
# -> GroupBenchmarkData(dataset, groups, group_interactions, test_instances)
```

Use `gd.test_instances` with `evaluate_sampled` / `benchmark(level="sampled")`.

## Bring your own

```python
gr.datasets.from_path("ratings.csv", user_col="u", item_col="i", rating_col="r")
gr.datasets.from_huggingface("some/repo", user_col="userId", item_col="movieId")
```

## Preprocessing

```python
data = gr.datasets.k_core(data, k=5)
data = gr.datasets.filter_min_interactions(data, min_per_user=5, min_per_item=5)
data = gr.datasets.binarize(data, threshold=4.0)
```
