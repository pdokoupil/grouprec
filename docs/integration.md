# Integration with other recommender frameworks

`grouprec` separates *who recommends* (a single-user base recommender) from *how the
group is served* (an aggregator). Anything that implements the tiny
`BaseRecommender` protocol — `fit(dataset)` and `score(users, items) -> (n_users, n_items)`
— can be the base. We ship thin adapters so you can drop in models from **implicit**,
**LensKit**, and **RecBole** without writing glue, then combine them with any aggregator.

```python
from grouprec import GroupRecommender
from grouprec.aggregators import WeightedAverageAggregator
# pick ANY base below, then:
rec = GroupRecommender(base, WeightedAverageAggregator(member_weights=w)).fit(data)
rec.recommend(members, k=5, candidates=cands)
```

The adapters normalize three things that differ across libraries: the **training call**
(each library trains differently), the **scoring call/shape** (we always return a dense
`(n_users, n_items)` block aligned to the `Dataset` vocabulary), and **id mapping**
(library-internal ids ↔ your dataset ids).

## EASE / ItemKNN (built in, no extra)

```python
from grouprec.backends import EASE, ItemKNN
base = EASE(reg=200.0)          # or ItemKNN(k=200)
```

## implicit  (`pip install grouprec[implicit]`)

ALS / BPR matrix factorization over a confidence-weighted sparse matrix.

```python
from grouprec import backends as B
base = B.implicit_als(factors=64, iterations=20)     # or B.implicit_bpr(...)
```
*API the adapter hides:* implicit consumes a `scipy` CSR user–item matrix and exposes
`recommend`/`model.user_factors @ item_factors`; the adapter builds the CSR from the
`Dataset`, fits, and turns factor dot-products into the dense score block. `use_ratings=True`
weights confidence by rating; otherwise interactions are binarized.

## LensKit  (`pip install grouprec[lenskit]`)

LensKit 2025 uses a *scorer component* + a pipeline; you pass the scorer.

```python
from grouprec import backends as B
from lenskit.als import ImplicitMFScorer        # or BiasedMFScorer, ItemKNNScorer, …
base = B.lenskit(ImplicitMFScorer(features=64))
```
*API the adapter hides:* LensKit trains via `Component.train(...)` and scores through a
pipeline keyed by a `query` with `history_items`; the adapter trains the scorer on the
`Dataset` and issues per-user queries, assembling the dense block. Any LensKit scorer
(ImplicitMF, BiasedMF, ItemKNN, UserKNN, EASE, …) works the same way.

## RecBole  (`pip install grouprec[recbole]`, experimental)

The adapter wraps an **already-trained** RecBole model together with the RecBole `Dataset`
it was trained on — it does not train anything itself. Train the model with RecBole, then
pass the model object and its dataset:

```python
from grouprec import backends as B
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import BPR
from recbole.trainer import Trainer

cfg = Config(model="BPR", dataset="ml-100k", config_dict={"epochs": 20, "device": "cpu"})
ds = create_dataset(cfg)
train, valid, _ = data_preparation(cfg, ds)
model = BPR(cfg, train.dataset)
Trainer(cfg, model).fit(train, valid)

base = B.recbole(model, ds)            # wrap the *trained model* + its RecBole dataset
```

(If you saved a checkpoint, `recbole.quick_start.load_data_and_model` returns the same
`(model, dataset)` pair.)

*API the adapter hides:* `fit(your_dataset)` only records the target item ordering; scoring
calls the model's `full_sort_predict` and maps RecBole's internal item ids back to your
`Dataset`'s vocabulary. **Caveats:** experimental; some models expect a GPU; RecBole's
preprocessing (its own filtering/splitting) can differ from your `Dataset`, so treat scores
as approximate unless you align the preprocessing. RecBole 1.2.x also requires **NumPy < 2**
(it references the removed `np.float_`), so it may need a separate environment.

## Writing your own adapter (30 seconds)

If your model isn't covered, the protocol is two methods:

```python
import numpy as np
from grouprec.data import Dataset

class MyRecommender:
    def fit(self, dataset: Dataset) -> "MyRecommender":
        self.dataset_ = dataset
        # ... train on dataset.interactions (columns: user, item, rating, …) ...
        return self
    def score(self, users, items=None) -> np.ndarray:
        items = list(items) if items is not None else list(self.dataset_.items)
        # return shape (len(users), len(items)); rows aligned to `users`, cols to `items`
        ...
```

Then `GroupRecommender(MyRecommender(), aggregator)` works, and it slots into
`gr.benchmark(...)` and the leaderboard alongside everything else.

## Deep group models are different

The adapters above are for **single-user base recommenders** used by results-aggregation.
The **profile-aggregation** deep models (NCFGroup, AGREE, GroupIM, ConsRec, HyperGroup,
HHGR, AlignGroup)
learn directly from group interactions and expose their own
`recommend(members, k, candidates=…)` — they are not wrapped in `GroupRecommender`. Both
families are evaluated by the same protocols and appear in one leaderboard; see
[Evaluation](evaluation.md) and [Design](design.md).
