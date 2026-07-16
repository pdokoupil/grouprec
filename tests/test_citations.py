"""Citation coverage + auto-collection.

The coverage tests are a **contract for contributors**: adding an aggregator, deep
model, or dataset without a citation in ``grouprec/references.py`` fails the build.
"""

from __future__ import annotations

import pytest

import grouprec as gr
from grouprec.aggregators import available as agg_available, get as agg_get
from grouprec.references import _DATASET_KEYS, citation_keys_for, has


# --------------------------------------------------------------------------- #
# coverage (enforced)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", agg_available())
def test_every_aggregator_has_citation(name):
    keys = citation_keys_for(agg_get(name))
    assert keys, f"aggregator {name!r} has no citation — add it to references.py (_AGG_KEYS + _REFS)"


@pytest.mark.parametrize("name", gr.datasets.available())
def test_every_registered_dataset_has_citation(name):
    key = _DATASET_KEYS.get(name.lower())
    assert key and has(key), f"dataset {name!r} has no citation — add it to references.py"


def test_every_deep_model_has_citation():
    pytest.importorskip("torch")
    from grouprec import models
    from grouprec.models.base import GroupNNModel

    # Derived from the package, not hardcoded: a hardcoded list silently exempts any
    # model added after it was written, which is exactly how HHGR shipped uncited.
    names = [n for n in models.__all__
             if isinstance(getattr(models, n), type)
             and getattr(getattr(models, n), "paradigm", None) == "profile"
             and getattr(models, n) is not GroupNNModel]
    assert len(names) >= 7, f"expected the full model zoo, found {names}"
    for cls_name in names:
        cls = getattr(models, cls_name)
        assert has(cls_name) or getattr(cls, "cite_key", None), \
            f"deep model {cls_name!r} has no citation — add it to references.py (_REFS)"


# --------------------------------------------------------------------------- #
# auto-collection
# --------------------------------------------------------------------------- #
def test_collect_citations_from_recommender_and_dataset():
    from grouprec import GroupRecommender, collect_citations, Dataset
    import pandas as pd
    rec = GroupRecommender(gr.backends.EASE(), gr.aggregators.get("GFAR"))
    d = Dataset(pd.DataFrame({"user": [1], "item": [2]}), name="ml-1m")
    cites = collect_citations(rec, d)
    assert {"GFAR", "EASE", "movielens"} <= set(cites)        # base + aggregator + dataset


def test_adapter_cites_the_framework():
    rec = gr.backends.implicit_als(factors=4)
    assert "implicit" in citation_keys_for(rec)               # transitive algo not resolved


def test_experiment_cite_accepts_objects_and_keys():
    from grouprec import GroupRecommender
    rec = GroupRecommender(gr.backends.EASE(), gr.aggregators.get("AVG"))
    exp = gr.Experiment("x", cite=[rec, "rift"])
    assert {"EASE", "social_choice", "rift"} <= set(exp.cite)
    assert "@" in exp.citations()["EASE"]
