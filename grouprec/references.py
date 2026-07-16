"""Citations for every implemented algorithm / protocol.

``gr.cite("GFAR")`` returns a BibTeX string; ``gr.references.all()`` lists them. Keep
this in sync when adding algorithms (see CONTRIBUTING.md).
"""

from __future__ import annotations

_REFS: dict[str, str] = {
    # --- aggregators / social choice ---
    "social_choice": "@book{masthoff2015group, title={Group recommender systems}, "
                     "author={Masthoff, Judith}, year={2015}, publisher={Springer}}",
    "GFAR": "@inproceedings{kaya2020gfar, title={Ensuring Fairness in Group "
            "Recommendations by Rank-Sensitive Balancing of Relevance}, "
            "author={Kaya, Mesut and Bridge, Derek and Tintarev, Nava}, "
            "booktitle={RecSys}, year={2020}, doi={10.1145/3383313.3412232}}",
    "EPFuzzDA": "@inproceedings{ribeiro2021epfuzzda, title={Multiobjective "
                "recommendation via fuzzy D'Hondt}, booktitle={UMAP}, year={2021}, "
                "doi={10.1145/3450614.3461679}}",
    "GreedyLM": "@inproceedings{xiao2017fairness, title={Fairness-Aware Group "
                "Recommendation with Pareto-Efficiency}, author={Xiao, Lin and others}, "
                "booktitle={RecSys}, year={2017}, doi={10.1145/3109859.3109887}}",
    "PAR": "@inproceedings{xiao2017fairness, title={Fairness-Aware Group "
           "Recommendation with Pareto-Efficiency}, author={Xiao, Lin and others}, "
           "booktitle={RecSys}, year={2017}, doi={10.1145/3109859.3109887}}",
    "SPGreedy": "@inproceedings{serbos2017fairness, title={Fairness in Package-to-Group "
                "Recommendations}, author={Serbos, Dimitris and others}, booktitle={WWW}, "
                "year={2017}, doi={10.1145/3038912.3052612}}",
    # --- sequential ---
    "LTP": "@article{dokoupil2026ltp, title={Long-term fairness in sequential group "
           "recommendations}, author={Dokoupil, Patrik and Peska, Ladislav}, "
           "journal={Knowledge and Information Systems}, year={2026}, "
           "doi={10.1007/s10115-025-02642-9}}",
    "RLProp": "@article{dokoupil2026ltp, note={RLProp predecessor}, "
              "doi={10.1007/s10115-025-02642-9}}",
    "SDAA": "@article{stratigi2022sequential, title={Sequential group recommendations "
            "based on satisfaction and disagreement scores}, author={Stratigi, Maria "
            "and others}, journal={JIIS}, year={2022}, doi={10.1007/s10844-021-00652-x}}",
    "SIAA": "@article{stratigi2022sequential, doi={10.1007/s10844-021-00652-x}}",
    # --- deep models ---
    "NCFGroup": "@inproceedings{he2017ncf, title={Neural Collaborative Filtering}, "
                "author={He, Xiangnan and others}, booktitle={WWW}, year={2017}}",
    "AGREE": "@inproceedings{cao2018agree, title={Attentive Group Recommendation}, "
             "author={Cao, Da and others}, booktitle={SIGIR}, year={2018}}",
    "GroupIM": "@inproceedings{sankar2020groupim, title={GroupIM: A Mutual Information "
               "Maximization Framework for Neural Group Recommendation}, "
               "author={Sankar, Aravind and others}, booktitle={SIGIR}, year={2020}}",
    "ConsRec": "@inproceedings{wu2023consrec, title={ConsRec: Learning Consensus Behind "
               "Interactions for Group Recommendation}, author={Wu, Xixi and others}, "
               "booktitle={WWW}, year={2023}}",
    "HHGR": "@inproceedings{zhang2021hhgr, title={Double-Scale Self-Supervised "
            "Hypergraph Learning for Group Recommendation}, author={Zhang, Junwei and "
            "Gao, Chongming and Jin, Depeng and Li, Yong}, booktitle={CIKM}, "
            "year={2021}, doi={10.1145/3459637.3482426}}",
    # --- evaluation protocols ---
    "coupled_decoupled": "@inproceedings{coupled_decoupled, title={Coupled or Decoupled "
                         "Evaluation for Group Recommender Systems}, year={2024}}",
    "rift": "@inproceedings{peska2025rift, title={Bridging the Rift in group "
            "recommendation}, author={Peska, Ladislav and others}, booktitle={UMAP}, "
            "year={2025}}",
    # --- datasets requiring citation ---
    "yelp": "@inproceedings{8731382, author={Yin, Hongzhi and Wang, Qinyong and Zheng, "
            "Kai and Li, Zhixu and Yang, Jiali and Zhou, Xiaofang}, booktitle={ICDE}, "
            "title={Social Influence-Based Group Representation Learning for Group "
            "Recommendation}, year={2019}, pages={566-577}, doi={10.1109/ICDE.2019.00057}}",
    "douban": "@inproceedings{8509309, author={Yin, Hongzhi and Zou, Lei and Nguyen, Quoc "
              "Viet Hung and Huang, Zi and Zhou, Xiaofang}, booktitle={ICDE}, title={Joint "
              "Event-Partner Recommendation in Event-Based Social Networks}, year={2018}, "
              "pages={929-940}, doi={10.1109/ICDE.2018.00088}}",
    "AlignGroup": "@inproceedings{xu2024aligngroup, title={AlignGroup: Learning and Aligning "
                  "Group Consensus with Member Preferences for Group Recommendation}, "
                  "author={Xu, Jinfeng and others}, booktitle={CIKM}, year={2024}}",
    "HyperGroup": "@article{guo2021hypergroup, title={Hierarchical hyperedge embedding-based "
                  "representation learning for group recommendation}, author={Guo, Lei and "
                  "others}, journal={TOIS}, year={2021}}",
    # --- base recommenders / backends ---
    "EASE": "@inproceedings{steck2019ease, title={Embarrassingly Shallow Autoencoders for "
            "Sparse Data}, author={Steck, Harald}, booktitle={WWW}, year={2019}}",
    "ItemKNN": "@inproceedings{sarwar2001itemknn, title={Item-based collaborative filtering "
               "recommendation algorithms}, author={Sarwar, Badrul and others}, "
               "booktitle={WWW}, year={2001}}",
    "implicit": "@misc{frederickson_implicit, title={implicit: Fast Python collaborative "
                "filtering for implicit datasets}, author={Frederickson, Ben}, "
                "howpublished={\\url{https://github.com/benfred/implicit}}}",
    "lenskit": "@inproceedings{ekstrand2020lenskit, title={LensKit for Python}, "
               "author={Ekstrand, Michael D.}, booktitle={CIKM}, year={2020}}",
    "recbole": "@inproceedings{zhao2021recbole, title={RecBole: Towards a Unified, "
               "Comprehensive and Efficient Framework for Recommendation Algorithms}, "
               "author={Zhao, Wayne Xin and others}, booktitle={CIKM}, year={2021}}",
    # --- datasets ---
    "movielens": "@article{harper2015movielens, title={The MovieLens Datasets: History and "
                 "Context}, author={Harper, F. Maxwell and Konstan, Joseph A.}, "
                 "journal={ACM TiiS}, year={2015}, doi={10.1145/2827872}}",
    "kgrec": "@inproceedings{oramas2015kgrec, title={Sound and Music Recommendation with "
             "Knowledge Graphs}, author={Oramas, Sergio and others}, booktitle={ACM TIST}, "
             "year={2016}}",
    "lastfm": "@inproceedings{bertin2011msd, title={The Million Song Dataset}, "
              "author={Bertin-Mahieux, Thierry and others}, booktitle={ISMIR}, year={2011}}",
    "weeplaces": "@inproceedings{sankar2020groupim, note={Weeplaces as used by GroupIM}, "
                 "title={GroupIM}, booktitle={SIGIR}, year={2020}}",
    "mafengwo": "@inproceedings{wu2023consrec, note={Mafengwo as used by ConsRec}, "
                "title={ConsRec}, booktitle={WWW}, year={2023}}",
    "camra2011": "@inproceedings{said2011camra, title={The CAMRa 2011 Mood Aware Music "
                 "Recommendation Challenge}, author={Said, Alan and others}, year={2011}}",
}

# aggregator short-name -> citation key (many social-choice ones share one reference)
_AGG_KEYS: dict[str, str] = {
    "ADD": "social_choice", "AVG": "social_choice", "wAVG": "social_choice",
    "LMS": "social_choice",
    "MUL": "social_choice", "MPL": "social_choice", "AVGNM": "social_choice",
    "BDC": "social_choice", "FAI": "social_choice", "PeriodicFAI": "social_choice",
    "GFAR": "GFAR", "EPFuzzDA": "EPFuzzDA", "EPFuzzDAWeighted": "EPFuzzDA",
    "GreedyLM": "GreedyLM", "PAR": "PAR", "SPGreedy": "SPGreedy",
    "RLProp": "RLProp", "LTP": "LTP", "SDAA": "SDAA", "SIAA": "SIAA",
}

# dataset name (lower-cased) -> citation key
_DATASET_KEYS: dict[str, str] = {
    "ml-100k": "movielens", "ml-1m": "movielens", "ml-25m": "movielens",
    "ml-32m": "movielens", "ml-latest-small": "movielens", "ml-latest": "movielens",
    "kgrec": "kgrec", "lastfm": "lastfm",
    "lastfm-tasteprofile": "lastfm", "weeplaces": "weeplaces", "mafengwo": "mafengwo",
    "camra2011": "camra2011", "yelp": "yelp", "douban": "douban",
}


def citation_keys_for(obj) -> set:
    """Resolve the citation keys for an object actually used in a run -- an aggregator,
    a base recommender, a ``GroupRecommender`` / ``ProfileGroupRecommender``, a deep
    model, a ``Dataset`` / ``GroupBenchmarkData``, or a backend. Recurses into
    ``.base`` / ``.aggregator`` / ``.dataset``. Returns only keys with a citation."""
    keys: set = set()
    ck = getattr(obj, "cite_key", None)
    if ck:
        keys |= {ck} if isinstance(ck, str) else set(ck)
    for attr in ("base", "aggregator"):
        sub = getattr(obj, attr, None)
        if sub is not None:
            keys |= citation_keys_for(sub)
    cls = type(obj).__name__
    if cls in _REFS:
        keys.add(cls)
    name = getattr(obj, "name", None)
    if isinstance(name, str):
        if name in _AGG_KEYS:
            keys.add(_AGG_KEYS[name])
        if name.lower() in _DATASET_KEYS:
            keys.add(_DATASET_KEYS[name.lower()])
    for attr in ("dataset", "data"):  # GroupBenchmarkData.dataset / BenchmarkTask.data
        ds = getattr(obj, attr, None)
        if ds is not None and ds is not obj:
            keys |= citation_keys_for(ds)
    return {k for k in keys if k in _REFS}


def collect_citations(*objs) -> dict:
    """BibTeX for everything used across ``objs`` (recommenders, datasets, ...)."""
    keys: set = set()
    for o in objs:
        keys |= citation_keys_for(o)
    return {k: _REFS[k] for k in sorted(keys)}


def _resolve(name: str) -> str | None:
    """Map a public name to a ``_REFS`` key: an entry of its own, an aggregator short
    name (several social-choice ones share the Masthoff reference), or a dataset."""
    for table in (_REFS, _AGG_KEYS, _DATASET_KEYS):
        for key, val in table.items():
            if key.lower() == name.lower():
                return key if table is _REFS else val
    return None


def cite(name: str) -> str:
    """BibTeX for an algorithm/protocol/dataset (case-insensitive)."""
    key = _resolve(name)
    if key is None:
        raise KeyError(
            f"no citation for {name!r}; available: "
            f"{sorted(set(_REFS) | set(_AGG_KEYS) | set(_DATASET_KEYS))}"
        )
    return _REFS[key]


def has(name: str) -> bool:
    """Whether a citation exists for ``name`` (case-insensitive)."""
    return _resolve(name) is not None


def all() -> dict[str, str]:  # noqa: A003
    return dict(_REFS)


__all__ = ["cite", "has", "all", "citation_keys_for", "collect_citations"]
