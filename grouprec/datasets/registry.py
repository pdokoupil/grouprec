"""License-aware dataset registry.

Each entry records the canonical URL(s), license, citation, and a **download
policy** -- because dataset licenses differ wildly and most cannot be redistributed:

* ``auto``    -- permissively licensed; fetched automatically on first use
  (e.g. MovieLens, whose GroupLens license permits redistribution-with-conditions).
* ``auto_nc`` -- non-commercial / derivative-restricted; auto-fetchable from the
  canonical host but only after the caller acknowledges the license
  (``load(..., accept_license=True)``).
* ``manual``  -- redistribution rights unclear or forbidden (Yelp ToS, crawled
  datasets); we ship a loader + instructions, never the bytes. ``load`` tells you
  exactly where to download and where to drop the file.

We **never bundle datasets in the wheel**. The library is MIT; each dataset keeps
its own license (see the note in the docs / README): MIT covers our *code*, not the
*data*, and because we don't redistribute the bytes the two never conflict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..data import Dataset
from . import loaders
from .cache import dataset_dir, download, extract


@dataclass
class DatasetSpec:
    name: str
    scenario: int
    policy: str                      # auto | auto_nc | manual
    license: str
    citation: str
    homepage: str
    loader: Callable[[Path], Dataset]
    urls: list[str] = field(default_factory=list)
    archive: str | None = None       # filename to save the (first) url as
    checksum: str | None = None
    manual_instructions: str | None = None
    notes: str = ""


_REGISTRY: dict[str, DatasetSpec] = {}


def register(spec: DatasetSpec) -> None:
    _REGISTRY[spec.name] = spec


def available() -> list[str]:
    return sorted(_REGISTRY)


def info(name: str) -> DatasetSpec:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown dataset {name!r}; available: {available()}") from None


def load(name: str, *, accept_license: bool = False, **loader_kwargs) -> Dataset:
    """Load a registered dataset, fetching/caching as the policy allows."""
    spec = info(name)
    ddir = dataset_dir(name)

    if spec.policy == "manual":
        # the loader reads from files the user placed in the cache dir
        try:
            return spec.loader(ddir, **loader_kwargs)
        except FileNotFoundError as exc:
            raise RuntimeError(_manual_message(spec, ddir, str(exc))) from None

    if spec.policy == "auto_nc" and not accept_license:
        raise RuntimeError(
            f"'{name}' is licensed as: {spec.license}\n"
            f"It is non-commercial / derivative-restricted. Re-run with "
            f"accept_license=True to download from {spec.homepage} and accept these terms.\n"
            f"Cite: {spec.citation}"
        )

    # auto / accepted auto_nc -> ensure download + extract, then load
    archive_name = spec.archive or (spec.urls[0].rsplit("/", 1)[-1] if spec.urls else None)
    if archive_name:
        archive_path = ddir / archive_name
        download(spec.urls[0], archive_path, checksum=spec.checksum)
        extract(archive_path, ddir)
    if spec.policy == "auto_nc":
        print(f"[grouprec] '{name}' license: {spec.license}. Cite: {spec.citation}")
    return spec.loader(ddir, **loader_kwargs)


def _manual_message(spec: DatasetSpec, ddir: Path, err: str) -> str:
    lines = [
        f"'{spec.name}' must be downloaded manually ({spec.license}).",
        f"  Homepage: {spec.homepage}",
    ]
    if spec.urls:
        lines.append(f"  Download:  {spec.urls[0]}")
    lines.append(f"  Then place the files under: {ddir}")
    if spec.manual_instructions:
        lines.append(spec.manual_instructions)
    lines.append(f"  Cite: {spec.citation}")
    lines.append(f"  (loader error: {err})")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# registry entries
# --------------------------------------------------------------------------- #
_ML = "https://files.grouplens.org/datasets/movielens"
_GROUPLENS_LICENSE = ("GroupLens / MovieLens usage license "
                      "(redistribution permitted with conditions; cite, do not re-host)")

register(DatasetSpec(
    name="ml-100k", scenario=1, policy="auto", license=_GROUPLENS_LICENSE,
    citation="Harper & Konstan 2015, The MovieLens Datasets, ACM TiiS",
    homepage="https://grouplens.org/datasets/movielens/100k/",
    urls=[f"{_ML}/ml-100k.zip"], loader=loaders.movielens_100k,
))
register(DatasetSpec(
    name="ml-1m", scenario=1, policy="auto", license=_GROUPLENS_LICENSE,
    citation="Harper & Konstan 2015, The MovieLens Datasets, ACM TiiS",
    homepage="https://grouplens.org/datasets/movielens/1m/",
    urls=[f"{_ML}/ml-1m.zip"], loader=loaders.movielens_1m,
))
register(DatasetSpec(
    name="ml-25m", scenario=1, policy="auto", license=_GROUPLENS_LICENSE,
    citation="Harper & Konstan 2015, The MovieLens Datasets, ACM TiiS",
    homepage="https://grouplens.org/datasets/movielens/25m/",
    urls=[f"{_ML}/ml-25m.zip"], loader=loaders.movielens_25m,
))
register(DatasetSpec(
    name="ml-32m", scenario=1, policy="auto", license=_GROUPLENS_LICENSE,
    citation="Harper & Konstan 2015, The MovieLens Datasets, ACM TiiS",
    homepage="https://grouplens.org/datasets/movielens/32m/",
    urls=[f"{_ML}/ml-32m.zip"], loader=loaders.movielens_32m,
))

register(DatasetSpec(
    name="kgrec", scenario=1, policy="auto_nc",
    license="CC BY-NC 3.0 (except 3rd-party; Last.fm interactions/tags, Songfacts text). "
            "Non-commercial; derivative work under Last.fm API ToS para 4.1.",
    citation="Oramas, Ostuni, Vigliensoni et al., KGRec dataset (MTG, UPF)",
    homepage="https://www.upf.edu/web/mtg/kgrec",
    urls=["http://mtg.upf.edu/system/files/projectsweb/KGRec-dataset.zip"],
    loader=loaders.kgrec_music,
))

register(DatasetSpec(
    name="lastfm", scenario=1, policy="auto_nc",
    license="Echo Nest Taste Profile / Million Song Dataset terms (research use; "
            "see MSD site). Treat as non-commercial.",
    citation="Bertin-Mahieux et al. 2011, The Million Song Dataset (Taste Profile subset)",
    homepage="http://millionsongdataset.com/tasteprofile/",
    urls=["http://labrosa.ee.columbia.edu/~dpwe/tmp/train_triplets.txt.zip"],
    loader=loaders.taste_profile,
))

# -- manual (redistribution unclear / forbidden) ---------------------------- #
def _manual_loader(name, default_file):
    def _load(ddir: Path, **kw):
        from .cache import find_file
        f = find_file(ddir, default_file, "*.csv", "*.txt")
        if f is None:
            raise FileNotFoundError(f"no interactions file for {name} in {ddir}")
        return loaders.generic_interactions(f, name=name, **kw)
    return _load


register(DatasetSpec(
    name="camra2011", scenario=3, policy="manual",
    license="research dataset; redistribution terms unclear",
    citation="CAMRa 2011 Challenge",
    homepage="https://github.com/FDUDSDE/WWW2023ConsRec",
    urls=["https://github.com/FDUDSDE/WWW2023ConsRec"],
    loader=_manual_loader("camra2011", "*camra*"),
    manual_instructions="  Clone the ConsRec repo and copy its CAMRa2011 data folder here.",
))
register(DatasetSpec(
    name="mafengwo", scenario=3, policy="manual",
    license="crawled; redistribution rights unclear",
    citation="Mafengwo (used by ConsRec, WWW'23)",
    homepage="https://github.com/FDUDSDE/WWW2023ConsRec",
    urls=["https://github.com/FDUDSDE/WWW2023ConsRec"],
    loader=_manual_loader("mafengwo", "*mafengwo*"),
    manual_instructions="  Copy the Mafengwo data folder from the ConsRec repo here.",
))
register(DatasetSpec(
    name="weeplaces", scenario=2, policy="manual",
    license="provenance messy; verify mirror license",
    citation="Weeplaces (used by GroupIM, SIGIR'20)",
    homepage="https://github.com/CrowdDynamicsLab/GroupIM",
    urls=["https://github.com/CrowdDynamicsLab/GroupIM"],
    loader=_manual_loader("weeplaces", "*weeplace*"),
    manual_instructions="  Copy the Weeplaces data from the GroupIM repo here.",
))
register(DatasetSpec(
    name="yelp", scenario=2, policy="auto_nc",
    license="Yin et al. group benchmark (Yelp-LA): no explicit license; non-commercial "
            "research use. Cite ICDE'19 (10.1109/ICDE.2019.00057) + ICDE'18 (10.1109/ICDE.2018.00088).",
    citation="Yin et al., Social Influence-Based Group Representation Learning, ICDE 2019",
    homepage="https://sites.google.com/view/hongzhi-yin/datasets",
    urls=["https://sites.google.com/view/hongzhi-yin/datasets"],
    loader=_manual_loader("yelp", "*yelp*"),
    manual_instructions="  Use gr.datasets.fetch_yin(accept_license=True) then "
                        "gr.datasets.load_yin(path, 'yelp') for the group benchmark.",
))
register(DatasetSpec(
    name="douban", scenario=2, policy="auto_nc",
    license="Yin et al. group benchmark (Douban-SH): no explicit license; non-commercial "
            "research use. Cite ICDE'18 (10.1109/ICDE.2018.00088) + ICDE'19 (10.1109/ICDE.2019.00057).",
    citation="Yin et al., Joint Event-Partner Recommendation, ICDE 2018",
    homepage="https://sites.google.com/view/hongzhi-yin/datasets",
    urls=["https://sites.google.com/view/hongzhi-yin/datasets"],
    loader=_manual_loader("douban", "*douban*"),
    manual_instructions="  Use gr.datasets.fetch_yin(accept_license=True) then "
                        "gr.datasets.load_yin(path, 'douban') for the group benchmark.",
))
