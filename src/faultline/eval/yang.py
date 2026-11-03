"""Public labeled leakage benchmark: Yang et al., ASE'22, "Data Leakage in Notebooks".

99 GitHub notebooks hand-labeled per notebook for three leakage kinds:
  pre     preprocessing fit on data before the split -> our preprocessing_before_split, feature_selection_full
  overlap test rows used in training                  -> our test_in_training
  multi   repeated evaluation / tuning on the test set -> our tuning_on_test
Their tool's own results on the same notebooks are the baseline. Notebooks are split 50/50 into
dev and test by a fixed hash; test runs only with final=True. The materials have no license
file, so they are downloaded for local evaluation and never committed.
"""

from __future__ import annotations

import csv
import hashlib
import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from faultline.detectors.leakage import LeakageDetector
from faultline.discovery.sources import SourceFile, SourceKind, notebook_to_code

BASE_URL = "https://github.com/malusamayo/GitHubAPI-Crawler/raw/master/evaluation_materials/"
FILES = ("sample-notebooks.zip", "ground-truth.csv", "analysis-result.csv", "README.md")
DEFAULT_DIR = Path(__file__).resolve().parents[3] / "corpus" / "benchmarks" / "yang2022"
KINDS = ("pre", "overlap", "multi")
OUR_TYPES = {
    "pre": {"data.leakage.preprocessing_before_split", "data.leakage.feature_selection_full"},
    "overlap": {"data.leakage.test_in_training", "data.leakage.resampling_before_split"},
    "multi": {"data.leakage.tuning_on_test"},
}
BenchSplit = Literal["dev", "test"]


@dataclass(frozen=True)
class LabeledNotebook:
    id: str  # "2021-09-05/nb_1244"
    split: BenchSplit
    source: SourceFile
    labels: dict[str, bool | None]  # None = not labeled (notebook trains no model)
    baseline: dict[str, bool]  # Yang et al. tool output (empty if it failed)


class Counts(BaseModel):
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def precision(self) -> float | None:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else None

    @property
    def recall(self) -> float | None:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)

    def add(self, predicted: bool, actual: bool) -> None:
        if predicted and actual:
            self.tp += 1
        elif predicted:
            self.fp += 1
        elif actual:
            self.fn += 1
        else:
            self.tn += 1


class YangReport(BaseModel):
    split: BenchSplit
    labeled: int
    ours: dict[str, Counts]
    baseline: dict[str, Counts]
    misses: dict[str, list[str]]  # kind -> notebook ids we got wrong (fn and fp), for analysis on dev


def _id_from_gt(path: str) -> str:
    # GitHub-data/notebooks/2021-09-05/nb_1244.py -> 2021-09-05/nb_1244
    parts = path.strip().split("/")
    return f"{parts[-2]}/{parts[-1].removesuffix('.py')}"


def _id_from_zip(name: str) -> str:
    # sample_notebooks/2021-09-05-nb_1244.ipynb -> 2021-09-05/nb_1244
    stem = name.rsplit("/", 1)[-1].removesuffix(".ipynb")
    return f"{stem[:10]}/{stem[11:]}"


def _split(nb_id: str) -> BenchSplit:
    return "dev" if int(hashlib.sha256(f"yang2022:{nb_id}".encode()).hexdigest(), 16) % 2 == 0 else "test"


def _label(cell: str) -> bool | None:
    cell = cell.strip()
    return None if not cell else cell.upper().startswith("Y")


def fetch(dest: Path = DEFAULT_DIR) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        if not (dest / name).exists():
            with urllib.request.urlopen(BASE_URL + name, timeout=120) as resp:
                (dest / name).write_bytes(resp.read())
    return dest


def load_benchmark(root: Path = DEFAULT_DIR) -> list[LabeledNotebook]:
    labels: dict[str, dict[str, bool | None]] = {}
    with (root / "ground-truth.csv").open(encoding="utf-8") as f:
        for row in list(csv.reader(f))[1:]:
            if row and row[0].strip():
                has_model = _label(row[1]) is True
                labels[_id_from_gt(row[0])] = {
                    k: (_label(row[2 + i]) if has_model else None) for i, k in enumerate(KINDS)
                }
    baseline: dict[str, dict[str, bool]] = {}
    with (root / "analysis-result.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("msg", "").startswith("Success"):
                baseline[_id_from_gt(r["nb"])] = {k: r.get(k) == "True" for k in KINDS}
    items = []
    with zipfile.ZipFile(root / "sample-notebooks.zip") as z:
        for name in sorted(n for n in z.namelist() if n.endswith(".ipynb")):
            nb_id = _id_from_zip(name)
            if nb_id not in labels:
                continue
            raw = io.TextIOWrapper(z.open(name), encoding="utf-8", errors="replace").read()
            try:
                code, starts = notebook_to_code(raw)
            except ValueError:
                code, starts = "", ()
            src = SourceFile(path=name, kind=SourceKind.notebook, code=code, cell_starts=starts)
            items.append(LabeledNotebook(nb_id, _split(nb_id), src, labels[nb_id], baseline.get(nb_id, {})))
    return items


def evaluate(items: list[LabeledNotebook], split: BenchSplit, *, final: bool = False) -> YangReport:
    if split == "test" and not final:
        raise PermissionError("the test half of the Yang et al. benchmark is held out; use final=True once")
    det = LeakageDetector()
    ours = {k: Counts() for k in KINDS}
    base = {k: Counts() for k in KINDS}
    misses: dict[str, list[str]] = {k: [] for k in KINDS}
    labeled = 0
    for item in (i for i in items if i.split == split):
        if all(v is None for v in item.labels.values()):
            continue
        labeled += 1
        found = {f.type for f in det.detect(item.source)}
        for k in KINDS:
            actual = item.labels[k]
            if actual is None:
                continue
            predicted = bool(found & OUR_TYPES[k])
            ours[k].add(predicted, actual)
            base[k].add(item.baseline.get(k, False), actual)
            if predicted != actual:
                misses[k].append(f"{item.id} ({'fp' if predicted else 'fn'})")
    return YangReport(split=split, labeled=labeled, ours=ours, baseline=base, misses=misses)


def summarize(report: YangReport) -> str:
    def fmt(v: float | None) -> str:
        return "  -  " if v is None else f"{v:5.1%}"

    lines = [
        f"Yang et al. ASE'22 leakage benchmark, {report.split} half: {report.labeled} labeled notebooks",
        "",
    ]
    lines.append(f"  {'kind':8s} {'':10s} {'P':>6s} {'R':>6s} {'F1':>6s}   tp fp fn")
    for k in KINDS:
        for who, c in (("faultline", report.ours[k]), ("yang-tool", report.baseline[k])):
            lines.append(
                f"  {k:8s} {who:10s} {fmt(c.precision)} {fmt(c.recall)} {fmt(c.f1)}   {c.tp:2d} {c.fp:2d} {c.fn:2d}"
            )
    return "\n".join(lines)
