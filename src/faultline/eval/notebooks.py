"""Score the rule detectors against the per-notebook hand labels (corpus/labels/notebooks/).

Each labeled Kaggle notebook is scanned like any other file. Findings are compared with the
labeled issues per finding type, at two levels: notebook (does it have this issue at all) and line
(a finding within LINE_TOLERANCE lines of a labeled issue of the same type). Notebooks labeled
`unsure` are left out.

With `context=True` the competition label (corpus/labels/competitions.yaml) is turned into a
dataset profile, so the data-fit rules know what the data requires, as `scan --profiles` would.
The test split only runs with final=True.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from faultline.contract import DatasetProfile
from faultline.detectors import DETECTORS, run_detectors, with_profiles
from faultline.detectors.base import Detector
from faultline.discovery.sources import SourceFile, load_source
from faultline.eval.corpus import Manifest, Split
from faultline.eval.labels import CompetitionLabel, NotebookLabel, NotebookLabelFile
from faultline.schema import Finding

LINE_TOLERANCE = 3
LARGE_ROWS = 100_000  # stands in for "not small" when a competition label has no row count
RARE_SHARE = 0.1  # stands in for "has a rare class" when a label says stratify


class CompetitionProfile(DatasetProfile):
    """A profile built from a competition label: it describes whatever data the notebook reads."""

    def matches(self, path: str) -> bool:
        return True


def competition_profile(name: str, label: CompetitionLabel) -> CompetitionProfile:
    time_cols = [label.time_key] if label.time_key else []
    return CompetitionProfile(
        name=name,
        rows=500 if label.small else LARGE_ROWS,  # no column list: column-level rules stay silent
        task="regression" if label.task == "regression" else "classification",
        minority_share=RARE_SHARE if label.stratify else None,
        group_keys=[label.group_key] if label.group_key else [],
        time_columns=time_cols,
        time_ordered=label.split_type == "temporal",
        metric=label.metric,
    )


@dataclass(frozen=True)
class LabeledNotebook:
    competition: str
    rel: str  # path relative to the competition's checkout, the label key
    src: SourceFile
    label: NotebookLabel
    competition_label: CompetitionLabel | None


def iter_labeled(
    manifest: Manifest,
    repos: Path,
    split: Split,
    notebook_labels: dict[str, NotebookLabelFile],
    competition_labels: dict[str, CompetitionLabel],
    excluded: set[str] = frozenset(),  # type: ignore[assignment]
    final: bool = False,
) -> Iterator[LabeledNotebook]:
    if split == "test" and not final:
        raise PermissionError("the test split is held out; run it once with final=True for reporting")
    for entry in manifest.in_split(split):
        comp = entry.competition or entry.name
        if entry.source != "kaggle" or comp not in notebook_labels:
            continue
        for rel, label in notebook_labels[comp].notebooks.items():
            path = repos / entry.name / rel
            if label.unsure or f"{entry.name}/{rel}" in excluded or not path.exists():
                continue
            src = load_source(path, rel)
            if src is not None and src.is_code:
                yield LabeledNotebook(comp, rel, src, label, competition_labels.get(comp))


def detectors_for(nb: LabeledNotebook, context: bool) -> tuple[Detector, ...]:
    if not context or nb.competition_label is None:
        return DETECTORS
    return with_profiles(DETECTORS, [competition_profile(nb.competition, nb.competition_label)])


def rule_findings(nb: LabeledNotebook, context: bool = True) -> list[Finding]:
    return run_detectors(nb.src, detectors_for(nb, context))


class TypeScore(BaseModel):
    tp: int = 0
    fp: int = 0
    fn: int = 0

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


def add_counts(scores: dict[str, TypeScore], predicted: set[str], actual: set[str]) -> None:
    for t in predicted | actual:
        s = scores.setdefault(t, TypeScore())
        s.tp += t in predicted and t in actual
        s.fp += t in predicted and t not in actual
        s.fn += t in actual and t not in predicted


def match_lines(
    findings: Iterable[tuple[str, int]], issues: Iterable[tuple[str, int]], tolerance: int = LINE_TOLERANCE
) -> tuple[int, int, int, dict[str, TypeScore]]:
    """Greedy one-to-one matching of (type, line) findings to labeled issues of the same type."""
    by_type: dict[str, TypeScore] = defaultdict(TypeScore)
    remaining = sorted(set(issues))
    for t, line in sorted(set(findings)):
        hit = next((i for i in remaining if i[0] == t and abs(i[1] - line) <= tolerance), None)
        if hit is not None:
            remaining.remove(hit)
            by_type[t].tp += 1
        else:
            by_type[t].fp += 1
    for t, _ in remaining:
        by_type[t].fn += 1
    tp = sum(s.tp for s in by_type.values())
    fp = sum(s.fp for s in by_type.values())
    fn = sum(s.fn for s in by_type.values())
    return tp, fp, fn, dict(by_type)


class NotebookResult(BaseModel):
    competition: str
    notebook: str
    labeled: list[str]
    found: list[str]


class LabelBenchReport(BaseModel):
    split: str
    context: bool
    notebooks: int
    relevant: int
    by_type: dict[str, TypeScore]  # notebook level
    by_type_lines: dict[str, TypeScore]  # line level
    results: list[NotebookResult]

    def micro(self, lines: bool = False) -> TypeScore:
        scores = (self.by_type_lines if lines else self.by_type).values()
        return TypeScore(
            tp=sum(s.tp for s in scores), fp=sum(s.fp for s in scores), fn=sum(s.fn for s in scores)
        )


def run_label_bench(
    notebooks: Iterable[LabeledNotebook], split: str, context: bool = True
) -> LabelBenchReport:
    by_type: dict[str, TypeScore] = {}
    by_type_lines: dict[str, TypeScore] = defaultdict(TypeScore)
    results: list[NotebookResult] = []
    n = relevant = 0
    for nb in notebooks:
        n += 1
        relevant += nb.label.relevant
        found = rule_findings(nb, context)
        predicted = {f.type for f in found}
        actual = {i.type for i in nb.label.issues}
        add_counts(by_type, predicted, actual)
        _, _, _, lines = match_lines(
            ((f.type, f.location.line_start or 0) for f in found), ((i.type, i.line) for i in nb.label.issues)
        )
        for t, s in lines.items():
            agg = by_type_lines[t]
            agg.tp, agg.fp, agg.fn = agg.tp + s.tp, agg.fp + s.fp, agg.fn + s.fn
        results.append(NotebookResult(competition=nb.competition, notebook=nb.rel, labeled=sorted(actual),
                                      found=sorted(predicted)))  # fmt: skip
    return LabelBenchReport(
        split=split,
        context=context,
        notebooks=n,
        relevant=relevant,
        by_type=dict(sorted(by_type.items())),
        by_type_lines=dict(sorted(by_type_lines.items())),
        results=results,
    )


def _pct(v: float | None) -> str:
    return "  –  " if v is None else f"{v:5.1%}"


def score_table(scores: dict[str, TypeScore], title: str) -> list[str]:
    rows = [title, f"{'type':45} {'P':>6} {'R':>6} {'F1':>6} {'tp':>4} {'fp':>4} {'fn':>4}"]
    for t, s in scores.items():
        rows.append(f"{t:45} {_pct(s.precision)} {_pct(s.recall)} {_pct(s.f1)} {s.tp:4d} {s.fp:4d} {s.fn:4d}")
    return rows


def summarize_label_bench(r: LabelBenchReport) -> str:
    out = [
        (
            f"split {r.split}: {r.notebooks} labeled notebooks ({r.relevant} train a model); "
            f"competition context {'on' if r.context else 'off'}"
        ),
        "",
        *score_table(r.by_type, "notebook level (issue type present in the notebook)"),
    ]
    m = r.micro()
    out += [
        f"{'micro':45} {_pct(m.precision)} {_pct(m.recall)} {_pct(m.f1)} {m.tp:4d} {m.fp:4d} {m.fn:4d}",
        "",
    ]
    out += score_table(r.by_type_lines, f"line level (same type within {LINE_TOLERANCE} lines)")
    ml = r.micro(lines=True)
    out.append(
        f"{'micro':45} {_pct(ml.precision)} {_pct(ml.recall)} {_pct(ml.f1)} {ml.tp:4d} {ml.fp:4d} {ml.fn:4d}"
    )
    return "\n".join(out)
