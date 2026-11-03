"""Benchmark detectors on real repos from one split of the corpus.

Two kinds of ground truth:
  1. Injected faults: each mutator plants a fault in a real file; recall = planted faults found
     at the planted lines.
  2. Real findings on the untouched repos: unlabeled until triaged by hand in
     corpus/triage/<repo>.yaml ({labels: {finding_id: tp|fp}}); precision is computed on those.
The test split is held out: it only runs with final=True, for the one final report.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from faultline.detectors import run_detectors
from faultline.discovery.sources import iter_sources
from faultline.eval.corpus import Manifest, Split
from faultline.inject.code import MUTATORS
from faultline.schema import Finding


class Injection(BaseModel):
    repo: str
    file: str
    mutator: str
    expected_type: str
    line_start: int
    line_end: int
    detected: bool
    collateral: int  # other finding types raised inside the injected lines


class BenchReport(BaseModel):
    split: Split
    seed: int
    repos: list[str]
    missing: list[str] = Field(default_factory=list)
    files_scanned: int = 0
    loc_scanned: int = 0
    injections: list[Injection] = Field(default_factory=list)
    recall_by_type: dict[str, float] = Field(default_factory=dict)
    recall_overall: float | None = None
    clean_findings: list[Finding] = Field(default_factory=list)
    clean_findings_by_type: dict[str, int] = Field(default_factory=dict)
    duplicates_skipped: int = 0  # identical files (forks, translations) counted once
    triaged: int = 0
    precision_on_real: float | None = None


def _labels(triage_dir: Path | None, repo: str) -> dict[str, Literal["tp", "fp"]]:
    if triage_dir is None or not (triage_dir / f"{repo}.yaml").exists():
        return {}
    data = yaml.safe_load((triage_dir / f"{repo}.yaml").read_text(encoding="utf-8")) or {}
    return {str(k): v for k, v in (data.get("labels") or {}).items() if v in ("tp", "fp")}


def run_bench(
    manifest: Manifest,
    split: Split,
    repos_dir: Path,
    *,
    per_mutator: int = 3,
    seed: int = 0,
    final: bool = False,
    triage_dir: Path | None = None,
    excluded: set[str] | None = None,
) -> BenchReport:
    if split == "test" and not final:
        raise PermissionError("the test split is held out; run it once with final=True for reporting")
    report = BenchReport(split=split, seed=seed, repos=[])
    labels_tp = labels_fp = 0
    seen: set[str] = set()
    for entry in manifest.in_split(split):
        root = repos_dir / entry.name
        if not root.is_dir():
            report.missing.append(entry.name)
            continue
        report.repos.append(entry.name)
        sources = []
        for s in iter_sources(root):
            if excluded and f"{entry.name}/{s.path}" in excluded:
                report.duplicates_skipped += 1
                continue
            digest = hashlib.sha256(s.code.encode()).hexdigest()
            if digest in seen:
                report.duplicates_skipped += 1
                continue
            seen.add(digest)
            sources.append(s)
        report.files_scanned += len(sources)
        report.loc_scanned += sum(len(s.lines) for s in sources if s.is_code)

        clean = [f for s in sources for f in run_detectors(s)]
        report.clean_findings.extend(clean)
        labels = _labels(triage_dir, entry.name)
        for f in clean:
            if f.id in labels:
                labels_tp += labels[f.id] == "tp"
                labels_fp += labels[f.id] == "fp"

        for m in MUTATORS:
            rng = random.Random(f"{seed}:{entry.name}:{m.name}")
            candidates = [s for s in sources if m.applicable(s)]
            for src in rng.sample(candidates, min(per_mutator, len(candidates))):
                mutated, fault = m.apply(src, rng)
                in_range = [
                    f
                    for f in run_detectors(mutated)
                    if fault.line_start <= (f.location.line_start or 0) <= fault.line_end
                ]
                report.injections.append(
                    Injection(
                        repo=entry.name,
                        file=fault.file,
                        mutator=fault.mutator,
                        expected_type=fault.expected_type,
                        line_start=fault.line_start,
                        line_end=fault.line_end,
                        detected=any(f.type == fault.expected_type for f in in_range),
                        collateral=sum(f.type != fault.expected_type for f in in_range),
                    )
                )

    by_type: dict[str, list[bool]] = {}
    for inj in report.injections:
        by_type.setdefault(inj.expected_type, []).append(inj.detected)
    report.recall_by_type = {t: sum(v) / len(v) for t, v in sorted(by_type.items())}
    if report.injections:
        report.recall_overall = sum(i.detected for i in report.injections) / len(report.injections)
    report.clean_findings_by_type = dict(sorted(Counter(f.type for f in report.clean_findings).items()))
    report.triaged = labels_tp + labels_fp
    if report.triaged:
        report.precision_on_real = labels_tp / report.triaged
    return report


def summarize(report: BenchReport) -> str:
    lines = [
        (
            f"split={report.split} seed={report.seed} repos={len(report.repos)} "
            f"files={report.files_scanned} loc={report.loc_scanned}"
        ),
    ]
    if report.missing:
        lines.append(f"missing (run `make corpus`): {', '.join(report.missing)}")
    lines.append("")
    lines.append("Injected-fault recall")
    for t, r in report.recall_by_type.items():
        n = sum(i.expected_type == t for i in report.injections)
        lines.append(f"  {t:45s} {r:6.1%}  (n={n})")
    if report.recall_overall is not None:
        lines.append(f"  {'overall':45s} {report.recall_overall:6.1%}  (n={len(report.injections)})")
    lines.append("")
    kloc = max(report.loc_scanned / 1000, 1e-9)
    lines.append(
        f"Findings on untouched real code ({len(report.clean_findings)}, {len(report.clean_findings) / kloc:.2f}/KLOC)"
    )
    for t, n in report.clean_findings_by_type.items():
        lines.append(f"  {t:45s} {n}")
    if report.precision_on_real is None:
        lines.append("  precision: not yet triaged (label findings in corpus/triage/<repo>.yaml)")
    else:
        lines.append(f"  precision on {report.triaged} triaged: {report.precision_on_real:.1%}")
    return "\n".join(lines)
