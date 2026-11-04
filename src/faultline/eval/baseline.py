"""Freeze the latest benchmark reports as a named, committed baseline.

reports/ is scratch (ignored by git); results/baselines/<name>/ is the record that README numbers
and later comparisons (e.g. rules vs rules+ML) point to. Code snippets are dropped from the
frozen copy so no third-party code is committed.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

from faultline.eval.bench import BenchReport
from faultline.eval.yang import KINDS, YangReport

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS = ROOT / "results" / "baselines"


def _latest(reports: Path, prefix: str) -> Path:
    files = sorted(reports.glob(f"{prefix}_*.json"))  # timestamps sort lexically
    if not files:
        raise FileNotFoundError(f"no {prefix}_*.json in {reports}; run the benchmark first")
    return files[-1]


def current_commit() -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head.stdout.strip() + (" (+uncommitted changes)" if dirty.stdout.strip() else "")


def _pct(v: float | None) -> str:
    return "–" if v is None else f"{v:.1%}"


def _bench_section(r: BenchReport) -> list[str]:
    n_by_type = {t: sum(i.expected_type == t for i in r.injections) for t in r.recall_by_type}
    lines = [
        f"### Corpus `{r.split}` split",
        f"{len(r.repos)} repos, {r.files_scanned} unique files, {r.loc_scanned:,} lines "
        f"({r.duplicates_skipped} duplicate files skipped).",
        "",
        "| Injected fault type | Recall | n |",
        "|---|---|---|",
        *[f"| {t} | {_pct(v)} | {n_by_type[t]} |" for t, v in r.recall_by_type.items()],
        f"| **overall** | **{_pct(r.recall_overall)}** | {len(r.injections)} |",
        "",
        f"Findings on untouched real code: {len(r.clean_findings)}"
        + (f" ({len(r.clean_findings) / (r.loc_scanned / 1000):.2f} per 1K lines)" if r.loc_scanned else ""),
        "",
        "| Finding type | Count |",
        "|---|---|",
        *[f"| {t} | {n} |" for t, n in r.clean_findings_by_type.items()],
        "",
        f"Hand-triaged: {r.triaged}"
        + (f", precision {_pct(r.precision_on_real)}" if r.precision_on_real is not None else " (precision not yet measured)"),
        "",
    ]
    return lines


def _yang_section(y: YangReport) -> list[str]:
    lines = [
        f"### Yang et al. (ASE'22) labeled leakage notebooks, `{y.split}` half",
        f"{y.labeled} labeled notebooks; notebook-level labels. Baseline = the paper's own tool (published outputs).",
        "",
        "| Kind | FaultLine P | FaultLine R | FaultLine F1 | Paper tool P | Paper tool R | Paper tool F1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for k in KINDS:
        o, b = y.ours[k], y.baseline[k]
        lines.append(
            f"| {k} | {_pct(o.precision)} | {_pct(o.recall)} | {_pct(o.f1)} "
            f"| {_pct(b.precision)} | {_pct(b.recall)} | {_pct(b.f1)} |"
        )
    lines += [
        "",
        "pre = preprocessing before split; overlap = test rows used in training; "
        "multi = test set reused for model selection.",
        "",
    ]
    return lines


CAVEATS = """## Caveats (keep these next to the numbers)
- **Rule-based only.** No ML model is trained or used; this is the baseline later ML components must beat.
- **Injected-fault recall is optimistic:** the same author wrote the fault templates and the rules.
- **Yang dev numbers are somewhat optimistic:** some rules (slicing splits, fit-on-all-data) were
  added after looking at dev misses. The held-out test half has not been run; it is the clean measure.
- **The paper's tool has a home advantage:** its authors also labeled the benchmark.
- **Real-code precision is not yet measured:** findings on untouched code still need hand triage.
"""


def freeze(name: str, reports: Path, results: Path = DEFAULT_RESULTS, commit: str | None = None) -> Path:
    out = results / name
    if out.exists():
        raise FileExistsError(f"{out} already exists; baselines are immutable, pick a new name")
    train = BenchReport.model_validate_json(_latest(reports, "bench_train").read_text(encoding="utf-8"))
    dev = BenchReport.model_validate_json(_latest(reports, "bench_dev").read_text(encoding="utf-8"))
    yang = YangReport.model_validate_json(_latest(reports, "yang2022_dev").read_text(encoding="utf-8"))
    out.mkdir(parents=True)
    for label, report in (("bench_train", train), ("bench_dev", dev)):
        for f in report.clean_findings:
            f.evidence.snippet = None
        (out / f"{label}.json").write_text(report.model_dump_json(indent=2), encoding="utf-8", newline="\n")
    (out / "yang2022_dev.json").write_text(yang.model_dump_json(indent=2), encoding="utf-8", newline="\n")
    summary = [
        f"# Baseline `{name}`",
        "",
        f"Frozen {datetime.now(UTC):%Y-%m-%d} from commit `{commit or current_commit()}`. "
        "Detectors: rule-based leakage + pipeline-vulnerability rules.",
        "",
        "## Results",
        "",
        *_yang_section(yang),
        *_bench_section(train),
        *_bench_section(dev),
        CAVEATS,
    ]
    (out / "SUMMARY.md").write_text("\n".join(summary), encoding="utf-8", newline="\n")
    return out
