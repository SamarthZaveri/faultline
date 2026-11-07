"""faultline scan | corpus | bench | benchmark | data | baseline"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from faultline.contract import load_profiles, save_profile
from faultline.eval import baseline, yang
from faultline.eval.bench import run_bench, summarize
from faultline.eval.build import dedupe_corpus, load_excluded, save_dedupe
from faultline.eval.corpus import (
    DEFAULT_DEDUPE,
    DEFAULT_MANIFEST,
    DEFAULT_REPOS,
    Manifest,
    fetch,
    fetch_kaggle,
    kernel_changed,
    load_manifest,
    save_manifest,
)
from faultline.eval.discover import (
    KaggleApiSource,
    discover_github,
    discover_kaggle,
    resolve_author_overlap,
)
from faultline.eval.extract import DEFAULT_EXTRACTS, write_extracts
from faultline.scan import scan
from faultline.schema import Finding, Severity

REPORTS = Path("reports")


def _cmd_scan(args: argparse.Namespace) -> int:
    profiles = load_profiles(Path(args.profiles)) if args.profiles else []
    result = scan(Path(args.path), profiles)
    if args.out:
        Path(args.out).write_text(result.model_dump_json(indent=2), encoding="utf-8")
    kinds: dict[str, int] = {}
    for n in result.graph.nodes:
        kinds[n.kind.value] = kinds.get(n.kind.value, 0) + 1
    print(f"{result.files_scanned} files, graph: {kinds}, parse errors: {len(result.parse_errors)}")
    for f in result.findings:
        loc = f"{f.location.file}:{f.location.line_start}"
        print(f"[{f.severity:8s}] {f.type:42s} {loc}  {f.title}")
    print(f"{len(result.findings)} findings")
    if args.fail_on:
        threshold = Severity(args.fail_on).rank
        if any(f.severity.rank <= threshold for f in result.findings):
            return 1
    return 0


# Projects that use ML pipelines (not ML frameworks/tools), where data and pipeline bugs live.
GITHUB_QUERIES = (
    "topic:kaggle-solution stars:>20",
    "topic:kaggle-competition language:Jupyter Notebook stars:>20",
    "end-to-end machine learning project language:Jupyter Notebook stars:>20",
    "topic:machine-learning-projects language:Jupyter Notebook stars:>20",
    "topic:data-science-project stars:>20",
    "mlops project pipeline language:Python stars:>20",
    "topic:airflow topic:machine-learning stars:>20",
    "churn prediction language:Jupyter Notebook stars:>20",
    "fraud detection model language:Jupyter Notebook stars:>20",
    "credit risk model language:Jupyter Notebook stars:>20",
    "house price prediction language:Jupyter Notebook stars:>20",
    "feature engineering pipeline language:Jupyter Notebook stars:>20",
)
COMPETITIONS = Path(DEFAULT_MANIFEST).parent / "kaggle_competitions.txt"


def _kaggle_source() -> KaggleApiSource | None:
    try:
        return KaggleApiSource()
    except ModuleNotFoundError as e:
        print(f"kaggle client unavailable ({e.name} missing): pip install -e .[corpus]")
    except (OSError, ValueError, SystemExit) as e:  # kaggle's client exits when not logged in
        print(rf"kaggle auth failed ({e}). Log in once with: .venv\Scripts\kaggle auth login")
    return None


def _cmd_corpus_discover(args: argparse.Namespace) -> int:
    path = Path(args.manifest)
    manifest = load_manifest(path)
    new = []
    if args.github:
        found = discover_github(list(GITHUB_QUERIES), manifest, limit=args.github)
        print(f"github: {len(found)} new repos")
        new += found
    if args.kaggle:
        source = _kaggle_source()
        if source is None:
            return 1
        known = {e.competition for e in manifest.repos}
        comps = [c.strip() for c in COMPETITIONS.read_text(encoding="utf-8").splitlines()]
        comps = [c for c in comps if c and not c.startswith("#") and c not in known]
        found, skipped = discover_kaggle(source, comps, per_competition=args.kaggle)
        print(f"kaggle: {len(found)} competitions, {sum(len(e.kernels) for e in found)} notebooks")
        if skipped:
            print(f"kaggle: skipped (not found / no python notebooks): {', '.join(skipped)}")
        new += found
    entries, dropped = resolve_author_overlap([*manifest.repos, *new])
    if dropped:
        print(f"dropped {len(dropped)} notebooks whose author lives in another split")
    save_manifest(Manifest(repos=entries), path)
    print(f"manifest: {len(entries)} entries -> {path}")
    return 0


def _cmd_corpus_fetch(args: argparse.Namespace) -> int:
    path = Path(args.manifest)
    manifest = load_manifest(path)
    repos = Path(args.repos)
    source = None
    updated = []
    failed: list[str] = []
    for entry in manifest.repos:
        if args.split and entry.split != args.split:
            updated.append(entry)
            continue
        if entry.source == "github":
            print(f"fetching {entry.name} ({entry.split}) @ {(entry.commit or '')[:10]}", flush=True)
            try:
                fetch(entry, repos)
            except subprocess.CalledProcessError as e:
                failed.append(entry.name)
                print(f"  failed: {(e.stderr or '').strip().splitlines()[-1:] or e}")
            updated.append(entry)
            continue
        source = source or _kaggle_source()
        if source is None:
            return 1
        print(f"fetching {entry.name} ({entry.split}): {len(entry.kernels)} notebooks", flush=True)
        entry = fetch_kaggle(entry, source, repos)
        changed = [k.ref for k in entry.kernels if kernel_changed(k, repos, entry.name)]
        if changed:
            print(f"  warning: changed since first pull: {', '.join(changed)}")
        updated.append(entry)
    save_manifest(Manifest(repos=updated), path)
    if failed:
        print(f"{len(failed)} failed (bench reports them as missing): {', '.join(failed)}")
    return 0


def _cmd_corpus_dedupe(args: argparse.Namespace) -> int:
    report = dedupe_corpus(load_manifest(Path(args.manifest)), Path(args.repos))
    save_dedupe(report, DEFAULT_DEDUPE)
    near = sum(d.kind == "near" for d in report.excluded.values())
    print(f"files by split: {report.files_by_split}")
    print(f"kept by split:  {report.kept_by_split}")
    print(
        f"excluded {len(report.excluded)} ({near} near-duplicates); {report.cross_split} copied across splits"
    )
    return 0


def _cmd_corpus_extract(args: argparse.Namespace) -> int:
    if args.split == "test" and not args.final:
        print("the test split is held out: label it last, with --final")
        return 2
    out = Path(args.out)
    n = write_extracts(
        load_manifest(Path(args.manifest)), Path(args.repos), args.split, load_excluded(DEFAULT_DEDUPE), out
    )
    print(f"{n} notebooks -> {out / args.split}")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    report = run_bench(
        load_manifest(Path(args.manifest)),
        args.split,
        Path(args.repos),
        per_mutator=args.per_mutator,
        seed=args.seed,
        final=args.final,
        triage_dir=Path(args.manifest).parent / "triage",
        excluded=load_excluded(DEFAULT_DEDUPE),
    )
    REPORTS.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out = REPORTS / f"bench_{args.split}_{stamp}.json"
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(summarize(report))
    print(f"\nwrote {out}")
    return 0


def _cmd_benchmark_yang(args: argparse.Namespace) -> int:
    root = yang.fetch() if args.fetch else yang.DEFAULT_DIR
    if not (root / "ground-truth.csv").exists():
        print("benchmark not downloaded; rerun with --fetch")
        return 1
    report = yang.evaluate(yang.load_benchmark(root), args.split, final=args.final)
    REPORTS.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out = REPORTS / f"yang2022_{args.split}_{stamp}.json"
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(yang.summarize(report))
    print(f"\nwrote {out}")
    return 0


def _read_table(path: str) -> Any:
    import pandas as pd  # the data extra is only needed for data commands

    return pd.read_csv(path, sep="\t" if path.lower().endswith(".tsv") else ",")


def _report_data(findings: list[Finding], out: str | None) -> int:
    if out:
        payload = [f.model_dump(mode="json") for f in findings]
        Path(out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for f in findings:
        cols = ",".join(f.location.columns) or "-"
        print(f"[{f.severity:8s}] {f.type:38s} {cols[:40]:40s} {f.title}")
    print(f"{len(findings)} findings")
    return 0


def _cmd_data_compare(args: argparse.Namespace) -> int:
    from faultline.data.drift import DriftDetector
    from faultline.data.poisoning import PoisoningDetector

    ref, cur = _read_table(args.reference), _read_table(args.current)
    dataset = args.dataset or args.current
    findings = DriftDetector().compare(ref, cur, dataset=dataset, target=args.target, seed=args.seed)
    findings += PoisoningDetector().check(
        cur, dataset=dataset, target=args.target, reference=ref, seed=args.seed
    )
    return _report_data(findings, args.out)


def _cmd_data_profile(args: argparse.Namespace) -> int:
    from faultline.data.profile import profile_frame

    data = Path(args.data)
    profile = profile_frame(_read_table(args.data), args.dataset or data.name, target=args.target)
    out = Path(args.out) if args.out else Path("profiles") / f"{data.stem}.yaml"
    save_profile(profile, out)
    task = f"{profile.task}" + (
        f", smallest class {profile.minority_share:.1%}" if profile.minority_share else ""
    )
    print(f"{profile.name}: {profile.rows} rows, {len(profile.columns)} columns, {task}")
    print(f"  time columns: {profile.time_columns or '-'} (ordered: {profile.time_ordered})")
    print(f"  entity keys: {profile.group_keys or '-'}; duplicate rows: {profile.duplicate_rate:.1%}")
    print(f"wrote {out}  (use: faultline scan REPO --profiles {out.parent})")
    return 0


def _cmd_data_overlap(args: argparse.Namespace) -> int:
    from faultline.data.overlap import SplitOverlapDetector

    train, test = _read_table(args.train), _read_table(args.test)
    findings = SplitOverlapDetector().check(
        train, test, dataset=args.dataset or args.test, target=args.target
    )
    return _report_data(findings, args.out)


def _cmd_baseline_freeze(args: argparse.Namespace) -> int:
    out = baseline.freeze(args.name, REPORTS)
    print(f"froze baseline -> {out}")
    print((out / "SUMMARY.md").read_text(encoding="utf-8"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="faultline")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="scan a pipeline repo")
    s.add_argument("path")
    s.add_argument("--out", help="write the full result (graph + findings) as JSON")
    s.add_argument("--fail-on", choices=[x.value for x in Severity], help="exit 1 at this severity or worse")
    s.add_argument(
        "--profiles", help="folder of dataset profiles (faultline data profile) to check the code against"
    )
    s.set_defaults(func=_cmd_scan)

    c = sub.add_parser("corpus", help="manage the real-pipeline corpus")
    csub = c.add_subparsers(dest="corpus_cmd", required=True)
    f = csub.add_parser("fetch", help="fetch pinned repos from the manifest")
    f.add_argument("--split", choices=["train", "dev", "test"])
    f.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    f.add_argument("--repos", default=str(DEFAULT_REPOS))
    f.set_defaults(func=_cmd_corpus_fetch)
    d = csub.add_parser("discover", help="add new Kaggle competitions / GitHub repos to the manifest")
    d.add_argument(
        "--kaggle", type=int, default=0, metavar="N", help="top-N python notebooks per competition"
    )
    d.add_argument("--github", type=int, default=0, metavar="N", help="up to N new GitHub repos")
    d.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    d.set_defaults(func=_cmd_corpus_discover)
    dd = csub.add_parser("dedupe", help="find exact/near-duplicate files across the corpus")
    dd.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    dd.add_argument("--repos", default=str(DEFAULT_REPOS))
    dd.set_defaults(func=_cmd_corpus_dedupe)
    x = csub.add_parser("extract", help="write compact notebook extracts for hand labeling")
    x.add_argument("--split", choices=["train", "dev", "test"], default="train")
    x.add_argument("--final", action="store_true", help="required for the held-out test split")
    x.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    x.add_argument("--repos", default=str(DEFAULT_REPOS))
    x.add_argument("--out", default=str(DEFAULT_EXTRACTS))
    x.set_defaults(func=_cmd_corpus_extract)

    b = sub.add_parser("bench", help="benchmark detectors on one corpus split")
    b.add_argument("--split", choices=["train", "dev", "test"], default="train")
    b.add_argument("--final", action="store_true", help="required for the held-out test split")
    b.add_argument("--per-mutator", type=int, default=3)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    b.add_argument("--repos", default=str(DEFAULT_REPOS))
    b.set_defaults(func=_cmd_bench)
    pb = sub.add_parser("benchmark", help="public labeled benchmarks")
    pbsub = pb.add_subparsers(dest="benchmark_cmd", required=True)
    y = pbsub.add_parser("yang", help="Yang et al. ASE'22 notebook leakage labels")
    y.add_argument("--split", choices=["dev", "test"], default="dev")
    y.add_argument("--final", action="store_true", help="required for the held-out test half")
    y.add_argument(
        "--fetch", action="store_true", help="download the materials first (local only, not committed)"
    )
    y.set_defaults(func=_cmd_benchmark_yang)

    dt = sub.add_parser("data", help="check data (CSV/TSV) for drift, poisoning and split overlap")
    dtsub = dt.add_subparsers(dest="data_cmd", required=True)
    dc = dtsub.add_parser("compare", help="reference vs current data: drift + poisoning")
    dc.add_argument("reference", help="trusted data the pipeline was built on")
    dc.add_argument("current", help="new batch / later snapshot")
    dc.set_defaults(func=_cmd_data_compare)
    do = dtsub.add_parser("overlap", help="rows shared by train and test")
    do.add_argument("train")
    do.add_argument("test")
    do.set_defaults(func=_cmd_data_overlap)
    dp = dtsub.add_parser("profile", help="profile a dataset (aggregates only) for scan --profiles")
    dp.add_argument("data")
    dp.set_defaults(func=_cmd_data_profile)
    for p_ in (dc, do, dp):
        p_.add_argument("--target", help="label column")
        p_.add_argument("--dataset", help="name used in findings/profile (default: the file path or name)")
        p_.add_argument(
            "--out", help="output file (findings JSON; profile YAML, default profiles/<name>.yaml)"
        )
    for p_ in (dc, do):
        p_.add_argument("--seed", type=int, default=0)

    bl = sub.add_parser("baseline", help="freeze benchmark results as a named, committed baseline")
    blsub = bl.add_subparsers(dest="baseline_cmd", required=True)
    fr = blsub.add_parser(
        "freeze", help="copy the latest reports into results/baselines/<name>/ + SUMMARY.md"
    )
    fr.add_argument("name", help="e.g. rules-v0.2 (baselines are never overwritten)")
    fr.set_defaults(func=_cmd_baseline_freeze)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
