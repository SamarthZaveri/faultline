"""faultline scan | corpus fetch | bench"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from faultline.eval import yang
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
from faultline.scan import scan
from faultline.schema import Severity

REPORTS = Path("reports")


def _cmd_scan(args: argparse.Namespace) -> int:
    result = scan(Path(args.path))
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="faultline")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="scan a pipeline repo")
    s.add_argument("path")
    s.add_argument("--out", help="write the full result (graph + findings) as JSON")
    s.add_argument("--fail-on", choices=[x.value for x in Severity], help="exit 1 at this severity or worse")
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
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
