"""faultline scan | corpus fetch | bench"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from faultline.eval.bench import run_bench, summarize
from faultline.eval.corpus import DEFAULT_MANIFEST, DEFAULT_REPOS, fetch, load_manifest
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


def _cmd_corpus_fetch(args: argparse.Namespace) -> int:
    manifest = load_manifest(Path(args.manifest))
    for entry in manifest.repos:
        if args.split and entry.split != args.split:
            continue
        print(f"fetching {entry.name} ({entry.split}) @ {entry.commit[:10]}", flush=True)
        fetch(entry, Path(args.repos))
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
    )
    REPORTS.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out = REPORTS / f"bench_{args.split}_{stamp}.json"
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(summarize(report))
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

    b = sub.add_parser("bench", help="benchmark detectors on one corpus split")
    b.add_argument("--split", choices=["train", "dev", "test"], default="train")
    b.add_argument("--final", action="store_true", help="required for the held-out test split")
    b.add_argument("--per-mutator", type=int, default=3)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    b.add_argument("--repos", default=str(DEFAULT_REPOS))
    b.set_defaults(func=_cmd_bench)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
