# FaultLine

Finds vulnerabilities in ML data pipelines (data poisoning, drift, leakage; pipeline
vulnerabilities), hardens the most critical points with checks, and uses agents to fix simple
problems and escalate complex ones. Plan: `docs/PLAN.md`. Phases and scope: `docs/PHASES.md`. Past decisions (don't
re-litigate): `docs/DECISIONS.md`.

## Layout
```
src/faultline/
  schema.py          Finding, PipelineGraph: the contract (FINDING_TYPES is the taxonomy)
  discovery/         sources.py (files, notebooks → code), graph.py (files ↔ datasets)
  detectors/         pipeline_vulns.py, leakage.py; registry in __init__.py
  data/              data detectors: drift.py, poisoning.py, overlap.py (frame.py, stats.py helpers)
  inject/code.py     fault mutators for real code (labeled ground truth)
  inject/data.py     data fault injectors (drift, poisoning, leaked duplicates) with exact rows/columns
  eval/              corpus.py (manifest, fetch), discover.py (GitHub/Kaggle), dedupe.py + build.py,
                     bench.py (injected recall, real-code precision), yang.py (public labels)
  scan.py, cli.py
corpus/manifest.yaml   real repos with train/dev/test split; checkouts in corpus/repos/ (ignored)
corpus/triage/         hand labels (tp/fp) for findings on untouched real code
```

## Commands (Windows or Linux; tools run from .venv)
```
make setup                 # .venv (Python 3.11) + pip install -e .[dev,data]
make test lint             # pytest; ruff check + format --check; mypy
make scan PIPELINE=path    # scan a repo
make corpus                # fetch pinned corpus repos
make bench SPLIT=train     # train | dev ; test needs: faultline bench --split test --final
faultline corpus discover --github N | --kaggle N   # add repos / competitions (Kaggle needs token)
faultline corpus dedupe   # exact + near-duplicate removal across splits (bench skips them)
faultline benchmark yang --split dev [--fetch]      # public labeled leakage set (test half: --final)
faultline data compare REF.csv CUR.csv --target y   # drift + poisoning; data overlap TRAIN.csv TEST.csv
```

## Rules
1. Detectors return `Finding` from `faultline.schema`; new types go in `FINDING_TYPES` first.
2. Never put raw secrets or PII in findings: mask them (`pipeline_vulns.mask`).
3. Splits are by repository. Write rules against `train`, calibrate on `dev`, never tune on
   `test`. Don't change rules just to fix a dev result: reproduce it in a unit test written from
   the train side, or log it in PLAN.md "Open issues".
4. Report injected recall only together with real-code precision and public-benchmark results.
5. Test first. Every new rule needs a true-positive and a false-positive test.
6. Agents only open PRs or issues; production collectors are read-only and aggregate-only.
7. Python 3.11+, full type hints, pydantic v2, no bare `except`. Ask before adding a dependency.
8. Fake credentials in tests are built by joining strings at runtime (GitHub push protection).
