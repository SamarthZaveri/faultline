# FaultLine plan

## Goal
Detect vulnerabilities in ML data pipelines (ingestion, transformation, and before data is used
for training), then:
1. **Harden**: add checks at the most critical points of the pipeline.
2. **Remediate** with a team of agents: rewrite code for simple problems (as PRs), escalate
   complex ones with an evidence report.

Problem families:
- **Data**: poisoning, drift, leakage (data seen by the model when it should not be).
- **Pipeline**: secrets, injection, unsafe deserialization, dependencies, supply chain, config/IAM,
  unvalidated trust boundaries. Each gets a report plus a small fix or an escalation.

Works on pre-production pipelines (full scans, agents may open PRs) and production pipelines
(read-only, summary statistics only, alerts and reports). Continuous monitoring is for
test/staging; production is passive.

Dashboard (Python + HTML): pipeline topology, most critical sites, data flow, installed checks,
findings and reports.

## Architecture
```
repo ─► discovery (sources.py: files+notebooks; graph.py: files, datasets, trust boundaries)
     ─► detectors (Finding)  ─► scoring (criticality per node) ─► checks at top nodes
     ─► agents (triage → fix PR | escalate)  ─► store (SQLite→Postgres) + API + dashboard
```
- `schema.py` is the contract: detectors emit `Finding`, discovery emits `PipelineGraph`.
  Everything downstream reads only these.
- Rules and statistics first; ML/LLM only where it measurably beats rules on dev.

## Evaluation: train and test on real pipelines and datasets
Ground truth comes from two sources, and both are reported:
1. **Faults injected into real repos and data** (`inject/`): known type and location.
2. **Public labeled benchmarks**: Yang et al. leakage notebooks, CVEfixes, BackdoorBench-style
   attacks, WILDS drift. Not yet added (M2).

In addition, findings on untouched real code are hand-triaged in `corpus/triage/<repo>.yaml`
to measure precision on real issues.

Corpus rules (`corpus/manifest.yaml`):
- Split by **repository**, fixed before tuning.
  - `train`: write rules and templates against these.
  - `dev`: calibrate thresholds and confidence.
  - `test`: held out; `bench --split test --final`, run once for the final report.
- Commits pinned. Identical files (forks, translations) are counted once.
- Injected recall is optimistic because the same author writes templates and rules. Never
  report it without the public-benchmark and triaged real-code numbers next to it.

## Phases and scope
See `docs/PHASES.md` (single source for phases, scope, and where ML/DL/RL is used).

## Open issues found on dev (M1, not yet fixed; fixing must not use dev as training data)
- **Nested splits**: a fit on `X_train_full` before splitting it into train/valid is skipped as
  "fit on a training subset", but the injector counts it as leakage into validation. Decide
  the definition, then make the injector and detector agree.
- **Notebook name reuse**: flow-insensitive dataflow links unrelated sections that reuse `X`/`y`.
  This causes most of the 20 dev leakage hits (15 in one SVM notebook). Needs reaching
  definitions, and should require the fitted object or its output to reach the evaluation.
- **SQL severity**: f-strings that interpolate internal config (e.g. `dataset_name`) are flagged
  `high`. Severity should depend on whether the values are tainted (function params or external
  input).
- **Secrets**: local-dev defaults in docker-compose files are flagged (true by rule, low risk).
  A credentials file *path* (`"my_service_account.json"`) is a false positive.
