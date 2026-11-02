# Evaluation plan

Purpose: measure FaultLine honestly and keep Claude (and you) from tuning against the test.

## Datasets and splits
- **DVP (Damn Vulnerable Pipeline):** 40 planted faults, ground truth in `fixtures/dvp_faults.yaml`. Used as the regression/dev suite.
- **Public benchmarks (held out, never tuned on):** BackdoorBench/ART attacks on MNIST & CIFAR-10; WILDS/Wild-Time and River streams for drift; CVEfixes/DiverseVul for code vulns; Kaggle/KGTorrent notebooks for leakage; ai4privacy for PII.
- **Real pipelines:** 3-5 open-source Airflow/dbt/Dagster/MLflow repos (no ground truth; manual triage).
- **Split policy:** split by repository/pipeline, never by sample or file. Thresholds are calibrated on `dev` repos only; report on `test` repos once at the end.

## Fault injection
Generate labeled data by injecting into clean data/code: label flips (1-20%), BadNets patch triggers (1-5%), clean-label attacks, OOD rows, mean/variance/covariance shifts (sudden, gradual), unit changes, missing spikes, leakage patterns (fit-before-split, target-derived feature, duplicates, time shuffle), seeded secrets/injection/unsafe loaders.

## Metrics and targets
| Area | Metric | Target |
|---|---|---|
| Poisoning | recall @ FPR ≤5% | ≥85% (5% flip, BadNets); report stealthier attacks honestly |
| Drift | detection delay; false alarms; feature localization | ≤ Evidently/PSI baseline FA; localization ≥80% |
| Leakage | F1 on held-out repos | ≥0.85 |
| Pipeline vulns | precision (post-triage) / recall | ≥80% / ≥85% |
| Criticality | Kendall tau vs hand ranking on DVP | ≥0.7 |
| Triage | tier agreement | ≥85% |
| Fixer | Tier 1 fix success, zero regressions | ≥70% |
| Escalation | reviewer rating (5+ reviewers) | ≥4/5 |
| Product | time to topology + first findings | <15 min |

## Baselines
Evidently/PSI for drift; Bandit/Semgrep default rulesets for vulns; random + rule-only ablation for LLM triage; no-check-placement baseline for criticality.

## Ablations
Rules-only vs rules+LLM triage; cleanlab vs +activation clustering; with/without code-transformer for leakage.

## Reporting
`make bench` writes `reports/bench_<date>.json` and a Markdown summary: per-detector P/R/F1, confusion by fault id, failures listed. Track in MLflow.

## Success milestones
1. MVP: all targets met on DVP, unattended end-to-end demo.
2. Credibility: real unknown issues found in ≥3 external pipelines, ≥1 fix merged or issue acknowledged.
3. Showstopper: public benchmark + write-up, plus the adoption number you set in advance (write it in DECISIONS.md at project start).
