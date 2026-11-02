# Roadmap (12 weeks). Update status after each task.

Legend: [ ] todo, [~] in progress, [x] done. Each task has acceptance criteria (AC).

## Week 1: Foundations
- [~] T1.1 Repo, Makefile, CI (lint+test), docker-compose (postgres, redis). AC: `make setup test lint up` pass on clean checkout. _Status: setup/test/lint verified on a clean copy (Windows, sh + cmd.exe); `up` not yet verified (Docker skipped); CI not yet run (no remote)._
- [x] T1.2 Finding/Check pydantic models + JSON Schema export. AC: tests/test_finding_schema.py passes; schema file generated.
- [ ] T1.3 Damn Vulnerable Pipeline v0 (10 faults from fixtures/dvp_faults.yaml). AC: pipeline runs end to end on sample data; faults are present at the listed locations.

## Week 2: Discovery
- [ ] T2.1 Static parser: extract nodes (sources, transforms, splits, sinks) from Python/SQL. AC: DVP yields expected graph (golden file).
- [ ] T2.2 OpenLineage ingestion into Postgres graph. AC: events from a sample Airflow run produce matching nodes/edges.
- [ ] T2.3 Graph API `GET /pipelines/{id}/graph`. AC: returns nodes, edges, trust-boundary flags.

## Week 3: Pipeline-vuln detectors
- [ ] T3.1 Semgrep ruleset (secrets, SQL/shell injection, unsafe deserialization). AC: finds PV-01..PV-05, PV-09 with recall 100% on DVP.
- [ ] T3.2 Dependency checks (unpinned, OSV). AC: PV-06, PV-07.
- [ ] T3.3 Supply chain + config lint (unverified downloads, IAM, buckets). AC: PV-08, PV-10, PV-13, PV-14.

## Week 4: Drift + quality
- [ ] T4.1 Univariate drift (PSI, KS, chi-square). AC: DR-01, DR-03, DR-06.
- [ ] T4.2 Classifier two-sample + MMD + feature localization. AC: DR-01..DR-05 localized correctly ≥80%.
- [ ] T4.3 Streaming (ADWIN/DDM), quality checks. AC: DR-02, DR-07.

## Week 5: Leakage + PII
- [ ] T5.1 AST dataflow engine for ML code. AC: LK-01, LK-04, LK-05, LK-09.
- [ ] T5.2 Data-level leakage (exact/near duplicates, single-feature AUC, MI, temporal order). AC: LK-02, LK-03, LK-06, LK-07.
- [ ] T5.3 PII/secrets in data (Presidio + regex/entropy). AC: LK-08.

## Week 6: Poisoning
- [ ] T6.1 cleanlab label-error detector. AC: PO-01, PO-02 recall ≥85% @ FPR ≤5%.
- [ ] T6.2 Spectral Signatures / Activation Clustering. AC: PO-03, PO-04.
- [ ] T6.3 Tabular anomaly + source profiling + flooding. AC: PO-05, PO-06, PO-07, PO-08.

## Week 7: Scoring + check injection
- [ ] T7.1 Criticality scorer. AC: top-5 nodes on DVP match hand-labeled ranking (Kendall tau ≥0.7).
- [ ] T7.2 Placement optimizer under budget. AC: covers ≥90% of risk mass within budget on DVP.
- [ ] T7.3 Check generators (GE/Pandera). AC: generated checks run and catch their target faults.

## Week 8: Agent team v1
- [ ] T8.1 Triage Agent. AC: tier routing ≥85% agreement with labels in fixtures/dvp_faults.yaml.
- [ ] T8.2 Fixer + Verifier (sandbox) for Tier 1. AC: ≥70% Tier 1 fixed with tests passing and zero regressions.
- [ ] T8.3 Escalation Agent evidence packs + GitHub issue/PR integration.

## Week 9: Dashboard v1
- [ ] T9.1 Topology view (React Flow) colored by criticality.
- [ ] T9.2 Critical sites, findings board, check registry.
- [ ] T9.3 Lineage trace view.

## Week 10: Monitoring + reports
- [ ] T10.1 Scheduled/CI scans, trend storage.
- [ ] T10.2 Prod passive collectors (aggregates only).
- [ ] T10.3 Report export (PDF/Markdown).

## Week 11: Evaluation
- [ ] T11.1 Full benchmark run per docs/EVAL_PLAN.md, ablations.
- [ ] T11.2 Scan 3-5 real open-source pipelines; triage results.
- [ ] T11.3 Threshold calibration on dev split only.

## Week 12: Launch
- [ ] T12.1 Docs, demo video, write-up, public repo.
- [ ] T12.2 Optional: contextual bandit prototype.

## Cut line if behind
Drop: poison attribution, GNN criticality, bandit, prod collectors. Keep: discovery, leakage, drift, pipeline vulns, agents, topology dashboard.
