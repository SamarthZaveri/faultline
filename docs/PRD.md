# FaultLine PRD (condensed; full version lives in the shared Claude Doc)

## Problem
ML systems are only as trustworthy as the data pipelines feeding them. Security tools check code, data-quality tools check tables, ML-security tools check models. Nobody owns the seam: poisoning, drift, leakage and pipeline weaknesses are found late or never.

## Solution
A closed loop: **Discover** (map the pipeline as a graph) → **Detect** (data + pipeline detectors) → **Harden** (install checks at the riskiest nodes) → **Remediate** (agents fix simple issues via PR, escalate complex ones) → **Observe** (dashboard, continuous monitoring in test/staging, passive monitoring in prod).

## Users
ML/data engineers who own pipelines; secondary: security/platform teams.

## Environments
- **Pre-production:** full scans (static analysis, sample data, test runs), continuous monitoring in CI/staging, agents may open PRs.
- **Production:** read-only, passive, summary statistics only, alerts and reports only. No direct modification.

## Non-goals (v1)
Model-level attacks after training, real-time blocking of prod traffic, streaming-engine internals, full orchestrator coverage.

## Problems detected
**Data:** poisoning (label flip, backdoor, clean-label, injected rows), drift (covariate, label, concept, schema semantics), leakage (train/test contamination, target, temporal, sensitive-data/PII exposure), quality/integrity.
**Pipeline:** hardcoded secrets, injection in transforms, unsafe deserialization, unpinned/vulnerable deps, unverified external datasets/models, over-permissive IAM/buckets, missing validation at trust boundaries, no lineage/versioning.
Each pipeline finding yields a report + small fix or escalation.

## Dashboard
Topology graph, critical sites, data flow/lineage, installed checks, findings and remediation board, exportable reports, continuous monitoring trends (test/staging).

## Success criteria
See `docs/EVAL_PLAN.md`. Summary: poisoning recall ≥85% @ ≤5% FPR; leakage F1 ≥0.85 on held-out repos; pipeline-vuln precision ≥80% / recall ≥85%; ≥70% of Tier 1 findings fixed by passing, non-regressing PRs; connection to first topology+findings under 15 minutes. Credibility: real unknown issues in ≥3 external pipelines. Showstopper: published benchmark + external adoption target set in advance.

## Domain
MLSecOps / ML data supply-chain security (DataOps × AppSec).

## Timeline
12 weeks solo with Claude Code. See `docs/ROADMAP.md`.
