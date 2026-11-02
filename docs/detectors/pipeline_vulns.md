# Detector: pipeline vulnerabilities

**Method:** rules/static analysis first, LLM triage second.
**Emits:** `pipeline.*` findings. **Mode:** preprod (full), prod (config/permission checks only).

## Inputs
Repo path (code, DAGs, notebooks, SQL, requirements, IaC, configs).

## Checks
| Check | Tool | Finding type |
|---|---|---|
| Hardcoded secrets | gitleaks, detect-secrets, regex+entropy | pipeline.secret.hardcoded |
| SQL built from strings | Semgrep custom rules | pipeline.injection.sql |
| Shell with data-derived args | Semgrep/Bandit | pipeline.injection.shell |
| pickle/joblib/yaml.load on untrusted input | Semgrep/Bandit | pipeline.deserialization.unsafe |
| Unpinned deps | requirements parser | pipeline.dependency.unpinned |
| Vulnerable deps | OSV API / local DB | pipeline.dependency.vulnerable |
| Dataset/model downloads without checksum, or over HTTP | Semgrep + URL analysis | pipeline.supply_chain.* |
| Public or wildcard IAM/bucket policy | Terraform/YAML lint | pipeline.iam.* |
| Missing validation at trust boundary | graph rule: external source → store with no check node | pipeline.boundary.unvalidated |
| No lineage/versioning | graph rule | pipeline.lineage.missing |

## LLM triage layer
For each raw finding, the LLM sees the rule id, code snippet ±30 lines, and file context, and returns `{is_true_positive, confidence, reason, proposed_patch?}`. Rules give recall; LLM gives precision. Log every verdict; never let the LLM suppress a critical finding without a second rule-based signal.

## Tier hints
Tier 1: secrets → env vars, parameterize SQL, safe loaders, pin deps, add checksum. Tier 2: IAM redesign, lineage adoption.

## Metrics / targets
Precision ≥80% after triage, recall ≥85% on planted faults (PV-*), held-out repos for generalization. Datasets: DVP, CVEfixes, DiverseVul, public Airflow/dbt repos.
