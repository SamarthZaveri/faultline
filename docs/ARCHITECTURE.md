# Architecture

## Flow
```
Connectors (Airflow / dbt / Dagster / Prefect / plain Python+SQL / notebooks)
   │
   ▼
[1] Discovery & Graph Builder   (OpenLineage events + static parse)
   │
   ▼
[2] Detection Engine
     ├─ Code/config analyzers  → pipeline_vulns, leakage-in-code
     └─ Data analyzers         → poisoning, drift, leakage-in-data, pii
   │            (all emit Finding)
   ▼
[3] Criticality Scorer          (graph centrality × exposure × sensitivity)
   │
   ▼
[4] Check Injector              (generates GE/Pandera checks at top-risk nodes)
   │
   ▼
[5] Agent Team                  (triage → fix PR or escalate → verify)
   │
   ▼
[6] Store (Postgres) + API (FastAPI) + Dashboard (Next.js)
```

## Design principles
- **Finding is the contract.** Detectors emit `Finding`; scorer, agents, API, dashboard only consume `Finding`. Adding a detector never requires touching downstream code.
- **Pipeline = graph.** Nodes: source, transform, feature_store, split, train_sink, serving_sink. Edges: data flow, with a trust-boundary flag.
- **Two modes.** `preprod` (full scans, agents may open PRs) and `prod` (read-only, summary stats only).
- **Rules first, ML second.** Rules give precision/explainability; ML/LLM adds recall and judgment (see detector docs).

## Components
| Module | Responsibility | Key libs |
|---|---|---|
| discovery | Parse code/config, ingest OpenLineage, build graph | tree-sitter, `ast`, OpenLineage, NetworkX |
| detectors/pipeline_vulns | Static analysis of code/config | Semgrep, Bandit, gitleaks, OSV |
| detectors/leakage | AST dataflow + data-level checks | `ast`, datasketch (MinHash), scikit-learn |
| detectors/drift | Statistical + classifier drift | SciPy, Evidently, River, alibi-detect |
| detectors/poisoning | Label errors, backdoors, anomalies | cleanlab, ART, PyTorch, scikit-learn |
| detectors/pii | PII and secrets | Presidio, transformers |
| scoring | Criticality + placement optimizer | NetworkX, OR-Tools/PuLP |
| checks | Generate and install checks | Great Expectations, Pandera |
| agents | Triage/Fixer/Verifier/Escalation/Reporter | Claude Agent SDK, Docker |
| api/store | Persistence and REST | FastAPI, Postgres, SQLAlchemy, Alembic |
| dashboard | UI | Next.js, React Flow, Recharts |
| infra | Jobs, metrics, experiments | Redis, Celery/Dramatiq, OpenTelemetry, MLflow, DVC |

## Environments
- `preprod`: scans in CI and staging, scheduled continuous scans, sandboxed fix verification, PR creation via GitHub App.
- `prod`: passive collectors as lightweight pipeline tasks emitting aggregates (histograms, null rates, hashes, counts), never raw rows. Alerts only.

## Security of FaultLine itself
- Secrets via env/secret manager. Least-privilege tokens (GitHub App: branch writes and PRs, no merge).
- Sandbox: allowlisted network egress, resource limits, throwaway containers.
- Every agent action logged with inputs, outputs and evidence.
