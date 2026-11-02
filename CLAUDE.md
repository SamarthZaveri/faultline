# FaultLine

FaultLine finds vulnerabilities in ML data pipelines (ingestion, transformation, pre-training use), hardens the most critical points with checks, and uses an agent team to fix simple problems (as PRs) and escalate complex ones.

Loop: **Discover → Detect → Harden → Remediate → Observe.**

## Read first (load only when relevant)
- Scope and success criteria: `docs/PRD.md`
- Components and data flow: `docs/ARCHITECTURE.md`
- The core contract: `docs/FINDING_SCHEMA.md` and `src/faultline/schema/finding.py`
- Current work: `docs/ROADMAP.md`
- Detector specs: `docs/detectors/*.md`
- Agents and guardrails: `docs/AGENTS.md`
- How we measure: `docs/EVAL_PLAN.md`, ground truth in `fixtures/dvp_faults.yaml`
- Past decisions (do not re-litigate): `docs/DECISIONS.md`

## Repo layout
```
src/faultline/
  schema/      Finding + Check pydantic models (source of truth)
  discovery/   parsers, OpenLineage ingestion, graph builder
  detectors/   pipeline_vulns/, leakage/, drift/, poisoning/, pii/
  scoring/     criticality scorer, check-placement optimizer
  checks/      check generators (Great Expectations / Pandera)
  agents/      triage, fixer, verifier, escalation, reporter
  api/         FastAPI app
  store/       Postgres models + migrations
dashboard/     Next.js + React Flow frontend
fixtures/      Damn Vulnerable Pipeline (DVP) + dvp_faults.yaml
tests/         mirrors src/
docs/          specs
```

## Commands
```
make setup        # create .venv (Python 3.11) and pip install -e .[dev]
make test         # pytest -q
make lint         # ruff check + ruff format --check + mypy
make format       # ruff --fix + ruff format
make schema       # regenerate schemas/*.json from the pydantic models
make up           # docker compose up (postgres, redis); api/dashboard added later
make scan PIPELINE=fixtures/dvp   # run all detectors on a pipeline
make bench        # run the evaluation benchmark (see docs/EVAL_PLAN.md)
```

## Rules (non-negotiable)
1. Every detector returns `Finding` objects from `faultline.schema`. Never invent ad-hoc result shapes.
2. Agents only open PRs or issues. They never write to production systems or merge.
3. All agent fixes are verified in a Docker sandbox (re-run detector + pipeline tests) before a PR is opened.
4. Production collectors are read-only and send summary statistics only, never raw rows.
5. Train/eval splits are by **repository/pipeline**, never by sample. Never tune on the held-out set.
6. Rules and statistics first; add ML/LLM only where docs/detectors/*.md says so. Do not swap libraries listed in docs/ARCHITECTURE.md without logging a decision in docs/DECISIONS.md.
7. Test first: add the failing test (usually from a DVP fault) before implementing.
8. Python 3.11+, full type hints, pydantic v2, no bare `except`, no secrets in code.

## Workflow
- One task at a time from `docs/ROADMAP.md`. Read its acceptance criteria first.
- Write test → implement → run `make test lint` → update ROADMAP status and DECISIONS if you made a choice.
- Prefer small PRs. Ask before adding a new dependency.
