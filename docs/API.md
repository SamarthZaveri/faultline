# API (FastAPI, JSON, `/api/v1`)

Auth: bearer token (v1), per-project scope.

## Pipelines
- `POST /pipelines` connect a pipeline (repo URL/path, orchestrator, env)
- `GET /pipelines` / `GET /pipelines/{id}`
- `GET /pipelines/{id}/graph` → `{nodes:[...], edges:[...]}` with criticality, trust-boundary flags
- `GET /pipelines/{id}/lineage?target={dataset_or_model}`

## Scans
- `POST /pipelines/{id}/scans` body: `{mode: preprod|prod, detectors?: [...]}` → `202 {scan_id}`
- `GET /scans/{scan_id}` status, progress, summary
- `POST /pipelines/{id}/schedules` continuous monitoring (cron) for preprod

## Findings
- `GET /findings?pipeline=&severity=&type=&status=&node=` (paginated)
- `GET /findings/{id}`
- `PATCH /findings/{id}` status (false_positive, suppressed) with reason
- `POST /findings/{id}/remediate` trigger agent team (preprod only)

## Checks
- `GET /checks`, `GET /checks/{id}`
- `POST /pipelines/{id}/checks/propose` run placement optimizer
- `POST /checks/{id}/approve` / `POST /checks/{id}/disable`

## Reports
- `POST /pipelines/{id}/reports` → `202 {report_id}`
- `GET /reports/{id}?format=pdf|md`

## Prod collectors
- `POST /ingest/summary` accepts aggregate summaries only (histograms, counts, hashes). Reject payloads that look like raw rows (size/shape guard).

## Conventions
Errors: `{error:{code,message,details}}`. Pagination: `?limit=&cursor=`. Timestamps UTC ISO-8601. Request/response models live in `src/faultline/api/models.py` and reuse `faultline.schema`. Generate OpenAPI at `/openapi.json`; dashboard types generated from it.
