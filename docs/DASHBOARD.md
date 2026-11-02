# Dashboard spec (Next.js + React Flow + Recharts)

Global: left nav (Pipelines, Findings, Checks, Reports, Settings); environment switch (preprod / prod); pipeline picker; last-scan timestamp.

## 1. Topology (home)
- React Flow graph: nodes (source, transform, feature_store, split, train_sink, serving_sink), edges = data flow.
- Node color = criticality (green→red); size = blast radius; badge = open findings count; shield icon = installed checks.
- Dashed boundary boxes for trust zones; edges crossing a boundary highlighted.
- Click node → side panel: score breakdown, findings, checks, upstream/downstream.
- Filters: finding category, severity, show only critical path.

## 2. Critical sites
Ranked table of top-N nodes: score, reasons (exposure, sensitivity, blast radius), open findings, checks coverage %. Row → node panel.

## 3. Data flow / lineage
Select a dataset or model → highlighted upstream path with per-edge indicators (drift, poisoning suspicion, PII). "Which sources fed this training run?" answer card.

## 4. Checks
Registry: check, node, guards (finding types), status (proposed/approved/installed/passing/failing), cost, history sparkline. Actions: approve, disable.

## 5. Findings and remediation
Kanban/table: detected → triaged → fix PR open → fixed / escalated → resolved. Card: severity, type, node, confidence, evidence, diff (Tier 1) or evidence pack (Tier 2), links to PR/issue. Actions: mark false positive, suppress with reason.

## 6. Reports
Generate/export per-pipeline report (PDF/Markdown): summary, top risks, findings by type, fixes applied, open escalations, drift/poison trends. Scheduled weekly.

## 7. Monitoring
Preprod: scan history, trend lines (findings over time, drift scores, check pass rate), CI-triggered runs. Prod: read-only status tiles + alert feed.

## Wireframe sketch (topology)
```
┌────────────┬──────────────────────────────────────────────┬──────────────┐
│ Nav        │  [source A]──►[clean]──►[split]──►[train]    │ Node panel   │
│            │  [source B]─┘ (boundary)      └──►[eval]     │ score 8.7    │
│            │  filters: ▢ data ▢ pipeline  sev ≥ high      │ findings (3) │
└────────────┴──────────────────────────────────────────────┴──────────────┘
```

## Non-functional
Loads topology for 200 nodes <2s; accessible color palette (not red/green only; use icons); dark mode.
