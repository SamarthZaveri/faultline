# Decisions (append-only, newest first)

Format: `date | decision | why | alternatives rejected`

- 2026-11-03 | Bench counts identical files once across a split | ML-For-Beginners ships ~50 translated copies of each notebook; 92% of train files were duplicates and inflated every count | per-file counting
- 2026-11-03 | Deserialization severity depends on the source: network → critical, fixed local path → one level lower, variable → base | 96 real hits were tutorials loading their own model from a literal path; same rule, honest severity | flat severity per call
- 2026-11-03 | Leakage treats sklearn composites (Pipeline, ColumnTransformer, FeatureUnion) as fit-before-split sources; LabelEncoder on the target is not leakage | found a real ColumnTransformer leak in the train corpus; label vocabulary leaks nothing about features | ignoring composites
- 2026-11-03 | Ground truth = injected faults in real repos + public labeled benchmarks + hand-triaged real findings, split by repository (train/dev/test), test run once with `--final` | user requirement to train and test on real data; injection alone is optimistic | synthetic-only fixture pipeline
- 2026-11-03 | First pipeline types: plain Python + notebooks; then Airflow, then dbt/SQL | most ML training code and the leakage benchmark live there | Airflow first
- 2026-11-03 | Storage SQLite first (SQLAlchemy later, Postgres via config); dashboard Python + HTML (FastAPI + Jinja2), not React | Docker usually off on the dev machine; user prefers Python + HTML | Postgres from day one; Next.js
- 2026-11-03 | Agent LLM deferred to M5 | detectors and checks first | choosing now
- 2026-11-03 | Detectors are stdlib-`ast` rules, not Semgrep/Bandit | runs natively on Windows, no extra deps, rules tailored to ML code (notebooks, sklearn) | Semgrep (weak Windows support), Bandit (no notebook/ML rules)
- 2026-11-03 | Rebuilt from scratch at user request; the previous version is commit 000ea99 on origin/main | user decision | keeping the old docs
