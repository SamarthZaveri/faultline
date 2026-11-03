# FaultLine

Finds vulnerabilities in ML data pipelines (data leakage, poisoning, drift, and pipeline
vulnerabilities like hardcoded secrets, SQL/shell injection and unsafe deserialization), then
hardens and fixes them.

```
make setup
make scan PIPELINE=path/to/your/pipeline
```

Status and plan: `docs/PLAN.md`.
