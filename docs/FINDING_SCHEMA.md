# Finding schema (the core contract)

Source of truth: `src/faultline/schema/finding.py`. JSON Schema: `schemas/finding.schema.json` (generated; do not hand-edit). Regenerate with `python -m faultline.schema.export`.

## Finding
| Field | Type | Notes |
|---|---|---|
| id | str (16 hex chars) | sha256 prefix of (type, node_id, file, line_start, dataset, sorted columns, evidence.rule_id) so re-scans dedupe |
| type | str | Dotted taxonomy, see below |
| title | str | One line |
| severity | enum | critical, high, medium, low, info |
| confidence | float 0-1 | Detector's calibrated confidence |
| environment | enum | preprod, prod |
| category | enum | data, pipeline |
| location | Location | node_id, file, line range, dataset, columns |
| evidence | Evidence | metrics, samples (redacted), rule id, explanation |
| suggested_fix | SuggestedFix? | description, optional unified diff, check_spec |
| complexity | enum | tier1, tier2, unknown (set by Triage) |
| status | enum | detected, triaged, fix_pr_open, fixed, escalated, resolved, false_positive, suppressed |
| detector | str | name@version |
| created_at / updated_at | datetime | UTC |
| refs | list[str] | CWE, OWASP-ML, paper, docs |

## Type taxonomy (dotted)
```
data.poisoning.label_flip | backdoor | clean_label | injected_rows | source_flooding | text_trigger
data.drift.covariate | label_prior | concept | schema_semantics | categorical_new | embedding
data.leakage.preprocessing_before_split | target_derived | duplicate_across_split | temporal_split |
             feature_selection_full | aggregation_across_split | near_duplicate | tuning_on_test | benchmark_contamination
data.pii.exposure | secret_in_data
data.quality.missing_spike | schema_break | out_of_range
pipeline.secret.hardcoded
pipeline.injection.sql | shell
pipeline.deserialization.unsafe
pipeline.dependency.unpinned | vulnerable
pipeline.supply_chain.unverified_dataset | unverified_model
pipeline.iam.overbroad | public_bucket
pipeline.boundary.unvalidated
pipeline.lineage.missing
```

## Check (installed by Harden stage)
id, node_id, kind (expectation|pandera|custom), spec, guards (list of finding types), cost_estimate, status (proposed|approved|installed|failing|passing|disabled), history.

## Rules
- Never put raw PII or secret values in `evidence.samples`; redact (hash/mask).
- `confidence` must be calibrated against the DVP benchmark before shipping a detector.
- Changing the schema requires a migration and a DECISIONS entry.
