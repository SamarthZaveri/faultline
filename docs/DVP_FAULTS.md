# Damn Vulnerable Pipeline: planted faults

Generated from `fixtures/dvp_faults.yaml` (the source of truth). 40 faults. Build the DVP under `fixtures/dvp/` so each fault exists at its listed location, then write one test per fault asserting the expected finding type.

| ID | Category | Expected type | Severity | Tier | Location | Note |
|---|---|---|---|---|---|---|
| PV-01 | pipeline | `pipeline.secret.hardcoded` | high | tier1 | `dags/ingest_orders.py` | AWS access key literal in DAG |
| PV-02 | pipeline | `pipeline.secret.hardcoded` | high | tier1 | `config/settings.yaml` | DB password in config |
| PV-03 | pipeline | `pipeline.injection.sql` | high | tier1 | `transforms/clean_orders.py` | SQL built with f-string from row values |
| PV-04 | pipeline | `pipeline.deserialization.unsafe` | critical | tier1 | `train/load_artifacts.py` | pickle.load on downloaded artifact |
| PV-05 | pipeline | `pipeline.deserialization.unsafe` | high | tier1 | `transforms/load_rules.py` | yaml.load without SafeLoader |
| PV-06 | pipeline | `pipeline.dependency.unpinned` | medium | tier1 | `requirements.txt` | unpinned deps |
| PV-07 | pipeline | `pipeline.dependency.vulnerable` | high | tier1 | `requirements.txt` | old version with known CVE |
| PV-08 | pipeline | `pipeline.supply_chain.unverified_dataset` | high | tier1 | `dags/ingest_external.py` | dataset downloaded with no checksum |
| PV-09 | pipeline | `pipeline.injection.shell` | high | tier1 | `transforms/convert_files.py` | subprocess shell=True with data-derived filename |
| PV-10 | pipeline | `pipeline.iam.public_bucket` | critical | tier2 | `infra/storage.tf` | training-data bucket publicly writable |
| PV-11 | pipeline | `pipeline.boundary.unvalidated` | high | tier1 | `dags/ingest_external.py` | external API data written to store with no validation |
| PV-12 | pipeline | `pipeline.lineage.missing` | medium | tier2 | `train/train.py` | training data not versioned, no lineage |
| PV-13 | pipeline | `pipeline.supply_chain.unverified_model` | high | tier1 | `train/pretrained.py` | model downloaded over http, no hash |
| PV-14 | pipeline | `pipeline.iam.overbroad` | high | tier2 | `infra/iam.tf` | wildcard s3:* on all resources |
| LK-01 | data | `data.leakage.preprocessing_before_split` | high | tier1 | `train/features.py` | StandardScaler.fit on full data before split |
| LK-02 | data | `data.leakage.target_derived` | critical | tier2 | `train/features.py` | feature computed from label |
| LK-03 | data | `data.leakage.duplicate_across_split` | high | tier1 | `data/orders.parquet` | 5% rows duplicated across train/test |
| LK-04 | data | `data.leakage.temporal_split` | high | tier1 | `train/split.py` | random split on time series |
| LK-05 | data | `data.leakage.feature_selection_full` | medium | tier1 | `train/select.py` | SelectKBest on full data |
| LK-06 | data | `data.leakage.aggregation_across_split` | high | tier2 | `train/features.py` | customer aggregate computed over all rows incl. test |
| LK-07 | data | `data.leakage.near_duplicate` | medium | tier2 | `data/reviews.parquet` | near-duplicate reviews across split |
| LK-08 | data | `data.pii.exposure` | high | tier2 | `data/customers.parquet` | SSN and email columns in training features |
| LK-09 | data | `data.leakage.tuning_on_test` | medium | tier1 | `train/tune.py` | GridSearchCV scored on test set |
| LK-10 | data | `data.leakage.benchmark_contamination` | high | tier2 | `data/corpus.jsonl` | eval benchmark items present in training corpus |
| PO-01 | data | `data.poisoning.label_flip` | high | tier2 | `data/images/train` | 5% label flips |
| PO-02 | data | `data.poisoning.label_flip` | high | tier2 | `data/images/train` | 15% label flips |
| PO-03 | data | `data.poisoning.backdoor` | critical | tier2 | `data/images/train` | BadNets 3x3 patch trigger on 3% of images |
| PO-04 | data | `data.poisoning.clean_label` | high | tier2 | `data/images/train` | clean-label poison on target class |
| PO-05 | data | `data.poisoning.injected_rows` | high | tier2 | `data/orders.parquet` | injected out-of-distribution rows (2%) |
| PO-06 | data | `data.poisoning.injected_rows` | high | tier2 | `data/ingest/batch_2026_03.parquet` | untrusted source batch with shifted stats |
| PO-07 | data | `data.poisoning.source_flooding` | medium | tier2 | `data/ingest/source_x` | one source floods near-duplicates |
| PO-08 | data | `data.poisoning.text_trigger` | high | tier2 | `data/reviews.parquet` | rare trigger phrase correlated with positive label |
| DR-01 | data | `data.drift.covariate` | high | tier2 | `data/serving/week_12` | mean shift in 3 features |
| DR-02 | data | `data.drift.covariate` | medium | tier2 | `data/serving/weeks_1_12` | gradual drift |
| DR-03 | data | `data.drift.schema_semantics` | high | tier2 | `data/serving/week_8` | unit change (USD to cents) in price |
| DR-04 | data | `data.drift.label_prior` | medium | tier2 | `data/serving/week_10` | label prior shift 10% to 30% |
| DR-05 | data | `data.drift.concept` | high | tier2 | `data/serving/week_11` | feature-label relationship changes |
| DR-06 | data | `data.drift.categorical_new` | medium | tier1 | `data/serving/week_9` | unseen category values |
| DR-07 | data | `data.quality.missing_spike` | medium | tier1 | `data/serving/week_7` | missing rate jumps 1% to 25% |
| DR-08 | data | `data.drift.embedding` | medium | tier2 | `data/serving/text_week_12` | text embedding distribution shift |

## Counts

PV: 14, LK: 10, PO: 8, DR: 8

Tier mix: tier1: 18, tier2: 22
