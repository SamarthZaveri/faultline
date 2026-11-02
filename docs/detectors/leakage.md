# Detector: data leakage

"Data shown to a model that it should not have seen." Covers train/test contamination, target leakage, temporal leakage, and sensitive-data exposure.

**Emits:** `data.leakage.*`, `data.pii.*`.

## A. Static dataflow analysis on code (novel core)
Parse Python (AST) into a dataflow graph. Track which dataframes/arrays derive from which source and whether they cross the split boundary.
Patterns:
- fit/fit_transform (scaler, imputer, encoder, PCA, target encoding) on data before `train_test_split` → `preprocessing_before_split`
- feature computed from the label column → `target_derived`
- random shuffle/split on time-indexed data → `temporal_split`
- feature selection or hyperparameter search using test/validation data → `feature_selection_full`, `tuning_on_test`
- groupby/rolling aggregation computed across the full frame before split → `aggregation_across_split`
Optional v2 boost: fine-tuned code transformer (CodeBERT/UniXcoder class) as a second opinion on ambiguous cases. Rules stay as the high-precision first pass.

## B. Data-level checks
- Exact duplicates across splits: row hashing.
- Near duplicates (text/embeddings): MinHash/LSH, cosine threshold.
- Single-feature predictive power: per-feature AUC/R² near perfect → suspect target leak.
- Mutual information with target; flag outliers.
- Temporal order violation: max(train_time) > min(test_time).
- Benchmark contamination: n-gram overlap between training corpus and known eval sets.

## C. Sensitive-data exposure
Presidio recognizers + optional fine-tuned NER (ai4privacy + synthetic); regex+entropy for secrets in data. Report columns and rates, never raw values (redact).

## Outputs
Finding with evidence: the offending code lines (A) or metrics and redacted samples (B/C).

## Tier hints
Tier 1: move fit inside train split, switch to time-based split, dedupe across splits. Tier 2: redesign a leaky feature, PII policy decisions, benchmark contamination.

## Targets
F1 ≥0.85 on held-out repos (injected + real leakage). Datasets: Kaggle/KGTorrent notebooks, self-injected pairs, DVP LK-*.
