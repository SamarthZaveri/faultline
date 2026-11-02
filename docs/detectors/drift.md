# Detector: data drift

**Emits:** `data.drift.*`, `data.quality.*`. **Mode:** preprod and prod (prod uses stored reference summaries, not raw rows).

## Methods
- Univariate: PSI, KS (numeric), chi-square (categorical), Wasserstein for magnitude.
- Multivariate: MMD and **classifier two-sample test** (train reference-vs-current classifier; AUC well above 0.5 = drift; feature importances localize drifting features).
- Streaming: ADWIN, DDM via River for sequential data.
- Unstructured: embedding drift (MMD/Wasserstein on embeddings).
- Label prior shift: compare label distributions.
- Concept drift: performance monitoring when labels arrive; otherwise report "covariate drift, impact unknown".
- Quality: missing-rate spikes, schema breaks, range violations.

## Design
- Reference window stored as compact summaries (histograms, quantile sketches, category counts, embedding centroids/samples) to support prod without raw rows.
- Multiple-testing correction (Benjamini-Hochberg) across features.
- Severity from effect size and from the criticality of the affected node and features, not from p-value alone.

## Tier hints
Tier 1: add range/schema/missing checks. Tier 2: retrain decisions, upstream contract fixes.

## Targets
Detection delay and false-alarm rate no worse than an Evidently/PSI baseline; correct feature localization ≥80% (DR-*). Datasets: River streams, Electricity, Airlines, WILDS/Wild-Time, NYC Taxi shifts.
