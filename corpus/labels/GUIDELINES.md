# Notebook labeling guidelines

Per-notebook labels in `notebooks/<competition>.yaml` (schema: `faultline.eval.labels.NotebookLabel`).
Each notebook is one training point; its competition's entry in `competitions.yaml` is the answer key
for what its data requires. Labels are written from the code alone, without looking at detector
findings, so rules and models can both be measured against them.

## Per notebook
- `relevant: false` when no model is trained (EDA only, a Python tutorial, a pure visualization).
  Nothing else is labeled for it.
- `validation`: how the notebook estimates its own score, at `validation_line`:
  `none` (fits on all training data and submits) | `random` | `stratified` | `group` | `temporal` |
  `slice` (positional `df[:n]` without a shuffle, on data whose order means nothing).
  If several are used, label the one the notebook relies on for its final choice.
- `issues`: every problem below, each with a finding type, the line where it happens, and a short
  note. An empty list means checked and clean.
- `unsure: true` when the code is too unclear to decide (hidden in helper modules, cut off).

## What counts (finding type: rule)
Lines are where the problem happens: the fit or statistic that leaks, the split that is wrong.

**Validation leakage** (the data used to score was seen during fitting):
- `data.leakage.preprocessing_before_split`: a fitted transform (scaler, imputer, PCA, binning) or a
  pandas statistic (`mean/median/std/quantile/min/max`, `fillna(df.mean())`) computed on rows later
  split into train/validation. Covers A1.
  Also counted: count/frequency encodings and group aggregates of feature columns
  (`groupby(col)[x].transform('mean')`, `value_counts()` maps) whose groups span rows that are later
  split, even when test rows are pooled in too: the validation rows' own values are in the fit.
  Not leakage: label/ordinal/one-hot encoding vocabularies; stateless transforms (`log1p`, ratios);
  aggregates over one entity's own rows (per card, per household) when validation splits by that
  entity or the entity is the row; statistics computed on the training part only.
- `data.leakage.target_derived`: target encoding, `groupby(...)[target].mean()`, or any feature built
  from the target on rows later used for validation.
- `data.leakage.feature_selection_full`: feature selection (univariate tests, model importances,
  correlation with the target) on all rows before the split or outside the CV loop.
- `data.leakage.resampling_before_split`: SMOTE / oversampling before the split (A3).
- `data.leakage.tuning_on_test`: the set whose score is reported is also used to choose: early
  stopping `eval_set`/`validation_data`, best model or threshold by its score (A2). Using a
  validation set for early stopping is fine when the reported score comes from other data
  (the leaderboard, or CV out-of-fold scores).
- `data.leakage.duplicate_across_split`: the notebook itself duplicates or augments rows before
  splitting (copies end up on both sides).

**Data–pipeline fit** (judged against the competition's label):
- `data.leakage.temporal_split`: the competition needs a temporal split and validation is `random`,
  `stratified` or a shuffled K-fold. Line: the split.
- `data.leakage.group_split`: the competition needs a group split and validation mixes the group's
  rows, unless the notebook first aggregates to one row per group.
- `data.fit.unstratified_split`: the competition says `stratify: true` and a classification holdout
  or K-fold isn't stratified (not raised when validation is `none`).
- `data.fit.single_split_small`: the competition says `small: true` and the notebook relies on one
  holdout instead of cross-validation.
- `data.fit.metric_mismatch`: the notebook optimizes or reports only a metric that disagrees with
  the official one in a way that changes decisions (accuracy on a heavily imbalanced AUC task).
- `data.fit.id_as_feature`: an identifier column (row id, passenger id) is used as a feature.
- `data.fit.schema_mismatch`: a feature the test set doesn't have (west-nile's `NumMosquitos`).

**Not counted:** adding the competition's unlabeled test features to a fit (transductive, allowed by
most competitions; noted only). The issue is the validation rows in the fit, not the test rows: a
fit on train+test before CV is still `preprocessing_before_split`. No validation at all (`validation: none` is a fact, not
an issue).

**Pipeline security and AI-era faults** (data-independent):
- `pipeline.dependency.suspicious_package`: installing or importing a package under a wrong or
  non-existent name (`pip install sklearn`), A4.
- `pipeline.deserialization.unsafe`: pickle / joblib / `torch.load` on files the notebook did not
  produce itself (downloaded, or from an input dataset), A5.
- `pipeline.supply_chain.trust_remote_code`, `pipeline.transport.tls_verification_disabled`,
  `pipeline.supply_chain.unverified_download`, `pipeline.secret.hardcoded` (Kaggle API keys,
  tokens), `pipeline.injection.sql`, `pipeline.injection.shell`: as named.

## Process
1. `faultline corpus extract --split train` writes one review file per competition (git-ignored).
2. Train is labeled first, then dev; test is extracted with `--final` and labeled last, once, after
   rules and models are frozen.
3. The labeler is recorded in each file. Disagreements found later are fixed in the label file with
   the reason in `note`, never by editing detectors to match.
