# Rule-based baseline results (2026-11-03)

The first measured results for FaultLine. Use this page as the single source for README and
resume numbers, and as the baseline that later ML components must beat.

- **What was measured:** the rule-based detectors only (`leakage@0.2`, `pipeline_vulns@0.1`).
  No ML model is trained or used anywhere.
- **Source files:** `reports/bench_train_20261103-174220.json`,
  `reports/bench_dev_20261103-174228.json`, `reports/yang2022_dev_20261103-174229.json`.
  These are local (reports/ is not committed); every number below is copied from them.
- **Code state:** the run came after the last detector change of the day. Later edits that day
  (removing a duplicate test, adding a `baseline freeze` command) do not affect detection.

## What was covered
| Set | Size | Role |
|---|---|---|
| Corpus **train** split | 23 repos, 2,137 unique files, 185,027 lines (2,835 duplicate files skipped) | Rules were written against this |
| Corpus **dev** split | 5 repos, 1,180 unique files, 397,131 lines (96 duplicates skipped) | Measured, not used to write rules (with one exception, see caveats) |
| Yang et al. ASE'22 leakage benchmark, **dev half** | 29 human-labeled notebooks | Measured |

**Not covered:**
- **Held out on purpose, run once at the end:** the corpus **test** split (5 repos) and the Yang
  **test** half.
- **Not fetched:** `goto_conversion` (train) failed to check out on Windows.
- **Not downloaded yet:** the Kaggle notebooks.

## 1. Human-labeled real notebooks: the headline result
Labels are per notebook (does it contain this kind of leakage or not). "Published tool" is the
analyzer from the same paper, using the outputs its authors released for these notebooks.

| Leakage kind | FaultLine P | FaultLine R | FaultLine F1 | Published tool P | R | F1 |
|---|---|---|---|---|---|---|
| Preprocessing before split (`pre`) | 85.7% (6/7) | 54.5% (6/11) | 66.7% | 100% | 72.7% | 84.2% |
| Test rows used in training (`overlap`) | 100% (1/1) | 25.0% (1/4) | 40.0% | 100% | 100% | 100% |
| Test set reused for model selection (`multi`) | 87.5% (7/8) | 43.8% (7/16) | 58.3% | 100% | 62.5% | 76.9% |

Before the AI-fault detector pass the same day, FaultLine scored: pre R 36.4% (P 80%), overlap R 0%,
and multi had no detector (R 0%).

**Why this is the most credible number:** real code, labels made by other people, and a published
tool to compare against.

**Why it's still not final:**
- 29 notebooks is small.
- Some rules (splits done by slicing, a model fit on all data) were added after looking at why
  dev notebooks were missed, so dev is slightly optimistic.
- The test half is the clean measure.
- The published tool has a home advantage: its authors also made the labels.

## 2. Faults planted in real repos (injected recall)
Each mutator inserts one known fault into a real file at a known line, in memory (the repo copies
on disk are never modified). Recall = planted faults found at the planted lines. Up to 3 files per
mutator per repo, seed 0.

| Fault type | Train recall | Dev recall |
|---|---|---|
| Hardcoded secret | 100% | 100% |
| SQL injection | 100% | 100% |
| Shell injection | 100% | 100% |
| Unsafe deserialization | 100% | 100% |
| Preprocessing before split | 100% | 80% |
| Test data used in training | 100% | 100% |
| Feature selection on all data | 87.5% | 25% |
| **Overall** | **99.6%** (227 planted) | **93.1%** (58 planted) |

**Why it's optimistic:** the same person wrote the fault templates and the detection rules, so they
naturally agree. It shows the plumbing works and catches regressions; it is not evidence of
real-world recall. Never quote it without table 1 next to it.

**What it does show:** the dev drop (feature selection 25%, preprocessing 80%) comes from real
code shapes the templates land in. For example, the split's input is an expression like
`housing.data`, or there is a second split of a training subset.

## 3. Findings on untouched real code
What the rules flag on the original, unmodified repos.

| Finding type | Train | Dev |
|---|---|---|
| Hardcoded secrets | 57 | 12 |
| Unsafe deserialization | 34 | 22 |
| Unpinned dependencies | 19 | 5 |
| Preprocessing before split | 16 | 19 |
| Test set used for model selection | 5 | 5 |
| Shell injection | 4 | 21 |
| SQL injection | 0 | 12 |
| Test data used in training | 1 | 2 |
| **Total** | **136** (0.74 per 1K lines) | **98** (0.25 per 1K lines) |

**Why there is no precision yet:** only 1 finding has been hand-checked (a real ColumnTransformer
leak, true positive). Until findings are labeled in `corpus/triage/<repo>.yaml`, this table says
what the tool flags, not how often it is right.

**Known false-positive sources from spot checks:**
- local-dev passwords in docker-compose files (some marked `# CHANGEME`);
- Kubernetes and test fixtures;
- SQL f-strings built from internal config values.

## How to quote these (README / resume)
- **Lead with table 1**, with its size: 29 labeled notebooks, dev half, vs the published tool.
- **Say "rule-based".** Don't imply ML.
- **If you use injected recall**, call it "recall on planted faults in real repos".
- **Present the dedupe as a finding of its own:** 92% of files in one tutorial corpus were
  translated copies. Removing them before measuring is a methodology point worth stating.
- **Don't claim precision on real code** until triage exists.
