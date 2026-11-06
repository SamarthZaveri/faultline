# FaultLine: phases and scope

The single source for what we build, in what order, and what is out of scope.
`docs/PLAN.md` holds the goal, architecture and evaluation rules; `docs/DECISIONS.md` holds choices.

## Where we are (2026-11-03)
- **Built (Phase 1):** a Python library + CLI. It reads repos and notebooks, builds a pipeline
  graph, and runs two **rule-based** detectors (leakage, pipeline vulnerabilities). It also has a
  code fault injector and a benchmark on 7 pinned real repos (train/dev/test split by repo).
- **No ML, DL or RL yet.** Everything is static analysis (Python `ast`) and regex rules.
- **No backend or frontend yet.** No API, database or dashboard.

## Scope
**Final scope (2026-11-04)**: four pillars and a dashboard. The full write-up, with reasons,
diagrams and a file-by-file guide, is `doc.txt` at the repo root.

1. **Data pipeline vulnerability detection.**
   - Data problems: drift, poisoning, leakage, and data–pipeline fit.
   - Code problems: SQL/shell injection; hardcoded things (secrets, credentials, paths, magic
     thresholds); other vulnerabilities (unsafe deserialization, supply chain, TLS, dependencies,
     trust boundaries, config).
   - Every finding gives the file and lines responsible and a node in the pipeline graph.
   - Detection is learned where learning wins (Phase 4) and rule-based where rules win.
2. **A custom test suite at the most vulnerable points, with the reason for each test**
   (Phase 5).
   - Tests cover data contracts, leakage invariants, drift thresholds and behavioural/metamorphic
     properties.
   - Each test is proven by mutation: it must fail on the fault it targets and pass on clean
     code.
   - Tests run on a sample of the data.
3. **Dataset–pipeline fit score**: how good the pipeline is for the given dataset, with the
   reasoning (Phase 3b). It runs only when a dataset is provided.
4. **Fixing easy issues with multiple agents trained with RL, and a human in the loop (HITL) for
   big or hard issues, with reasons** (Phase 6). Agents open PRs or issues only.
5. **Dashboard** (Phase 7), in this order:
   1. the pipeline as an architecture map, with **red edges** at vulnerable points;
   2. the test suite: where tests should be added, with a button to add them;
   3. the dataset–pipeline fit score (only when a dataset is uploaded);
   4. the agents' fixes: the new pipeline with changes, the tests still at critical points, what
      needs human attention, what is done, and the final pipeline.

**Scope details**
- **Pipelines:** Python scripts and Jupyter notebooks (pandas / scikit-learn first; PyTorch
  later); then Airflow DAGs and dbt/SQL.
- **Environments:** pre-production (full scans, PRs) and production (passive, read-only,
  aggregate-only monitoring); continuous scans in CI/staging.

**Out of scope (v1)**
- Attacks on trained models (adversarial examples, model extraction), real-time blocking of
  production traffic, streaming-engine internals, every orchestrator.
- Agents merging PRs or writing to production systems.
- Executing untrusted notebooks outside a sandbox.

## Where ML / DL / NLP / RL are used (the core of the project)
FaultLine is an **MLSecOps** project built around learned components with our own trained
weights. **Best method wins, per component.** Each job uses whatever measures best on dev: a
model, a rule, or a rule + model combination. A model never replaces a rule that beats it. Every
result is reported as an ablation (rules vs model vs both), so the choice is evidence-based and
visible.

Rules and statistics also serve as:
1. the **baseline** every model is compared against;
2. the **feature extractor** (the facts the rules collect are model inputs);
3. a **label source**, together with injected faults, runtime traces and public benchmarks;
4. the **shipped component** wherever they win. Exact checks (secrets, exact duplicate rows,
   schema) will likely stay rules, possibly with a model ranking them.

**Hardware:**
- **Local GPU:** RTX 5050, 8 GB.
- **What it covers:** LightGBM (CPU), fine-tuning the ~125M code encoder, QLoRA fine-tuning of
  0.5–1.5B code LLMs, and small-group GRPO (0.5B; 1.5B with 4-bit weights, short contexts and
  group size around 4).
- **Heavier RL runs** go to Kaggle's free GPUs.
- **Blackwell card:** it needs a PyTorch build for CUDA 12.8 or newer. Check this before
  installing torch.

| Component | Technique | Status / phase |
|---|---|---|
| Pipeline-fault detector (leakage, fit mismatches) | **ML** v1: LightGBM on per-site facts + dataset profile. **DL/NLP** v2: fine-tuned code encoder (UniXcoder/CodeBERT) on the code around each site, exported to ONNX | Phase 4, starts with the Kaggle corpus |
| Finding ranker / false-positive filter | **ML** classifier on finding + context features, trained on triage labels | Phase 4 |
| Drift and joint drift | Statistics + **ML** classifier two-sample test (gradient boosting, permutation importance) | written (Phase 3) |
| Label flips | **ML** confident learning (out-of-fold gradient-boosting probabilities) | written (Phase 3) |
| Injected / out-of-distribution rows | **ML** cross-fitted Isolation Forest + Mahalanobis distance | written (Phase 3) |
| Backdoor triggers in image/text data | **DL** spectral signatures / activation clustering | Phase 5 |
| Fixer agent | **RL** with verifiable rewards (GRPO) on a small open code LLM with LoRA; rewards from a separate judge (see the RL track below) | Phase 6 |
| Triage policy (fix / escalate / skip) and check placement under a budget | **RL**: contextual bandit first, then policy gradient, on the same judge-based rewards | Phases 5–6 |
| Explanations and escalation reports | **NLP/LLM**, grounded in the finding's evidence | Phase 6 |
| MLOps around our own models | Experiment tracking, model registry + model cards, eval gates in CI, ONNX serving, drift monitoring of our own model's inputs | Phases 4–7 |

## Track: RL for the agents (verifiable rewards, separate judge)
**Environment.**
- An episode is a repo, plus a list of findings or sites. Some carry injected faults with known
  ground truth: a fix is needed, and what a correct one does. Others are clean sites where no fix
  is needed.
- Runs happen in the Docker sandbox on sampled data (Phase 3).

**Actions per site:** `fix` (emit a patch), `escalate`, `skip`.

**Judge (anti reward hacking).** A separate process the policy can't see into or write to:
- It applies the patch in its own sandbox.
- It runs the checks: the patched code parses; a re-scan shows the target finding gone and no
  new findings; the pipeline still runs on sampled data; the tracer shows no test rows reaching
  `fit`; held-out tests the policy never sees pass.
- It assigns the reward. The judge is programmatic, not an LLM grading another LLM, because LLM
  graders can be gamed.
- The policy's output is only the patch. Patches that touch tests, the judge or the detectors are
  rejected and penalized.

**Rewards: the user's design (v0)**

| Outcome | Reward |
|---|---|
| Each correct fix | +5 |
| All fixes in the episode correct | +50 (bonus) |
| Wrong fix | −6 |
| No fix, but a fix was needed | −45 |
| No fix, and none was needed | +15 |

**Incentive check:** on a site that needs a fix, attempting costs at most −6 while skipping costs
−45, so v0 teaches the policy to always attempt, even when it is very unsure.

**Rewards in use: v1** (chosen 2026-11-06). v0 plus:
- `escalate`: +8 when the fix needed judgment (no correct fix exists in the templates or the
  policy's own attempts), −4 when it was trivial;
- wrong fixes split: one the judge catches costs −6; one that passes the visible checks but fails
  the held-out tests or re-introduces leakage costs −40.

| Outcome | Reward |
|---|---|
| Each correct fix | +5 |
| All fixes in the episode correct | +50 (bonus) |
| Wrong fix, caught by the judge's visible checks | −6 |
| Wrong fix that passes visible checks but fails hidden tests / re-leaks | −40 |
| Escalate, and the fix really needed a human | +8 |
| Escalate a trivial fix | −4 |
| No fix, but a fix was needed | −45 |
| No fix, and none was needed | +15 |

With v1, an unsure policy escalates (+8) instead of gambling on a fix that may slip past the
checks (−40), and a confident one still fixes (+5, plus the bonus).

The same judge and reward setup is reused for check placement and the triage policy.

**Training:**
- **Policy:** a small open code LLM (Qwen2.5-Coder-0.5B first, then 1.5B; Apache-2.0) with LoRA,
  warm-started by supervised fine-tuning on (faulty code, verified fix) pairs from the injectors
  and the deterministic tier1 fixer, then GRPO.
- **Compute:** the local 8 GB RTX 5050 (4-bit weights, group size around 4); larger runs on
  Kaggle GPUs.
- **RL vs the deterministic fixer:** the RL agent is only used where it beats the deterministic
  fixer on held-out repos. For templated tier1 fixes the deterministic fixer may simply be
  better, and then it stays.
- **Reported results:** pass rate on held-out repos, reward curves, the escalate/skip mix, and
  the comparison with the deterministic fixer and with SFT only.

## Phases

### Phase 1: Foundations + static detectors (done)
Schema, discovery, graph, leakage and pipeline-vuln rules, code fault injector, real-repo
benchmark, CLI.
*Exit:* tests + lint green; bench runs on train/dev. Open issues found on dev are in PLAN.md.

### Phase 2: Real corpus at scale (Kaggle + GitHub) (in progress)
**Status 2026-11-03:**
- **Done:** GitHub discovery (34 repos, pinned); corpus-wide exact + near-duplicate removal (2,242
  files dropped, 2 across splits); group/author split invariants; Yang et al. labeled leakage set
  (dev half evaluated, test half held out).
- **Ready but blocked:** Kaggle discovery + fetch, waiting on the Kaggle token and approval of the
  `kaggle` dependency.
- **Not started:** CVEfixes.

Goal: thousands of real ML notebooks/pipelines to train and test on.
- **Kaggle:** use Kaggle's official sources, not browser scraping. That means the **Meta Kaggle
  Code** dataset (public notebooks plus metadata linking each to its competition and author) and
  the Kaggle API (`kernels list` / `kernels pull`, and dataset download for the data each
  notebook uses). Scraping kaggle.com with Playwright is slower and breaks when the site changes,
  and the terms of use may forbid it. Check the terms and each dataset's license before bulk
  downloads.
- **GitHub:** the GitHub API (`gh search repos/code`) for ML pipelines (sklearn/pandas notebooks,
  Airflow, dbt), pinned to commits like today's manifest.
- **Dedupe and split:** Kaggle notebooks are heavily forked. Dedupe by normalized code hash plus
  near-duplicate detection (MinHash). Split by **competition/dataset and author** (Kaggle) and by
  **repository** (GitHub), so near-copies never land on both sides.
- Add public labeled sets: Yang et al. leakage notebooks, CVEfixes.
- Target: about 500 Kaggle notebooks across about 50 competitions, plus about 30 GitHub repos.
  Grow once the tooling holds up.
*Exit:* corpus manifest with fixed splits; static bench on all of it; dedupe stats reported.

### Phase 3: Run the pipelines with data (dynamic analysis + data detectors)
**Status 2026-11-04:**
- **Written, not yet run:** the data detectors (`src/faultline/data/`) and data fault injectors
  (`inject/data.py`), with unit tests on scikit-learn's bundled breast-cancer data and on
  synthetic frames. The CLI commands are `faultline data compare` and `faultline data overlap`.
  - Drift: PSI + KS / chi-squared per column, a classifier two-sample test (gradient boosting +
    permutation importance), label prior, and concept drift (a reference model's error rate).
  - Label flips: confident learning, implemented in-house.
  - Injected rows: a cross-fitted isolation forest and Mahalanobis distance, plus duplicate floods.
  - Overlap: exact duplicates across the split.
- **Not started:** sampler, Docker runner, runtime tracer, and a data bench on real datasets.

Goal: send data through each pipeline as intended, detect data problems, and point to the
lines that cause them. We need the **data flow and the issues, not trained models**, so runs use
sampled data and cheap training.
- **Sampled datasets:** download each competition's data once, then write a small sample (default
  2,000 rows per table) that keeps what the pipeline's code depends on:
  - same columns, dtypes and file names;
  - every category value at least once where possible, so encoders and dummies keep their shape;
  - the target class balance (stratified);
  - time order for time-series data (a contiguous window, not random rows);
  - join keys consistent across tables (sample by key, so merges still match);
  - duplicate groups together (sample by row hash), so real duplicate leakage isn't sampled away.

  The notebook only ever sees the sample. Full-data statistics (row counts, null rates,
  histograms, duplicate rates) are profiled once outside the notebook, so drift and duplicate
  detectors can compare against the real distribution. Data faults are injected **after**
  sampling, so the ground truth stays exact.
- **Cheap training:** the tracer caps training cost; model quality is irrelevant:
  - boosting rounds and `n_estimators` capped (e.g. 10);
  - neural-net `epochs` set to 1;
  - grid/random search cut to its first candidate.

  Calls and data flow are unchanged, so every `fit`/`predict` is still traced on the real
  code path.
- **Sandboxed runner:** each notebook/script runs in a throwaway Docker container with CPU,
  memory and time limits. Network is allowed only in a separate dataset-download step. Notebooks
  that fail (missing deps, GPU, huge data) are recorded, not forced to run. Success rate is a
  reported metric.
- **Runtime tracer:** an import hook wraps pandas readers, `train_test_split`, and
  `fit`/`transform`/`predict`/`score`. It tags rows with content hashes and records the
  **file, line (cell)** and operation for each step. From the trace:
  - test rows reaching any `fit` → leakage at that line
  - rows shared across train/test → duplicate leakage
  - a feature computed from the label → target leakage
  - which dataset nodes feed which code nodes → the real data-flow graph
- **Data fault injection on the real datasets** before the run: label flips, injected rows,
  backdoor triggers, covariate/unit drift, missing spikes, duplicates across split. Ground truth
  says which rows were changed; the tracer says which lines touched them.
- **Data detectors** (statistics first, then ML; see the table above) run on the data at each
  traced step.
- **Limits of sampling (reported, not hidden):** issues that only show at full scale (e.g. a 0.1%
  duplicate rate, rare categories, memory-driven code paths) may not reproduce on the sample.
  Full-data profiles cover part of this.
*Exit:* data-fault recall reported per type on dev, with the responsible lines located; runner
success rate known.

### Phase 3b: Data–pipeline fit, fit score (pillar 3), and fixing the easy mismatches
**Fit score (0–100), with reasons:**
- **Weighted checks over five areas:** split validity, leakage, data handling, evaluation and
  robustness.
- **Coverage is reported next to the score:** the share of checks that could be judged. A score
  built on 3 of 20 checks must not look as certain as one built on 18.
- **Later calibration:** against the measured **optimism gap**. Run the pipeline on the sample
  with its own split, then with the split the data requires; the drop in score is the ground
  truth for how badly the pipeline fits.
- **Model:** a regression model predicts that gap statically, and it replaces the hand weights
  only if it predicts better on dev.
Every dataset needs its own split, preprocessing, metrics and drift checks. This phase checks
that the pipeline matches the data it runs on, and fixes the simple mismatches.

**1. Profile the data.** One dataset profile per table, computed once (full-data statistics from
Phase 3), never with raw values in findings. It records:
- time columns and whether the target lies in the future;
- repeated entities (group keys);
- target type and class balance;
- size;
- cardinality, skew and missing rate per column;
- near-perfect target proxies;
- duplicates.

**2. Derive requirements, then compare them with what the code does.** Static analysis covers
splits (with `stratify`/`groups`), fits, encoders and metrics, using leakage.py's dataflow; the
Phase 3 tracer then confirms which columns and rows reach `fit`. Each mismatch is a finding at
the responsible line.

| Data shows | Pipeline must | Mismatch | Fix (tier) |
|---|---|---|---|
| Rows ordered in time; target in the future | Temporal split / `TimeSeriesSplit` | Random split or `KFold(shuffle=True)` | Sort by the time column, `shuffle=False` / `TimeSeriesSplit` (tier1 if exactly one time column; else tier2) |
| Same entity on many rows | Group split | `train_test_split` / `KFold` ignoring the key | `GroupShuffleSplit` / `GroupKFold(groups=df[key])` (tier1 if the key is in the same frame) |
| Imbalanced classes | Stratified split; metrics robust to imbalance | No `stratify`; accuracy as the only metric | Add `stratify=y`; report `average_precision` / `balanced_accuracy` next to accuracy (tier1) |
| Duplicate rows | Dedupe before the split | Split on raw rows | `drop_duplicates(subset=features)` before the split (tier1) |
| Numbers stored as text | Cast at load | Code treats the column as numeric, or encodes it as a category | `pd.to_numeric(..., errors="coerce")` after the read + a check (tier1) |
| Columns not named in the code | Explicit feature list | `X = df.drop(columns=[target])` picks up new columns | Explicit feature list from the contract (tier1) |
| High-cardinality categorical | Target encoding fit inside folds | Encoding computed on all data | sklearn `TargetEncoder` in a `Pipeline` (tier1 if local) |
| Small dataset (< ~1k rows) | Cross-validation | One split reported as the result | Suggest `cross_val_score` (tier2: changes what is reported) |
| Code reads a column the data lacks; ambiguous time/group key; target proxies | — | — | Escalate with evidence (tier2: needs domain judgment) |

**3. Data contract.** One YAML per dataset, with:
- schema and types;
- split strategy and key columns;
- metrics;
- the drift setup per column: PSI+KS or Wasserstein for numbers, chi² or JS for categories,
  unseen-value rate for high-cardinality columns, and a seasonal reference window for
  time-dependent data.

The drift detectors read their configuration from the contract. Phase 5 turns it into checks.
Any change to the code or the data re-runs the fit check, which keeps the two in sync.

**4. Fixer for the tier1 rows above.** These are deterministic rewrites, not an LLM. Each fix
is a patch with an explanation, and its verification has to pass:
1. the patched code parses;
2. a re-scan shows the finding gone and no new findings;
3. once the sandbox exists, a re-run on sampled data still completes, and the tracer shows no
   test rows reaching `fit`.

Agents open PRs or issues only (rule 6). The PR text says the score may drop and that the lower
number is the honest one. Anything that fails verification is escalated instead. Phase 6 builds
on this fixer and verifier, adding LLM agents for tier2 and for fixes that aren't templated.

**Ground truth:**
- **Kaggle competitions define their own correct split.** A temporal or grouped official test set
  is a label for "how this data must be split".
- **Injected mismatches:** swap a correct split for a random one, drop `stratify`.
- **Hand triage.**

Later (Phase 4), profile → requirements can be learned from the corpus, carefully, because a
popular choice isn't necessarily a correct one.

*Exit:* mismatch recall and precision per row type on dev, the tier1 fix rate with verification
passing, and the share escalated.

### Phase 4: Learn from the corpus: FaultLine's own model
Goal: a model with **our own trained weights**, shipped with the tool and run locally. It
finds problems that rules miss and works **without** running the pipeline.

- **Unit of prediction: a site.** A site is each split / fit / evaluate call in the code.
- **Inputs:**
  - the static facts the rules already collect (`detectors/leakage.py`, `detectors/fit.py`);
  - the dataset profile (`contract.DatasetProfile`);
  - graph context.

  The rules double as feature extractors.
- **Labels:**
  - **required split type** from each Kaggle competition's official split (temporal / group /
    random);
  - runtime traces (Phase 3);
  - injected faults;
  - Yang et al.;
  - hand triage.
- **Splits:** train on **train**, calibrate on **dev**, report on **test** once.
- **Ablation:** rules only vs rules + model. The model ships only if it wins on dev.

| Version | Model | Training | Shipped as |
|---|---|---|---|
| v1 | Gradient boosting over site features (LightGBM) | CPU, minutes | LightGBM text model (a few MB) |
| v2 | Fine-tuned code encoder (UniXcoder / CodeBERT, ~125M params, open license): code window around the site + profile tokens → per-type probabilities | A free GPU (Kaggle/Colab), hours | ONNX, int8-quantized (~130 MB), CPU inference |
| v3 (optional) | LoRA fine-tune of a small open code LLM (e.g. Qwen2.5-Coder-1.5B, Apache-2.0) on (faulty code, verified fix) pairs from our fixer and injectors, for tier2 fix suggestions | A single GPU | Adapter weights / GGUF, run through Ollama |

**Deployment rules:**
- **Safe formats only.** No pickle/joblib for our own weights; that would trip our own
  deserialization rule.
- **Distribution:** versioned under `models/<name>/<version>/`, published as GitHub release
  assets (not in git), each with a model card. The card covers training data and its licenses,
  splits, and dev/test metrics next to the rule baseline. The SHA-256 is pinned in code.
- **Local inference:** pipelines and data never leave the user's machine.

**Needs:**
- the Kaggle corpus (thousands of notebooks), so the Kaggle login is on the critical path;
- dependencies approved when each version starts: `lightgbm` for v1, `onnxruntime` /
  `transformers` / `torch` for v2.

*Exit:* precision/recall on held-out competitions/repos, with and without the model; the model
card is published.

### Phase 5: Custom test suite at the most vulnerable points (pillar 2)
- **Criticality score per graph node:** centrality × external exposure × data sensitivity ×
  finding severity × the model's risk score. The highest-risk nodes get tests first, within a
  budget.
- **Test kinds, generated from the dataset profile/contract and the findings:**
  - **data contract:** types, ranges, nullability, allowed categories;
  - **leakage invariants:** train and test disjoint by row hash and entity key; `fit` only ever
    sees training rows, checked by the tracer;
  - **drift thresholds:** taken from the contract;
  - **behavioural/metamorphic:** shuffling rows doesn't change results; changing future rows
    doesn't change past features; permuting the id column doesn't change the score.
- **Reason per test:** each test states the finding or risk it guards and why that node was
  chosen.
- **Proven by mutation:** the injector puts the targeted fault in, and the test must fail; on
  clean code it must pass. A test that never fails is dropped.
- **Runs on sampled data** (Phase 3), so the suite takes seconds in CI.
- **Placement under a budget:** greedy by risk covered per test first; a contextual bandit only
  if it measurably covers more.
- **Backdoor detection** for image/text datasets (DL) also lands here.

*Exit:* every generated test is proven by mutation; risk coverage within the budget is reported.

### Phase 6: Agent team + RL + human in the loop (pillar 4)
- **Agents:**
  - **orchestrator:** deterministic;
  - **triage:** fix, escalate or skip; a contextual bandit, then RL;
  - **fixer:** the deterministic libcst templates, plus an RL-trained small code LLM for fixes
    that aren't templated;
  - **test-writer:** Phase 5;
  - **judge/verifier:** programmatic, with its own sandbox and hidden tests; it assigns the
    rewards;
  - **escalation reporter:** an LLM that writes the human report, grounded in the evidence.
- **Human in the loop:** big or hard issues go to a review queue in the dashboard with the
  evidence, options and a recommendation. Approve / reject / edit decisions are logged and become
  preference data (DPO) and bandit feedback.
- **Opens PRs or issues only.**
- **RL details:** see the RL track (rewards, judge, which technique and why).

*Exit:*
- tier1 fix rate with verification passing and no regressions on held-out repos, compared with
  the deterministic fixer and with SFT only;
- precision of the escalations.

### Phase 7: Backend, dashboard, monitoring
- **Backend:** FastAPI + SQLite (Postgres later): scans, findings, checks, history.
- **Dashboard:** Python + HTML (FastAPI + Jinja2 + Cytoscape.js), with these sections in order:
  1. the architecture map, with red edges at vulnerable points;
  2. the test suite: where to add tests, plus an "add" action;
  3. the fit score (only when a dataset is uploaded);
  4. the agents: the changed pipeline, the tests still at critical points, the human-attention
     queue, what's done, and the final pipeline.
- **Monitoring:** scheduled scans in CI/staging with trends; passive production collectors that
  send aggregates only.
*Exit:* repo connected → topology + first findings in under 15 minutes.

### Phase 8: Final evaluation and write-up
One run on the held-out test split, ablations, scans of 3–5 outside pipelines, write-up.

## Track: faults AI coding models make in ML pipelines
More pipeline code is now written by frontier models. These five mistakes are ones they're known
or reported to make; each needs a detector, a fix, and a measured rate.

**Evidence so far:**
- **Package hallucination:** well studied. Spracklen et al. measured at least 5.2% hallucinated
  packages for commercial models and 21.7% for open ones. A 2026 re-run on frontier models
  measured 4.6–6.1%.
- **Unsafe loading:** documented in AI-code security studies such as DeVAIC.
- **Leakage in AI-written ML code:** reported anecdotally only (models scale the full dataset
  before splitting). No measured rate yet, which is a gap FaultLine can fill.

| # | Fault | Example | Detect | Fix (tier) |
|---|---|---|---|---|
| A1 | **Statistics computed on all data before the split** | `df.fillna(df.mean())`, `(x - x.mean()) / x.std()`, `pd.get_dummies(pd.concat([train, test]))`, target encoding with `groupby(...)[target].mean()` before splitting | Extend the leakage dataflow from sklearn `fit` to pandas reductions (`mean/std/median/quantile/min/max`, `value_counts`, `groupby().agg`) whose result reaches data that's split later | Compute on the train split and apply to test, or use an sklearn `Pipeline`/`ColumnTransformer` (tier1 when local) |
| A2 | **Test set used for model selection** | `eval_set=[(X_test, y_test)]` with early stopping in XGBoost/LightGBM, `validation_data=(X_test, y_test)` in Keras, choosing the best model or threshold by test score | Test-side split outputs passed to `eval_set`/`validation_data`/`EarlyStopping`, or test score inside a loop that keeps the max (Yang et al.'s "multi" category) | Carve a validation split from train and use it for early stopping and selection; touch test once (tier1) |
| A3 | **Resampling before the split** | `SMOTE().fit_resample(X, y)` then `train_test_split` | `fit_resample` (imblearn) on data that is split later; synthetic test rows are built from training rows | Split first, resample the train split only, or use `imblearn.pipeline.Pipeline` (tier1) |
| A4 | **Hallucinated or wrong-name packages** | `pip install sklearn`, imports of packages that don't exist on PyPI, or exist only as recently registered look-alikes ("slopsquatting") | Check every import and requirement against PyPI: missing, very new, very low downloads, or a known wrong name; offline allowlist plus cached lookups | Known aliases fixed directly (`sklearn` → `scikit-learn`); anything unknown is escalated, never auto-installed (tier1 / tier2) |
| A5 | **Insecure loading defaults** | `trust_remote_code=True` (Hugging Face), `requests.get(url, verify=False)`, `pickle`/`joblib`/`torch.load(weights_only=False)` on downloaded files, dataset/model downloads with no checksum | Extend `pipeline_vulns`: `trust_remote_code`, `verify=False`, unverified downloads (the existing deserialization rules already cover pickle/yaml/torch) | `trust_remote_code=False` plus a pinned `revision`, TLS verification back on, checksum check before load, safe formats (tier1) |

Already on the list and also common in AI-written code: random splits of time-ordered data
(`data.leakage.temporal_split`) and splits done by slicing (PLAN.md open issues).

**How we measure it:**
- Build an **AI-generated pipelines corpus**: prompt models to write a pipeline for each Kaggle
  task in our corpus, then scan the output.
- Local open models run through Ollama, which is already installed. Frontier models need API
  keys (provider decided in Phase 6).
- Report the rate of each fault per model, and whether FaultLine's fix removes it without
  breaking the pipeline.
- This becomes its own corpus source, split by task like everything else.

**When:**
- A1–A3 detectors with the Phase 1 detector pass (before Phase 3, since the tracer confirms
  them).
- A4–A5 with Phase 2 (static).
- Fixes with the Phase 6 fixer.
- AI-generated corpus after Phase 2's Kaggle pull.

## Risks
- **Executing untrusted code:** Kaggle/GitHub notebooks only ever run inside the sandbox
  (Phase 3). Docker must be running for that phase.
- **Many notebooks won't run** (deps, GPUs, missing data). Measure it and report on the ones that do.
- **Fork leakage in the corpus** inflates scores unless dedupe and split-by-competition/author
  are enforced.
- **Optimistic injected recall:** always reported next to real-code precision and
  public-benchmark results.
- **Licenses and terms:** check Kaggle terms and dataset/repo licenses before bulk download; keep
  downloaded code and data out of git.
