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
**In scope (v1)**
- Pipelines written as Python scripts and Jupyter notebooks (pandas / scikit-learn first; PyTorch
  later). Then Airflow DAGs and dbt/SQL.
- Data problems: leakage (train/test contamination, preprocessing before split, duplicates across
  split, target-derived features), poisoning (label flips, injected rows, backdoor triggers),
  drift (covariate, label, schema/unit changes, missing spikes).
- Pipeline problems: secrets, SQL/shell injection, unsafe deserialization, dependencies, unverified
  downloads, unvalidated external input, bucket/IAM config.
- For every finding: the **file and lines** responsible, plus a node in the pipeline graph.
- Checks installed at the most critical nodes; agents that fix simple issues as PRs and escalate
  the rest; a Python + HTML dashboard; continuous scans in CI/staging; passive (read-only,
  aggregate-only) production monitoring.

**Out of scope (v1)**
- Attacks on trained models (adversarial examples, model extraction), real-time blocking of
  production traffic, streaming-engine internals, every orchestrator.
- Agents merging PRs or writing to production systems.
- Executing untrusted notebooks outside a sandbox.

## Where ML / DL / RL will be used
Rules and statistics come first. A learned component ships only if it beats the rule baseline
on dev repos.

| Component | Technique | Phase |
|---|---|---|
| Leakage and vulnerability rules | static rules (today) | 1 |
| Finding ranker / false-positive filter | ML classifier (gradient boosting on AST + graph features), trained on labeled findings | 4 |
| Static leakage detector that generalizes beyond the rules | ML on code (graph features first, then a fine-tuned code model such as CodeBERT if it measurably helps), trained on runtime-traced labels | 4 |
| Drift | statistics (PSI, KS, chi²) first, then a classifier two-sample test to localize drifted features | 3 |
| Label flips | cleanlab (confident learning: a model's out-of-sample probabilities) | 3 |
| Injected / out-of-distribution rows | Isolation Forest, per-source profiling | 3 |
| Backdoor triggers in image/text data | DL: spectral signatures / activation clustering on a trained network | 5 |
| Triage, fixer, escalation agents | LLM (provider decided in Phase 6) | 6 |
| Remediation or check-placement policy | RL (contextual bandit) only after accept/reject logs exist | optional, after v1 |

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

### Phase 4: Learn from the corpus
Goal: make the system learn from the Phase 2–3 data so it finds problems that rules miss, and
works **without** running code.
- Labels come from: runtime traces (Phase 3), injected faults, public benchmarks, hand triage.
- Train the finding ranker and the static leakage model on **train**, calibrate on **dev**, and
  report on **test** once.
- Ablation: rules only vs rules + ML. ML stays only if it wins on dev.
*Exit:* precision/recall on held-out competitions/repos, with and without ML.

### Phase 5: Harden (checks at critical points)
- Criticality score per graph node (centrality × external exposure × data sensitivity ×
  findings).
- Generate Pandera / Great Expectations checks at the top nodes within a budget. Verify each check
  catches the injected data faults it targets.
- Backdoor detection for image/text datasets (DL) lands here if Phase 3 covers tabular first.
*Exit:* checks catch their target faults; coverage of risk within budget reported.

### Phase 6: Agent team
- Triage (tier1 = local, deterministic fix; tier2 = needs judgment), fixer + verifier (sandbox
  re-runs the detector and the pipeline's tests), escalation report (evidence, options,
  recommendation). Opens PRs or issues only.
- Choose the LLM provider here.
*Exit:* share of tier1 findings fixed with passing, non-regressing verification.

### Phase 7: Backend, dashboard, monitoring
- **Backend:** FastAPI + SQLite (Postgres later): scans, findings, checks, history.
- **Dashboard:** Python + HTML (Jinja2 + a graph library), showing topology, critical sites,
  data flow, installed checks, findings board, reports.
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
