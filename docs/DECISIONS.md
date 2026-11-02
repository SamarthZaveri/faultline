# Decisions log (append-only; newest first)

Format: `YYYY-MM-DD | decision | why | alternatives rejected | consequences`

- 2026-11-02 | Schema export writes LF line endings and takes an `out` dir; committed schemas are checked against the models by a test | regenerating on Windows produced CRLF churn; stale schemas would silently drift | hand-edited schemas, no drift test | `make schema` must be rerun after any model change or tests fail
- 2026-11-02 | `make scan` takes `PIPELINE=` instead of `PATH=` | `PATH=` overrode the shell PATH so recipes could not find python | keep `PATH=` | CLAUDE.md updated
- 2026-11-02 | Dev linters pinned to minor series (ruff 0.17.x, mypy 2.x, pytest <10); ruff uses its built-in default rule set, mypy uses pydantic plugin + disallow_untyped_defs | ruff 0.17 widened defaults, so unpinned versions made local and CI disagree; rule 8 requires full type hints | explicit narrow `select`, unpinned tools | bump pins deliberately and fix new findings in the same PR
- 2026-11-02 | Dev env is a Python 3.11 `.venv` created by `make setup` with pip; no uv or pre-commit for now | neither is listed in ARCHITECTURE.md; 3.11 is the minimum supported version | uv, global pip install, pre-commit hooks | revisit pre-commit once the DVP secrets allowlist exists
- 2026-11-02 | Makefile is cross-platform (Windows cmd.exe/sh via `ifeq ($(OS),Windows_NT)`, Linux/macOS); CI runs on ubuntu-latest and windows-latest | primary dev machine is Windows | WSL-only, separate PowerShell scripts | Windows devs need GNU make (`winget install ezwinports.make`)
- 2026-11-02 | docker-compose has only postgres:16 + redis:7, bound to 127.0.0.1, creds from `.env` (template `.env.example`) | api/dashboard do not exist yet; rule 8 no secrets in code | adding stub api/dashboard services | add services when T2.x/T9.x land
- 2026-11-02 | Finding is the single contract between detectors and everything else | decouples detectors from scorer/agents/dashboard | per-detector result types | schema changes need migrations
- 2026-11-02 | Rules/statistics first, ML/LLM only where documented | precision, explainability, cost | ML-everywhere | ML components must justify themselves via ablation
- 2026-11-02 | RL limited to an optional v2 contextual bandit | no simulator/reward in v1 | full RL for remediation | revisit after PR accept/reject logs exist
- 2026-11-02 | Agents only open PRs/issues; never merge or touch production | safety and trust | autonomous merge | humans stay in the loop
- 2026-11-02 | Splits by repository, thresholds tuned on dev repos only | avoid self-leakage in evaluation | random sample splits | smaller effective dataset
- 2026-11-02 | Prod mode is passive and summary-stats-only | privacy, low risk adoption | full raw-data scanning in prod | reduced detection power in prod
- TODO (set at project start) | Adoption target for "showstopper": e.g. 3 outside users or 500+ stars | prevents goalpost drift | | 
