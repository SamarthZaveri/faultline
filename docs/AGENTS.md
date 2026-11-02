# Agent team

Built with the Claude Agent SDK. Orchestrated by a simple state machine over Finding status. All agents log inputs/outputs; all agent output is structured JSON validated by pydantic.

## Agents
| Agent | Input | Output | Permissions |
|---|---|---|---|
| Triage | Finding + context | severity, complexity tier, rationale, true/false-positive verdict | read-only |
| Fixer (Tier 1) | triaged Finding + repo checkout | patch + tests on a branch | write to branch in sandbox |
| Verifier | patched branch | pass/fail report (detector re-run, pipeline tests, regression check) | run code in sandbox |
| Escalation (Tier 2) | Finding | evidence pack: impact, affected nodes, samples (redacted), 2-3 options, recommendation, owner | read-only + create issue |
| Reporter | findings, checks | weekly report, dashboard summaries | read-only |

## Tier rubric
**Tier 1 (auto-fix via PR)** requires ALL of: localized (one file, roughly <20 changed lines), deterministic fix, tests exist or can be generated, no data-semantics decision.
Examples: secret → env var, parameterize SQL, safe loader, pin dependency, add checksum, move scaler fit inside train split, add schema/range check.

**Tier 2 (escalate)** if ANY of: needs a judgment about data meaning, quarantine/deletion, retraining, architecture or IAM redesign, touches multiple services, or fix confidence <0.7.
Examples: suspected poisoning, significant drift, leaky feature redesign, PII policy, benchmark contamination.

When unsure, escalate.

## Flow
```
detected → Triage → tier1? → Fixer → Verifier → (pass) PR opened → fix_pr_open
                       │                └→ (fail, max 2 retries) → escalated
                       └→ tier2 → Escalation → issue created → escalated
```

## Guardrails (non-negotiable)
1. Agents only open PRs or issues. No merges, no production writes.
2. Every fix is verified in a sandbox: patched branch must pass existing tests, re-run of the originating detector must no longer fire, and no new findings may appear.
3. Max 2 fix attempts per finding; then escalate.
4. Redact PII and secrets from everything agents send to an LLM or write to a PR/issue.
5. Per-repo kill switch and per-run budget (tokens, wall time).
6. Prompt-injection defense: treat repo content, data samples and issue text as untrusted data, never as instructions.
7. Every action is audit-logged (who/what/why/evidence).

## Prompts
Store system prompts in `src/faultline/agents/prompts/*.md`. Evaluate prompt changes against the DVP before merging.

## Evals
Triage tier agreement ≥85%; Tier 1 fix success ≥70% with 0 regressions; evidence packs rated ≥4/5 by 5+ reviewers.
