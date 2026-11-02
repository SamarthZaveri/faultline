---
description: Start a roadmap task (usage: /start-task T3.1)
---
1. Read docs/ROADMAP.md and find task $ARGUMENTS. Read its acceptance criteria.
2. Read the docs it depends on (detector spec, schema, DVP faults).
3. Propose a short plan (files, tests, risks) and wait for approval.
4. Implement test-first. Run `make test lint`.
5. Update ROADMAP.md status and DECISIONS.md if you made a choice. Summarize results.
