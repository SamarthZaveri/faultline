---
description: Add a planted fault to the DVP (usage: /add-fault <category> <description>)
---
1. Add an entry to fixtures/dvp_faults.yaml with the next free id, expected_type (valid taxonomy in docs/FINDING_SCHEMA.md), severity, tier, location, note.
2. Plant the fault in fixtures/dvp/ at that location.
3. Regenerate docs/DVP_FAULTS.md from the YAML.
4. Add a test asserting the expected finding. Run `make test`.
