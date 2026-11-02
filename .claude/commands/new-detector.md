---
description: Scaffold a new detector (usage: /new-detector <category> <name>)
---
Create a detector `$ARGUMENTS` following docs/detectors/*.md:
1. If no spec exists, draft docs/detectors/<category>.md section (method, inputs, finding types, tier hints, targets) and ask me to confirm.
2. Create src/faultline/detectors/<category>/<name>.py with a `detect(...) -> list[Finding]` function and a module-level `DETECTOR = "<name>@0.1"`.
3. Create tests/detectors/<category>/test_<name>.py with failing tests from the relevant DVP fault ids.
4. Implement until tests pass. Run `make test lint`.
