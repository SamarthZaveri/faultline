"""Detectors that look at the data flowing through a pipeline, not at its code (Phase 3).

drift.py      reference vs current data: schema, missing values, new categories, per-column
              shift, joint shift (classifier two-sample test), label prior, concept drift
poisoning.py  label flips (confident learning), injected rows (isolation forest), duplicate floods
overlap.py    rows present in both train and test
Needs the `data` extra: pip install -e .[data]
"""
