from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from faultline.schema import Evidence, Finding, Location


def make(**kw: Any) -> Finding:
    base: dict[str, Any] = {
        "type": "pipeline.secret.hardcoded",
        "title": "Hardcoded AWS key",
        "severity": "high",
        "confidence": 0.95,
        "category": "pipeline",
        "location": Location(file="dags/ingest_orders.py", line_start=12),
        "evidence": Evidence(rule_id="gitleaks.aws", explanation="AWS key literal"),
        "detector": "secrets@0.1",
    }
    base.update(kw)
    return Finding(**base)


def test_stable_id_is_deterministic():
    assert make().id == make().id and len(make().id) == 16


def test_id_changes_with_location():
    other = make(location=Location(file="dags/other.py", line_start=3))
    assert make().id != other.id


def test_category_must_match_type_prefix():
    with pytest.raises(ValidationError):
        make(category="data")


def test_bad_type_rejected():
    with pytest.raises(ValidationError):
        make(type="not-a-type")


def test_location_needs_anchor():
    with pytest.raises(ValidationError):
        Location()


def test_confidence_bounds():
    with pytest.raises(ValidationError):
        make(confidence=1.5)


def test_all_dvp_fault_types_are_valid_types():
    faults = yaml.safe_load((Path(__file__).parents[1] / "fixtures" / "dvp_faults.yaml").read_text())[
        "faults"
    ]
    assert len(faults) == 40
    for f in faults:
        make(type=f["expected_type"], category=f["category"])
