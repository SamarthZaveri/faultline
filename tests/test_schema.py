import pytest
from pydantic import ValidationError

from faultline.schema import FINDING_TYPES, Category, Evidence, Finding, Location


def make(**kw: object) -> Finding:
    base: dict[str, object] = {
        "type": "pipeline.secret.hardcoded",
        "title": "Hardcoded AWS key",
        "severity": "high",
        "confidence": 0.9,
        "location": Location(file="dags/ingest.py", line_start=3),
        "evidence": Evidence(rule_id="secrets.aws_access_key", explanation="AWS key literal"),
        "detector": "pipeline_vulns@0.1",
    }
    base.update(kw)
    return Finding.model_validate(base)


def test_id_is_stable_and_location_sensitive() -> None:
    assert make().id == make().id
    assert len(make().id) == 16
    assert make().id != make(location=Location(file="dags/other.py", line_start=3)).id


def test_category_derived_from_type() -> None:
    assert make().category is Category.pipeline
    assert make(type="data.leakage.preprocessing_before_split").category is Category.data


def test_unknown_type_rejected() -> None:
    with pytest.raises(ValidationError):
        make(type="pipeline.secret.made_up")


def test_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        make(confidence=1.2)


def test_line_range_ordered() -> None:
    with pytest.raises(ValidationError):
        Location(file="a.py", line_start=5, line_end=2)


def test_category_serialized() -> None:
    assert make().model_dump(mode="json")["category"] == "pipeline"


def test_taxonomy_covers_every_problem_family() -> None:
    prefixes = {".".join(t.split(".")[:2]) for t in FINDING_TYPES}
    assert {
        "data.poisoning",
        "data.drift",
        "data.leakage",
        "pipeline.secret",
        "pipeline.injection",
        "pipeline.deserialization",
        "pipeline.dependency",
    } <= prefixes
