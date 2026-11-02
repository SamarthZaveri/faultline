import json
from pathlib import Path

import pytest

from faultline.schema import Check, Finding, export

SCHEMAS = Path(__file__).parents[1] / "schemas"


@pytest.mark.parametrize(
    ("model", "filename"), [(Finding, "finding.schema.json"), (Check, "check.schema.json")]
)
def test_committed_schema_matches_model(model: type[Finding] | type[Check], filename: str) -> None:
    committed = json.loads((SCHEMAS / filename).read_text())
    assert committed == model.model_json_schema(), f"{filename} is stale; run `make schema`"


def test_export_writes_both_schemas(tmp_path: Path) -> None:
    export.main(out=tmp_path)
    assert json.loads((tmp_path / "finding.schema.json").read_text()) == Finding.model_json_schema()
    assert json.loads((tmp_path / "check.schema.json").read_text()) == Check.model_json_schema()
