"""Regenerate schemas/finding.schema.json:  python -m faultline.schema.export"""

import json
from pathlib import Path

from .finding import Check, Finding

DEFAULT_OUT = Path(__file__).resolve().parents[3] / "schemas"


def main(out: Path = DEFAULT_OUT) -> None:
    out.mkdir(exist_ok=True)
    for name, model in (("finding", Finding), ("check", Check)):
        # newline="\n" keeps regenerated files byte-identical on Windows.
        (out / f"{name}.schema.json").write_text(
            json.dumps(model.model_json_schema(), indent=2) + "\n", encoding="utf-8", newline="\n"
        )
    print("wrote", out)


if __name__ == "__main__":
    main()
