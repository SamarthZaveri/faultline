"""A4/A5: package-name and insecure-default faults common in AI-written pipeline code."""

import pytest
from helpers import py

from faultline.detectors.pipeline_vulns import PipelineVulnDetector
from faultline.discovery.sources import SourceFile, SourceKind

DET = PipelineVulnDetector()


def found(src: SourceFile) -> list[tuple[str, str]]:
    return [(f.type, f.severity.value) for f in DET.detect(src)]


# ---- A5: insecure loading defaults ----
def test_trust_remote_code() -> None:
    code = (
        "from transformers import AutoModel\n"
        "m = AutoModel.from_pretrained('someone/model', trust_remote_code=True)\n"
    )
    assert found(py(code)) == [("pipeline.supply_chain.trust_remote_code", "high")]


def test_trust_remote_code_with_pinned_revision_is_lower() -> None:
    code = "m = AutoModel.from_pretrained('org/m', trust_remote_code=True, revision='3f2a9c1d8e7b6a5f4e3d2c1b0a9f8e7d6c5b4a39')\n"
    assert found(py(code)) == [("pipeline.supply_chain.trust_remote_code", "medium")]


@pytest.mark.parametrize(
    "code",
    [
        "m = AutoModel.from_pretrained('org/m', trust_remote_code=False)\n",
        "m = AutoModel.from_pretrained('org/m')\n",
    ],
)
def test_no_remote_code(code: str) -> None:
    assert found(py(code)) == []


@pytest.mark.parametrize(
    "code",
    [
        "import requests\nr = requests.get(url, verify=False)\n",
        "import requests\ns = requests.Session()\nr = s.post(url, json=body, verify=False)\n",
        "import httpx\nr = httpx.get(url, verify=False)\n",
    ],
)
def test_tls_verification_disabled(code: str) -> None:
    assert found(py(code)) == [("pipeline.transport.tls_verification_disabled", "medium")]


def test_tls_verification_on_is_clean() -> None:
    assert found(py("import requests\nr = requests.get(url, verify=True, timeout=10)\n")) == []


@pytest.mark.parametrize(
    ("code", "severity"),
    [
        (
            "import urllib.request\nurllib.request.urlretrieve('https://x.org/data/train.csv', 'train.csv')\n",
            "medium",
        ),
        ("import requests\nopen('m.pt', 'wb').write(requests.get('http://x.org/m.pt').content)\n", "high"),
    ],
)
def test_unverified_download(code: str, severity: str) -> None:
    assert ("pipeline.supply_chain.unverified_download", severity) in found(py(code))


def test_download_with_checksum_is_clean() -> None:
    code = (
        "import hashlib, urllib.request\n"
        "urllib.request.urlretrieve('https://x.org/data/train.csv', 'train.csv')\n"
        "assert hashlib.sha256(open('train.csv', 'rb').read()).hexdigest() == EXPECTED\n"
    )
    assert found(py(code)) == []


def test_api_calls_are_not_downloads() -> None:
    assert found(py("import requests\nr = requests.get('https://api.x.org/v1/items', timeout=5)\n")) == []


# ---- A4: wrong-name / look-alike packages ----
def test_wrong_package_names_in_requirements() -> None:
    src = SourceFile(
        path="requirements.txt",
        kind=SourceKind.requirements,
        code="pandas==2.2.2\nsklearn==0.0\nscikit-learn==1.5.0\nPIL==9.0\n",
    )
    findings = [f for f in DET.detect(src) if f.type == "pipeline.dependency.suspicious_package"]
    assert [(f.location.line_start, f.evidence.metrics["use_instead"]) for f in findings] == [
        (2, "scikit-learn"),
        (4, "pillow"),
    ]
