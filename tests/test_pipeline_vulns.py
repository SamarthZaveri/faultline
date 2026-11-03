import pytest
from helpers import FAKE_AWS_KEY, FAKE_GH_TOKEN, py

from faultline.detectors.pipeline_vulns import PipelineVulnDetector
from faultline.discovery.sources import SourceFile, SourceKind

DET = PipelineVulnDetector()


def types(src: SourceFile) -> list[str]:
    return [f.type for f in DET.detect(src)]


# ---- secrets ----
def test_provider_key_in_code_is_flagged_and_masked() -> None:
    findings = DET.detect(py(f'AWS_KEY = "{FAKE_AWS_KEY}"\n'))
    assert [f.type for f in findings] == ["pipeline.secret.hardcoded"]
    f = findings[0]
    assert f.location.line_start == 1
    assert FAKE_AWS_KEY not in f.model_dump_json()  # never leak the raw secret


@pytest.mark.parametrize(
    "line",
    [
        'DB_PASSWORD = "s3cr3t-Pa55word"\n',
        'connect(host="db", password="hunter2hunter2")\n',
        'cfg = {"api_key": "q8Zr2LmX9vT4wK1p"}\n',
        'URI = "postgresql://admin:Sup3rS3cret@db.internal:5432/sales"\n',
        f'token = "{FAKE_GH_TOKEN}"\n',
    ],
)
def test_named_and_url_secrets(line: str) -> None:
    assert types(py(line)) == ["pipeline.secret.hardcoded"]


@pytest.mark.parametrize(
    "line",
    [
        'password = os.environ["DB_PASSWORD"]\n',
        'password = "changeme"\n',
        'PASSWORD_ENV = "DB_PASSWORD"\n',
        'max_tokens = "4096"\n',
        'tokenizer = "bert-base-uncased"\n',
        'password_hint = "Your first pet"\n',
        'api_key = "<YOUR_API_KEY>"\n',
        'URI = "postgresql://user:{password}@db/x"\n',
        'KEY = "AKIAIOSFODNN7EXAMPLE"\n',  # AWS documentation key
        'boto3.client("s3", aws_access_key_id="123abc", aws_secret_access_key="abc123")\n',
        'password = "qwerty123"\n',
    ],
)
def test_secret_false_positives(line: str) -> None:
    assert types(py("import os\n" + line)) == []


def test_secret_in_config_file() -> None:
    src = SourceFile(
        path="config/settings.yaml",
        kind=SourceKind.config,
        code="db:\n  host: localhost\n  password: Pr0d-Db-Pass!\n  user: ${DB_USER}\n",
    )
    findings = DET.detect(src)
    assert [(f.type, f.location.line_start) for f in findings] == [("pipeline.secret.hardcoded", 3)]
    assert "Pr0d-Db-Pass!" not in findings[0].model_dump_json()


# ---- injection ----
SQL_BAD = [
    'def q(conn, uid):\n    return conn.execute(f"SELECT * FROM orders WHERE id = {uid}")\n',
    'def q(cur, uid):\n    cur.execute("DELETE FROM t WHERE id = %s" % uid)\n',
    'def q(cur, name):\n    cur.execute("SELECT * FROM t WHERE n = \'" + name + "\'")\n',
    'import pandas as pd\ndef q(conn, t):\n    return pd.read_sql("SELECT * FROM {}".format(t), conn)\n',
    'def q(cur, uid):\n    sql = f"UPDATE t SET x = 1 WHERE id = {uid}"\n    cur.execute(sql)\n',
    "def q(spark, day):\n    return spark.sql(f\"select * from events where day = '{day}'\")\n",
]


@pytest.mark.parametrize("code", SQL_BAD)
def test_sql_injection(code: str) -> None:
    assert types(py(code)) == ["pipeline.injection.sql"]


@pytest.mark.parametrize(
    "code",
    [
        'def q(cur, uid):\n    cur.execute("SELECT * FROM t WHERE id = %s", (uid,))\n',
        'def q(cur):\n    cur.execute("SELECT 1")\n',
        'def run(executor, job):\n    executor.execute(f"job-{job}")\n',  # not SQL
    ],
)
def test_sql_safe(code: str) -> None:
    assert types(py(code)) == []


@pytest.mark.parametrize(
    "code",
    [
        'import subprocess\ndef c(f):\n    subprocess.run(f"convert {f} out.png", shell=True)\n',
        'import os\ndef c(f):\n    os.system("gzip " + f)\n',
        'from subprocess import check_output\ndef c(d):\n    check_output("ls %s" % d, shell=True)\n',
    ],
)
def test_shell_injection(code: str) -> None:
    assert types(py(code)) == ["pipeline.injection.shell"]


@pytest.mark.parametrize(
    "code",
    [
        'import subprocess\ndef c(f):\n    subprocess.run(["convert", f, "out.png"])\n',
        'import subprocess\nsubprocess.run("ls -la", shell=True)\n',
        'import os\nos.system("clear")\n',
    ],
)
def test_shell_safe(code: str) -> None:
    assert types(py(code)) == []


# ---- deserialization ----
@pytest.mark.parametrize(
    "code",
    [
        "import pickle\ndef l(p):\n    with open(p, 'rb') as f:\n        return pickle.load(f)\n",
        "import pickle as pk\nobj = pk.loads(blob)\n",
        "from joblib import load\nm = load(path)\n",
        "import pandas as pd\ndf = pd.read_pickle(url)\n",
        "import yaml\ncfg = yaml.load(open('r.yaml'))\n",
        "import yaml\ncfg = yaml.load(f, Loader=yaml.Loader)\n",
        "import torch\nm = torch.load(p, weights_only=False)\n",
        "import numpy as np\na = np.load(p, allow_pickle=True)\n",
    ],
)
def test_unsafe_deserialization(code: str) -> None:
    assert types(py(code)) == ["pipeline.deserialization.unsafe"]


@pytest.mark.parametrize(
    "code",
    [
        "import yaml\ncfg = yaml.safe_load(f)\n",
        "import yaml\ncfg = yaml.load(f, Loader=yaml.SafeLoader)\n",
        "import torch\nm = torch.load(p)\n",
        "import numpy as np\na = np.load(p)\n",
        "import json\nd = json.load(f)\n",
    ],
)
def test_safe_deserialization(code: str) -> None:
    assert types(py(code)) == []


# ---- dependencies ----
def test_unpinned_requirements() -> None:
    src = SourceFile(
        path="requirements.txt",
        kind=SourceKind.requirements,
        code="# deps\npandas==2.2.2\nscikit-learn\nrequests>=2\n-r base.txt\ngit+https://x/y.git@abc\n",
    )
    findings = DET.detect(src)
    assert [f.type for f in findings] == ["pipeline.dependency.unpinned"]
    assert findings[0].location.line_start == 3
    assert findings[0].evidence.metrics["unpinned"] == 2


def test_syntax_error_returns_nothing() -> None:
    assert types(py("def broken(:\n")) == []


@pytest.mark.parametrize(
    ("code", "severity"),
    [
        ("import pickle\nm = pickle.load(open('model.pkl', 'rb'))\n", "medium"),
        ("import joblib\nm = joblib.load('model.joblib')\n", "low"),
        ("import pickle\ndef f(p):\n    return pickle.load(open(p, 'rb'))\n", "high"),
        ("import pickle, requests\nm = pickle.loads(requests.get(u).content)\n", "critical"),
    ],
)
def test_deserialization_severity_reflects_source(code: str, severity: str) -> None:
    [f] = DET.detect(py(code))
    assert f.severity == severity


def test_huge_single_line_config_is_fast() -> None:
    # Regression: a real test fixture with a 189K-character line made a regex go quadratic.
    import time

    line = "args: " + "a" * 100_000 + "://" + "b" * 50_000 + "@" + "c" * 40_000
    src = SourceFile(path="big.yaml", kind=SourceKind.config, code=line + "\n")
    start = time.perf_counter()
    DET.detect(src)
    assert time.perf_counter() - start < 2.0
