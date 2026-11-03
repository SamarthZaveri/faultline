"""Inject known faults into real code to get labeled ground truth.

Each mutator inserts one realistic fault into a real file (at the top level after imports, or
right before/after a train/test split) and records exactly where it went. Several templates
with randomized names per mutator keep detectors from overfitting to one string. Injected
recall is still optimistic (same author for templates and rules), so public benchmarks and
hand-triaged real findings are reported alongside it.
"""

from __future__ import annotations

import ast
import random
import string
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol

from faultline.detectors import _ast as A
from faultline.detectors.pipeline_vulns import is_placeholder
from faultline.discovery.sources import SourceFile


@dataclass(frozen=True)
class InjectedFault:
    mutator: str
    expected_type: str
    file: str
    line_start: int
    line_end: int


class Mutator(Protocol):
    name: str
    expected_type: str

    def applicable(self, src: SourceFile) -> bool: ...

    def apply(self, src: SourceFile, rng: random.Random) -> tuple[SourceFile, InjectedFault]: ...


@dataclass(frozen=True)
class _Site:
    after: int  # insert after this line (0 = top of file)
    indent: str
    ctx: dict[str, str] = field(default_factory=dict)


Template = Callable[[random.Random, dict[str, str]], list[str]]
Placement = Literal["top", "before_split", "after_split"]


def insert_lines(src: SourceFile, after: int, snippet: list[str], indent: str) -> tuple[SourceFile, int, int]:
    lines = src.code.splitlines()
    n = len(snippet)
    new = lines[:after] + [indent + s if s else s for s in snippet] + lines[after:]
    # A cell starting right at the insertion point absorbs the new lines; later cells shift.
    starts = tuple(s + n if s > after + 1 else s for s in src.cell_starts)
    return SourceFile(src.path, src.kind, "\n".join(new) + "\n", starts), after + 1, after + n


def _indent_of(src: SourceFile, line: int) -> str:
    text = src.line(line)
    return text[: len(text) - len(text.lstrip())]


def _after_imports(tree: ast.Module, src: SourceFile) -> int:
    last = 0
    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            last = stmt.end_lineno or stmt.lineno
    if last == 0 and src.line(1).startswith("#"):
        last = 1  # keep shebang / encoding line first
    return last


def _split_assigns(tree: ast.Module) -> list[tuple[ast.Assign, ast.Call]]:
    out = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and A.last_segment(node.value) == "train_test_split"
        ):
            out.append((node, node.value))
    return sorted(out, key=lambda t: t[0].lineno)


@dataclass(frozen=True)
class TemplateMutator:
    name: str
    expected_type: str
    placement: Placement
    templates: tuple[Template, ...]
    needs: tuple[str, ...] = ()  # context keys the site must provide

    def sites(self, src: SourceFile) -> list[_Site]:
        tree = src.tree
        if tree is None:
            return []
        if self.placement == "top":
            return [_Site(_after_imports(tree, src), "")]
        sites = []
        for stmt, call in _split_assigns(tree):
            ctx: dict[str, str] = {}
            if self.placement == "before_split":
                for key, a in zip(("X", "y"), call.args, strict=False):
                    if isinstance(a, ast.Name):
                        ctx[key] = a.id
                site = _Site(stmt.lineno - 1, _indent_of(src, stmt.lineno), ctx)
            else:
                target = stmt.targets[0]
                n_arrays = len([a for a in call.args if not isinstance(a, ast.Starred)])
                if isinstance(target, (ast.Tuple, ast.List)) and len(target.elts) == 2 * n_arrays:
                    for key, elt in zip(("X_test", "y_test"), target.elts[1::2], strict=False):
                        if isinstance(elt, ast.Name):
                            ctx[key] = elt.id
                site = _Site(stmt.end_lineno or stmt.lineno, _indent_of(src, stmt.lineno), ctx)
            if all(k in site.ctx for k in self.needs):
                sites.append(site)
        return sites

    def applicable(self, src: SourceFile) -> bool:
        return bool(self.sites(src))

    def apply(self, src: SourceFile, rng: random.Random) -> tuple[SourceFile, InjectedFault]:
        site = rng.choice(self.sites(src))
        snippet = rng.choice(self.templates)(rng, site.ctx)
        mutated, start, end = insert_lines(src, site.after, snippet, site.indent)
        return mutated, InjectedFault(self.name, self.expected_type, src.path, start, end)


# ---- random values ---------------------------------------------------------------------------
def _ident(rng: random.Random, *bases: str) -> str:
    return f"{rng.choice(bases)}_{rng.randint(1, 99)}"


def _token(rng: random.Random, alphabet: str, n: int) -> str:
    while True:
        v = "".join(rng.choice(alphabet) for _ in range(n))
        if not is_placeholder(v):
            return v


def _password(rng: random.Random) -> str:
    return _token(rng, string.ascii_letters + string.digits + "!#%^*-_", rng.randint(12, 16))


ALNUM = string.ascii_letters + string.digits
TABLES = ("orders", "customers", "events", "transactions", "features", "labels")
COLUMNS = ("id", "user_id", "region", "day", "batch_id", "source")
PARAMS = ("path", "fname", "src", "key", "batch", "value")


# ---- templates: pipeline vulnerabilities -----------------------------------------------------
def _secret_aws(rng: random.Random, ctx: dict[str, str]) -> list[str]:
    key = "AKIA" + _token(rng, string.ascii_uppercase + string.digits, 16)
    return [f'{rng.choice(["AWS_ACCESS_KEY_ID", "aws_key", "ACCESS_KEY"])} = "{key}"']


def _secret_password(rng: random.Random, ctx: dict[str, str]) -> list[str]:
    return [f'{rng.choice(["DB_PASSWORD", "db_password", "PASSWORD", "warehouse_pwd"])} = "{_password(rng)}"']


def _secret_url(rng: random.Random, ctx: dict[str, str]) -> list[str]:
    scheme = rng.choice(["postgresql", "mysql", "mongodb", "redshift"])
    user = rng.choice(["etl", "admin", "loader", "analytics"])
    host = rng.choice(["db.internal", "warehouse.corp", "10.0.3.7"])
    url = f"{scheme}://{user}:{_password(rng)}@{host}:5432/{rng.choice(TABLES)}"
    return [f'{rng.choice(["DATABASE_URL", "conn_str", "DB_URI"])} = "{url}"']


def _secret_token(rng: random.Random, ctx: dict[str, str]) -> list[str]:
    return [f'GITHUB_TOKEN = "ghp_{_token(rng, ALNUM, 36)}"']


def _secret_dict(rng: random.Random, ctx: dict[str, str]) -> list[str]:
    return [f'{_ident(rng, "creds", "db_config")} = {{"user": "etl", "password": "{_password(rng)}"}}']


def _secret_api_key(rng: random.Random, ctx: dict[str, str]) -> list[str]:
    return [f'{rng.choice(["API_KEY", "apiKey", "client_secret"])} = "{_token(rng, ALNUM, 32)}"']


def _sql_fstring(rng: random.Random, ctx: dict[str, str]) -> list[str]:
    fn, p, t, c = (
        _ident(rng, "fetch_rows", "load_batch"),
        rng.choice(PARAMS),
        rng.choice(TABLES),
        rng.choice(COLUMNS),
    )
    return [f"def {fn}(conn, {p}):", f"    return conn.execute(f\"SELECT * FROM {t} WHERE {c} = '{{{p}}}'\")"]


def _sql_percent(rng: random.Random, ctx: dict[str, str]) -> list[str]:
    fn, p, t, c = (
        _ident(rng, "delete_rows", "purge"),
        rng.choice(PARAMS),
        rng.choice(TABLES),
        rng.choice(COLUMNS),
    )
    return [f"def {fn}(cursor, {p}):", f'    cursor.execute("DELETE FROM {t} WHERE {c} = %s" % {p})']


def _sql_concat(rng: random.Random, ctx: dict[str, str]) -> list[str]:
    fn, p, t, c = (
        _ident(rng, "lookup", "get_rows"),
        rng.choice(PARAMS),
        rng.choice(TABLES),
        rng.choice(COLUMNS),
    )
    return [
        f"def {fn}(cur, {p}):",
        f'    query = "SELECT * FROM {t} WHERE {c} = \'" + {p} + "\'"',
        "    cur.execute(query)",
        "    return cur.fetchall()",
    ]


def _sql_read_sql(rng: random.Random, ctx: dict[str, str]) -> list[str]:
    fn, p, t, c = (
        _ident(rng, "read_table", "extract"),
        rng.choice(PARAMS),
        rng.choice(TABLES),
        rng.choice(COLUMNS),
    )
    return [
        f"def {fn}(conn, {p}):",
        "    import pandas as pd",
        f'    return pd.read_sql("SELECT * FROM {t} WHERE {c} > {{}}".format({p}), conn)',
    ]


def _sql_spark(rng: random.Random, ctx: dict[str, str]) -> list[str]:
    fn, p, t, c = (
        _ident(rng, "daily_slice", "partition"),
        rng.choice(PARAMS),
        rng.choice(TABLES),
        rng.choice(COLUMNS),
    )
    return [f"def {fn}(spark, {p}):", f"    return spark.sql(f\"select * from {t} where {c} = '{{{p}}}'\")"]


def _shell_run(rng: random.Random, ctx: dict[str, str]) -> list[str]:
    fn, p = _ident(rng, "decompress", "convert_file"), rng.choice(PARAMS)
    cmd = rng.choice(["gzip -d", "convert", "unzip -o", "tar xzf"])
    return [f"def {fn}({p}):", "    import subprocess", f'    subprocess.run(f"{cmd} {{{p}}}", shell=True)']


def _shell_system(rng: random.Random, ctx: dict[str, str]) -> list[str]:
    fn, p = _ident(rng, "unpack", "fetch_file"), rng.choice(PARAMS)
    return [f"def {fn}({p}):", "    import os", f'    os.system("unzip -o " + {p})']


def _shell_check_output(rng: random.Random, ctx: dict[str, str]) -> list[str]:
    fn, p = _ident(rng, "count_lines", "file_info"), rng.choice(PARAMS)
    return [
        f"def {fn}({p}):",
        "    import subprocess",
        f'    return subprocess.check_output("wc -l %s" % {p}, shell=True)',
    ]


def _shell_popen(rng: random.Random, ctx: dict[str, str]) -> list[str]:
    fn, p = _ident(rng, "sync_bucket", "copy_in"), rng.choice(PARAMS)
    return [f"def {fn}({p}):", "    import os", f'    return os.popen(f"aws s3 cp {{{p}}} .").read()']


def _deser(module: str, call: str) -> Template:
    def template(rng: random.Random, ctx: dict[str, str]) -> list[str]:
        fn, p = _ident(rng, "load_artifact", "get_model", "read_cache"), rng.choice(PARAMS)
        return [f"def {fn}({p}):", f"    import {module}", f"    return {call.format(p=p)}"]

    return template


def _deser_pickle_file(rng: random.Random, ctx: dict[str, str]) -> list[str]:
    fn, p = _ident(rng, "load_artifact", "restore"), rng.choice(PARAMS)
    return [
        f"def {fn}({p}):",
        "    import pickle",
        f'    with open({p}, "rb") as fh:',
        "        return pickle.load(fh)",
    ]


def _deser_pickle_remote(rng: random.Random, ctx: dict[str, str]) -> list[str]:
    fn = _ident(rng, "download_model", "fetch_artifact")
    return [
        f"def {fn}(url):",
        "    import pickle",
        "    import requests",
        "    return pickle.loads(requests.get(url, timeout=30).content)",
    ]


# ---- templates: leakage ----------------------------------------------------------------------
PREP = (
    ("StandardScaler", "sklearn.preprocessing"),
    ("MinMaxScaler", "sklearn.preprocessing"),
    ("RobustScaler", "sklearn.preprocessing"),
    ("QuantileTransformer", "sklearn.preprocessing"),
    ("PCA", "sklearn.decomposition"),
    ("SimpleImputer", "sklearn.impute"),
)


def _prep_object(rng: random.Random, ctx: dict[str, str]) -> list[str]:
    cls, mod = rng.choice(PREP)
    v, x = _ident(rng, "scaler", "prep", "transformer"), ctx["X"]
    return [f"from {mod} import {cls}", f"{v} = {cls}()", f"{x} = {v}.fit_transform({x})"]


def _prep_inline(rng: random.Random, ctx: dict[str, str]) -> list[str]:
    cls, mod = rng.choice(PREP)
    x = ctx["X"]
    return [f"from {mod} import {cls}", f"{x} = {cls}().fit_transform({x})"]


def _prep_fit_then_transform(rng: random.Random, ctx: dict[str, str]) -> list[str]:
    cls, mod = rng.choice(PREP)
    v, x = _ident(rng, "norm", "sc"), ctx["X"]
    return [f"from {mod} import {cls}", f"{v} = {cls}()", f"{v}.fit({x})", f"{x} = {v}.transform({x})"]


def _select_kbest(rng: random.Random, ctx: dict[str, str]) -> list[str]:
    x, y, k = ctx["X"], ctx["y"], rng.choice([5, 10, 20])
    return [
        "from sklearn.feature_selection import SelectKBest, f_classif",
        f"{x} = SelectKBest(f_classif, k=min({k}, {x}.shape[1])).fit_transform({x}, {y})",
    ]


def _select_percentile(rng: random.Random, ctx: dict[str, str]) -> list[str]:
    v, x, y = _ident(rng, "selector", "fs"), ctx["X"], ctx["y"]
    return [
        "from sklearn.feature_selection import SelectPercentile",
        f"{v} = SelectPercentile(percentile={rng.choice([30, 50, 70])})",
        f"{x} = {v}.fit_transform({x}, {y})",
    ]


def _select_rfe(rng: random.Random, ctx: dict[str, str]) -> list[str]:
    x, y = ctx["X"], ctx["y"]
    return [
        "from sklearn.feature_selection import RFE",
        "from sklearn.linear_model import LogisticRegression",
        f"{x} = RFE(LogisticRegression(max_iter=500), n_features_to_select=5).fit_transform({x}, {y})",
    ]


def _test_refit_scaler(rng: random.Random, ctx: dict[str, str]) -> list[str]:
    cls, mod = rng.choice(PREP[:4])
    xt = ctx["X_test"]
    return [f"from {mod} import {cls}", f"{xt} = {cls}().fit_transform({xt})"]


def _test_object(rng: random.Random, ctx: dict[str, str]) -> list[str]:
    cls, mod = rng.choice(PREP)
    v, xt = _ident(rng, "test_scaler", "sc_test"), ctx["X_test"]
    return [f"from {mod} import {cls}", f"{v} = {cls}()", f"{xt} = {v}.fit_transform({xt})"]


def _test_probe_model(rng: random.Random, ctx: dict[str, str]) -> list[str]:
    if "y_test" not in ctx:
        return _test_refit_scaler(rng, ctx)
    v = _ident(rng, "probe", "check_model")
    return [
        "from sklearn.linear_model import LogisticRegression",
        f"{v} = LogisticRegression(max_iter=200).fit({ctx['X_test']}, {ctx['y_test']})",
    ]


MUTATORS: tuple[TemplateMutator, ...] = (
    TemplateMutator(
        "hardcoded_secret",
        "pipeline.secret.hardcoded",
        "top",
        (_secret_aws, _secret_password, _secret_url, _secret_token, _secret_dict, _secret_api_key),
    ),
    TemplateMutator(
        "sql_injection",
        "pipeline.injection.sql",
        "top",
        (_sql_fstring, _sql_percent, _sql_concat, _sql_read_sql, _sql_spark),
    ),
    TemplateMutator(
        "shell_injection",
        "pipeline.injection.shell",
        "top",
        (_shell_run, _shell_system, _shell_check_output, _shell_popen),
    ),
    TemplateMutator(
        "unsafe_deserialization",
        "pipeline.deserialization.unsafe",
        "top",
        (
            _deser_pickle_file,
            _deser_pickle_remote,
            _deser("joblib", "joblib.load({p})"),
            _deser("pandas", "pandas.read_pickle({p})"),
            _deser("yaml", "yaml.load(open({p}), Loader=yaml.Loader)"),
            _deser("numpy", "numpy.load({p}, allow_pickle=True)"),
            _deser("torch", "torch.load({p}, weights_only=False)"),
        ),
    ),
    TemplateMutator(
        "preprocess_before_split",
        "data.leakage.preprocessing_before_split",
        "before_split",
        (_prep_object, _prep_inline, _prep_fit_then_transform),
        needs=("X",),
    ),
    TemplateMutator(
        "feature_selection_full",
        "data.leakage.feature_selection_full",
        "before_split",
        (_select_kbest, _select_percentile, _select_rfe),
        needs=("X", "y"),
    ),
    TemplateMutator(
        "test_in_training",
        "data.leakage.test_in_training",
        "after_split",
        (_test_refit_scaler, _test_object, _test_probe_model),
        needs=("X_test",),
    ),
)
