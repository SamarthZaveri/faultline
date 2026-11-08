"""Pipeline vulnerabilities found by static rules: secrets, SQL/shell injection, unsafe
deserialization, unpinned dependencies. Rules first: every hit names the rule that fired.
"""

from __future__ import annotations

import ast
import hashlib
import re
from itertools import pairwise

from faultline.detectors import _ast as A
from faultline.detectors.base import make_finding
from faultline.discovery.sources import SourceFile, SourceKind
from faultline.schema import Finding, Severity

# ---- secrets ---------------------------------------------------------------------------------
PROVIDER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("aws_access_key", re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[0-9A-Z]{16}(?![A-Z0-9])")),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})")),
    ("slack_token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}")),
    ("stripe_live_key", re.compile(r"\b[sr]k_live_[0-9A-Za-z]{20,}")),
    ("llm_api_key", re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}")),
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----")),
)
# Bounded repetition: unbounded runs went quadratic on a 189K-char line in a real repo.
URL_CREDENTIALS = re.compile(
    r"[a-z][a-z0-9+.-]{0,20}://[^\s/:@'\"]{1,128}:([^\s/@'\"]{1,256})@", re.IGNORECASE
)
SECRET_WORDS = frozenset(
    {"password", "passwd", "pwd", "secret", "token", "apikey", "credential", "credentials"}
)
SECRET_PAIRS = frozenset(
    {("api", "key"), ("access", "key"), ("private", "key"), ("secret", "key"), ("client", "secret")}
)
NOT_SECRET_WORDS = frozenset(
    {
        "name", "path", "file", "dir", "url", "uri", "env", "type", "length", "len", "count", "max",
        "min", "id", "prefix", "field", "header", "param", "var", "format", "pattern", "regex",
        "hint", "label", "placeholder", "prompt", "input", "column", "col", "limit", "size", "num",
        "endpoint", "policy", "expiry", "expires", "ttl",
    }
)  # fmt: skip
PLACEHOLDER_WORDS = (
    "changeme", "change_me", "change-me", "your", "example", "dummy", "placeholder", "xxxx",
    "redacted", "replace", "todo", "none", "null", "test", "fake", "sample", "secret_here",
    "abc123", "123abc", "qwerty", "123456",
)  # fmt: skip
PLACEHOLDER_SHAPES = re.compile(r"^(<.*>|\$\{?.*|\{\{.*\}\}|%\(.*\)s|\*+|x+|\.+|true|false)$", re.IGNORECASE)
CONFIG_PAIR = re.compile(r"""^\s*["']?([A-Za-z0-9_.\-]+)["']?\s*[:=]\s*["']?([^"'\s#,]+)""")


def _tokens(name: str) -> list[str]:
    return [t.lower() for t in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", name)]


def is_secret_name(name: str) -> bool:
    toks = _tokens(name)
    if any(t in NOT_SECRET_WORDS for t in toks):
        return False
    return any(t in SECRET_WORDS for t in toks) or any(p in SECRET_PAIRS for p in pairwise(toks))


def is_placeholder(value: str) -> bool:
    v = value.strip()
    low = v.lower()
    if len(v) < 6 or " " in v or "{" in v or PLACEHOLDER_SHAPES.match(v):
        return True
    if any(w in low for w in PLACEHOLDER_WORDS):
        return True
    if re.fullmatch(r"[A-Z][A-Z0-9_]+", v) and "_" in v:  # an env var *name*, not a value
        return True
    return v.startswith(("/", "./", "http://", "https://"))


def mask(value: str) -> str:
    digest = hashlib.sha256(value.encode()).hexdigest()[:8]
    return f"{value[:4]}…<sha256:{digest}>"


def _provider_hits(text: str) -> list[tuple[str, str]]:
    hits = [(rule, m.group(0)) for rule, pat in PROVIDER_PATTERNS for m in pat.finditer(text)]
    hits = [(r, v) for r, v in hits if "EXAMPLE" not in v]  # vendor documentation keys
    for m in URL_CREDENTIALS.finditer(text) if "://" in text and "@" in text else ():
        pw = m.group(1)
        if not is_placeholder(pw) and not pw.startswith(("$", "{", "%")):
            hits.append(("url_credentials", pw))
    return hits


# ---- sinks -----------------------------------------------------------------------------------
SQL_METHODS = frozenset({"execute", "executemany", "executescript", "read_sql", "read_sql_query", "sql"})
SQL_NEEDS_KEYWORD = frozenset({"execute", "executemany", "executescript"})
SQL_KEYWORDS = re.compile(
    r"\b(select|insert|update|delete|from|where|into|values|drop|create|alter)\b", re.IGNORECASE
)
SUBPROCESS = frozenset(
    {
        "subprocess.run",
        "subprocess.call",
        "subprocess.Popen",
        "subprocess.check_call",
        "subprocess.check_output",
    }
)
ALWAYS_SHELL = frozenset(
    {"os.system", "os.popen", "subprocess.getoutput", "subprocess.getstatusoutput", "commands.getoutput"}
)
PICKLE_LIKE = {
    "pickle.load": Severity.high,
    "pickle.loads": Severity.high,
    "_pickle.load": Severity.high,
    "_pickle.loads": Severity.high,
    "cPickle.load": Severity.high,
    "cPickle.loads": Severity.high,
    "dill.load": Severity.high,
    "dill.loads": Severity.high,
    "cloudpickle.load": Severity.high,
    "cloudpickle.loads": Severity.high,
    "marshal.load": Severity.high,
    "marshal.loads": Severity.high,
    "shelve.open": Severity.medium,
    "joblib.load": Severity.medium,
    "sklearn.externals.joblib.load": Severity.medium,
    "pandas.read_pickle": Severity.medium,
}
UNSAFE_YAML_LOADERS = frozenset({"yaml.Loader", "yaml.UnsafeLoader", "yaml.CLoader", "yaml.CUnsafeLoader"})


class PipelineVulnDetector:
    name = "pipeline_vulns"
    version = "0.1"

    def detect(self, src: SourceFile) -> list[Finding]:
        if src.kind is SourceKind.requirements:
            return self._requirements(src)
        if src.kind is SourceKind.config:
            return self._config_secrets(src)
        shell = self._shell_installs(src)
        tree = src.tree
        if tree is None:
            return shell
        nodes = src.nodes
        aliases = A.import_aliases(nodes)
        assigns = A.latest_assignments(nodes)
        verifies_hashes = _uses_checksums(nodes, aliases)
        out = self._code_secrets(src, nodes)
        for node in nodes:
            if isinstance(node, ast.Call):
                f = (
                    self._sql(src, node, assigns)
                    or self._shell(src, node, aliases, assigns)
                    or self._deserialization(src, node, aliases)
                )
                if f:
                    out.append(f)
                out += [
                    f
                    for f in (
                        self._remote_code(src, node),
                        self._tls(src, node, aliases),
                        None if verifies_hashes else self._download(src, node, aliases),
                    )
                    if f
                ]
        return sorted(out + shell, key=lambda f: (f.location.line_start or 0, f.type))

    # -- insecure defaults (common in AI-written code)
    def _remote_code(self, src: SourceFile, call: ast.Call) -> Finding | None:
        if not A.is_true(A.kwarg(call, "trust_remote_code")):
            return None
        revision = A.const_str(A.kwarg(call, "revision")) or ""
        pinned = bool(re.fullmatch(r"[0-9a-f]{40}", revision))
        return make_finding(
            self,
            src,
            type="pipeline.supply_chain.trust_remote_code",
            title="Model/dataset loader runs code from the hub",
            severity=Severity.medium if pinned else Severity.high,
            confidence=0.9,
            line=call.lineno,
            rule_id="supply_chain.trust_remote_code",
            explanation="trust_remote_code=True executes Python shipped with the model or dataset"
            + (
                " (pinned to a commit)."
                if pinned
                else "; without a pinned revision, any later push runs too."
            ),
            fix="Prefer trust_remote_code=False; if the code is required, review it and pin revision=<commit sha>.",
        )

    def _tls(self, src: SourceFile, call: ast.Call, aliases: dict[str, str]) -> Finding | None:
        if not A.is_false(A.kwarg(call, "verify")):
            return None
        name = A.call_name(call, aliases) or ""
        if A.last_segment(call) not in HTTP_METHODS and not name.startswith(("requests.", "httpx.")):
            return None
        return make_finding(
            self,
            src,
            type="pipeline.transport.tls_verification_disabled",
            title="TLS certificate verification disabled",
            severity=Severity.medium,
            confidence=0.9,
            line=call.lineno,
            rule_id="transport.verify_false",
            explanation="verify=False accepts any certificate, so data or models fetched here can be "
            "swapped by anyone on the network path.",
            fix="Remove verify=False; for a private CA, pass verify='/path/to/ca-bundle.pem'.",
        )

    def _download(self, src: SourceFile, call: ast.Call, aliases: dict[str, str]) -> Finding | None:
        name = A.call_name(call, aliases) or ""
        if name not in DOWNLOADERS and name not in HTTP_GETS:
            return None
        url = A.const_str(A.arg(call, 0, "url"))
        if url is None:
            return None
        path = url.lower().split("?", 1)[0]
        if name in HTTP_GETS and not path.endswith(ARTIFACT_SUFFIXES):
            return None  # an API call, not a file download
        plain_http = path.startswith("http://")
        return make_finding(
            self,
            src,
            type="pipeline.supply_chain.unverified_download",
            title="Dataset/model downloaded without an integrity check",
            severity=Severity.high if plain_http else Severity.medium,
            confidence=0.75,
            line=call.lineno,
            rule_id="supply_chain.download_no_checksum",
            explanation="The file is fetched"
            + (" over plain HTTP" if plain_http else "")
            + " and nothing in this file verifies its hash, so a changed or tampered file flows "
            "straight into the pipeline.",
            fix="Pin the expected sha256 and compare it before use (hashlib.sha256(...).hexdigest()); use https.",
        )

    # -- secrets
    def _secret(self, src: SourceFile, line: int, rule: str, value: str, why: str, conf: float) -> Finding:
        return make_finding(
            self,
            src,
            type="pipeline.secret.hardcoded",
            title="Hardcoded credential",
            severity=Severity.high,
            confidence=conf,
            line=line,
            rule_id=f"secrets.{rule}",
            explanation=why,
            snippet=src.line(line).strip().replace(value, mask(value))[:200],
            fix="Move the value to an environment variable or secret manager, read it at runtime, "
            "and rotate the exposed credential.",
        )

    def _code_secrets(self, src: SourceFile, nodes: list[ast.AST]) -> list[Finding]:
        by_line: dict[int, Finding] = {}

        def add(line: int, rule: str, value: str, why: str, conf: float) -> None:
            if line not in by_line or by_line[line].confidence < conf:
                by_line[line] = self._secret(src, line, rule, value, why, conf)

        for node in nodes:
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for rule, value in _provider_hits(node.value):
                    add(node.lineno, rule, value, f"String literal matches the {rule} pattern.", 0.95)
            pairs: list[tuple[str, ast.expr]] = []
            if isinstance(node, ast.Assign):
                pairs = [(n, node.value) for t in node.targets for n in _bound_names(t)]
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                pairs = [(n, node.value) for n in _bound_names(node.target)]
            elif isinstance(node, ast.Call):
                pairs = [(a, kw.value) for kw in node.keywords if (a := kw.arg)]
            elif isinstance(node, ast.Dict):
                pairs = [(k, v) for key, v in zip(node.keys, node.values) if (k := A.const_str(key))]
            for name, value_node in pairs:
                literal = A.const_str(value_node)
                if literal and is_secret_name(name) and not is_placeholder(literal):
                    why = f"Literal value assigned to secret-like name {name!r}."
                    add(value_node.lineno, "named_literal", literal, why, 0.7)
        return list(by_line.values())

    def _config_secrets(self, src: SourceFile) -> list[Finding]:
        out = []
        for i, line in enumerate(src.lines, start=1):
            if line.lstrip().startswith(("#", ";", "//")):
                continue
            hits = _provider_hits(line)
            if hits:
                rule, value = hits[0]
                out.append(
                    self._secret(src, i, rule, value, f"Config value matches the {rule} pattern.", 0.95)
                )
                continue
            m = CONFIG_PAIR.match(line)
            if m and is_secret_name(m.group(1)) and not is_placeholder(m.group(2)):
                why = f"Config key {m.group(1)!r} holds a literal secret."
                out.append(self._secret(src, i, "config_literal", m.group(2), why, 0.75))
        return out

    # -- injection
    def _sql(
        self, src: SourceFile, call: ast.Call, assigns: dict[str, list[tuple[int, ast.expr]]]
    ) -> Finding | None:
        method = A.last_segment(call)
        if method not in SQL_METHODS:
            return None
        query = A.arg(call, 0, "sql", "query", "statement", "operation")
        if query is None:
            return None
        conf = 0.85
        if isinstance(query, ast.Name):
            query = A.value_before(assigns, query.id, call.lineno)
            conf = 0.75
        if query is None or not A.is_dynamic_str(query):
            return None
        if method in SQL_NEEDS_KEYWORD and not SQL_KEYWORDS.search(A.str_fragments(query)):
            return None
        return make_finding(
            self,
            src,
            type="pipeline.injection.sql",
            title="SQL built from runtime values",
            severity=Severity.high,
            confidence=conf,
            line=call.lineno,
            rule_id="injection.sql_dynamic_string",
            explanation=f"`{method}` receives a query assembled with string formatting; values that "
            "come from data or users can change the query.",
            fix="Use bound parameters (e.g. cursor.execute(sql, params) or sqlalchemy.text with "
            "bindparams) instead of formatting values into the SQL string.",
        )

    def _shell(
        self,
        src: SourceFile,
        call: ast.Call,
        aliases: dict[str, str],
        assigns: dict[str, list[tuple[int, ast.expr]]],
    ) -> Finding | None:
        name = A.call_name(call, aliases)
        if name not in SUBPROCESS and name not in ALWAYS_SHELL:
            return None
        if name in SUBPROCESS and not A.is_true(A.kwarg(call, "shell")):
            return None
        cmd = A.arg(call, 0, "args", "command", "cmd")
        if cmd is None or isinstance(cmd, (ast.List, ast.Tuple)):
            return None
        if isinstance(cmd, ast.Name):
            cmd = A.value_before(assigns, cmd.id, call.lineno) or cmd
        if A.const_str(cmd) is not None:
            return None
        return make_finding(
            self,
            src,
            type="pipeline.injection.shell",
            title="Shell command built from runtime values",
            severity=Severity.high,
            confidence=0.85 if A.is_dynamic_str(cmd) else 0.65,
            line=call.lineno,
            rule_id="injection.shell_dynamic_command",
            explanation=f"`{name}` runs a command through the shell with a non-constant string; file "
            "names or values from data can inject extra commands.",
            fix="Pass an argument list without shell=True (subprocess.run([...])), or quote values "
            "with shlex.quote.",
        )

    # -- deserialization
    def _deserialization(self, src: SourceFile, call: ast.Call, aliases: dict[str, str]) -> Finding | None:
        name = A.call_name(call, aliases)
        if name is None:
            return None
        sev: Severity | None = PICKLE_LIKE.get(name)
        if sev is not None:
            sev = _by_source(sev, _load_source(A.arg(call, 0, "file", "filepath_or_buffer"), aliases))
        why = f"`{name}` can execute arbitrary code embedded in the file it loads."
        fix = (
            "Load with a data-only format (JSON, Parquet, safetensors, ONNX), or verify a checksum "
            "or signature of the artifact before loading it."
        )
        if name in ("yaml.load", "yaml.load_all"):
            loader = A.arg(call, 1, "Loader")
            loader_name = A.resolve(A.dotted(loader) or "", aliases) if loader is not None else None
            if loader is not None and loader_name not in UNSAFE_YAML_LOADERS:
                return None
            sev = Severity.high
            why = "yaml.load without SafeLoader can construct arbitrary Python objects."
            fix = "Use yaml.safe_load (or Loader=yaml.SafeLoader)."
        elif name in ("yaml.unsafe_load", "yaml.unsafe_load_all"):
            sev, fix = Severity.high, "Use yaml.safe_load."
        elif name == "torch.load":
            if not A.is_false(A.kwarg(call, "weights_only")):
                return None
            sev, fix = Severity.high, "Use torch.load(..., weights_only=True) or safetensors."
        elif name == "numpy.load":
            if not A.is_true(A.kwarg(call, "allow_pickle")):
                return None
            sev = Severity.medium
        if sev is None:
            return None
        return make_finding(
            self,
            src,
            type="pipeline.deserialization.unsafe",
            title=f"Unsafe deserialization via {name}",
            severity=sev,
            confidence=0.9,
            line=call.lineno,
            rule_id=f"deserialization.{name}",
            explanation=why,
            fix=fix,
        )

    # -- dependencies
    def _shell_installs(self, src: SourceFile) -> list[Finding]:
        """`!pip install git+https://...` or a branch archive, `!git clone` with no commit checked out."""
        out: list[Finding] = []
        text = "\n".join(t for _, t in src.shell)
        pinned_checkout = bool(re.search(r"git\s+(checkout|reset\s+--hard)\s+[0-9a-f]{7,40}\b", text))
        for line, cmd in src.shell:
            why = None
            if re.search(r"\bpip3?\s+install\b", cmd):
                for url in re.findall(r"(git\+\S+|https?://\S+)", cmd):
                    if url.startswith("git+") and not re.search(r"@[\w.\-/]+(#.*)?$", url):
                        why = f"{url} installs the repository's default branch"
                    elif re.search(r"/(archive|tarball|zipball)/(refs/heads/)?(master|main|HEAD)\b", url):
                        why = f"{url} installs whatever the branch holds today"
            elif (
                re.search(r"\bgit\s+clone\b", cmd)
                and not re.search(r"(--branch|-b)\s+v?\d", cmd)
                and not pinned_checkout
            ):
                why = "git clone takes the default branch, and no commit is checked out"
            if why:
                out.append(
                    make_finding(
                        self,
                        src,
                        type="pipeline.dependency.unpinned",
                        title="Code installed from a moving branch",
                        severity=Severity.medium,
                        confidence=0.85,
                        line=line,
                        rule_id="dependency.unpinned_vcs",
                        explanation=f"{why}, so the code that runs changes whenever the branch does, without "
                        "review. A compromised or broken commit is picked up on the next run.",
                        snippet=cmd[:200],
                        fix="Pin a release or commit (git+https://...@<tag or sha>, or git checkout <sha>).",
                    )
                )
        return out

    def _requirements(self, src: SourceFile) -> list[Finding]:
        unpinned: list[tuple[int, str]] = []
        out: list[Finding] = []
        for i, raw in enumerate(src.lines, start=1):
            s = raw.split("#", 1)[0].strip()
            if not s or s.startswith(("-", ".", "/")) or "://" in s:
                continue
            m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]*\])?\s*(.*)$", s)
            if not m:
                continue
            if not re.search(r"===?|@", m.group(3)):
                unpinned.append((i, m.group(1)))
            right = WRONG_PACKAGE_NAMES.get(re.sub(r"[-_.]+", "-", m.group(1).lower()))
            if right:
                out.append(
                    make_finding(
                        self,
                        src,
                        type="pipeline.dependency.suspicious_package",
                        title=f"{m.group(1)!r} is not the package you want; use {right!r}",
                        severity=Severity.medium,
                        confidence=0.9,
                        line=i,
                        rule_id="dependency.wrong_name",
                        explanation=f"{m.group(1)!r} is an import name, not the PyPI distribution that provides "
                        f"it ({right!r}). Import-name and look-alike packages are a known target for "
                        "malicious uploads, and AI assistants often suggest them.",
                        metrics={"use_instead": right},
                        fix=f"Replace {m.group(1)!r} with {right!r} and pin its version.",
                    )
                )
        if not unpinned:
            return out
        names = ", ".join(n for _, n in unpinned[:10])
        return out + [
            make_finding(
                self,
                src,
                type="pipeline.dependency.unpinned",
                title=f"{len(unpinned)} unpinned dependencies",
                severity=Severity.low,
                confidence=0.95,
                line=unpinned[0][0],
                line_end=unpinned[-1][0],
                rule_id="dependency.unpinned",
                explanation=f"Not pinned to an exact version: {names}. A compromised or breaking "
                "release is picked up silently on the next install.",
                snippet=names,
                metrics={"unpinned": len(unpinned)},
                fix="Pin exact versions (pkg==x.y.z), ideally with hashes (pip-compile --generate-hashes).",
            )
        ]


def _bound_names(target: ast.expr) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Attribute):
        return [target.attr]
    if isinstance(target, ast.Subscript) and (key := A.const_str(target.slice)):
        return [key]
    return []


REMOTE_CALLS = frozenset(
    {"requests.get", "requests.post", "httpx.get", "urllib.request.urlopen", "urllib.request.urlretrieve"}
)
SEVERITY_LADDER = (Severity.info, Severity.low, Severity.medium, Severity.high, Severity.critical)


def _load_source(node: ast.expr | None, aliases: dict[str, str]) -> str:
    """Where loaded bytes come from: 'remote' (network), 'literal' (fixed local path), 'dynamic'."""
    if node is None:
        return "dynamic"
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and A.call_name(n, aliases) in REMOTE_CALLS:
            return "remote"
    path = A.const_str(node)
    if path is None and isinstance(node, ast.Call) and A.last_segment(node) == "open":
        path = A.const_str(A.arg(node, 0, "file"))
    if path is not None:
        return "remote" if path.lower().startswith(("http://", "https://", "s3://", "gs://")) else "literal"
    return "dynamic"


def _by_source(sev: Severity, source: str) -> Severity:
    """Loading from the network is worse; loading a fixed local artifact (often your own model) is milder."""
    i = SEVERITY_LADDER.index(sev)
    if source == "remote":
        return Severity.critical
    if source == "literal":
        return SEVERITY_LADDER[max(i - 1, 0)]
    return sev


HTTP_METHODS = frozenset({"get", "post", "put", "patch", "delete", "head", "options", "request", "stream"})
HTTP_GETS = frozenset({"requests.get", "httpx.get", "urllib.request.urlopen"})
DOWNLOADERS = frozenset({"urllib.request.urlretrieve", "wget.download", "gdown.download"})
ARTIFACT_SUFFIXES = (
    ".csv", ".tsv", ".parquet", ".json", ".jsonl", ".zip", ".gz", ".tgz", ".tar", ".bz2", ".xz",
    ".pkl", ".pickle", ".joblib", ".pt", ".pth", ".bin", ".h5", ".hdf5", ".onnx", ".safetensors",
    ".npy", ".npz", ".xlsx", ".feather",
)  # fmt: skip
CHECKSUM_CALLS = frozenset(
    {"hashlib.sha256", "hashlib.sha512", "hashlib.md5", "hashlib.sha1", "hashlib.new", "hashlib.file_digest"}
)
# Import name (or common mistake) -> the PyPI distribution that actually provides it.
WRONG_PACKAGE_NAMES = {
    "sklearn": "scikit-learn",
    "pil": "pillow",
    "cv2": "opencv-python",
    "skimage": "scikit-image",
    "yaml": "pyyaml",
    "dotenv": "python-dotenv",
}


def _uses_checksums(nodes: list[ast.AST], aliases: dict[str, str]) -> bool:
    return any(isinstance(n, ast.Call) and (A.call_name(n, aliases) or "") in CHECKSUM_CALLS for n in nodes)
