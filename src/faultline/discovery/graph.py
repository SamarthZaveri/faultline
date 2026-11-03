"""Build the pipeline graph statically: code files and the datasets they read and write.

A file node's kind is its most downstream role (train > split > source > transform). Dataset
nodes come from literal paths/URLs passed to readers and writers; reads from outside the repo
(URLs, buckets, databases, HTTP APIs) are marked as crossing the trust boundary.
"""

from __future__ import annotations

import ast

from faultline.detectors import _ast as A
from faultline.detectors.base import file_node_id
from faultline.detectors.leakage import CV_FUNCS, PREPROCESSORS, SELECTORS
from faultline.discovery.sources import SourceFile
from faultline.schema import Edge, Node, NodeKind, PipelineGraph

READERS = frozenset(
    {
        "read_csv", "read_parquet", "read_json", "read_excel", "read_table", "read_feather",
        "read_orc", "read_sql", "read_sql_query", "read_sql_table", "read_pickle", "read_hdf",
        "load_dataset", "loadtxt", "genfromtxt", "urlretrieve",
    }
)  # fmt: skip
WRITERS = frozenset(
    {"to_csv", "to_parquet", "to_json", "to_excel", "to_feather", "to_pickle", "to_sql", "to_hdf", "savetxt"}
)
HTTP_GETTERS = frozenset({"requests.get", "requests.post", "httpx.get", "urllib.request.urlopen"})
SQL_READERS = frozenset({"read_sql", "read_sql_query", "read_sql_table"})
EXTERNAL_PREFIXES = (
    "http://",
    "https://",
    "s3://",
    "gs://",
    "gcs://",
    "az://",
    "abfs://",
    "hdfs://",
    "ftp://",
)
MODEL_MODULES = ("sklearn.", "xgboost", "lightgbm", "catboost", "torch.nn", "tensorflow.keras", "keras")
NOT_MODELS = PREPROCESSORS | SELECTORS | {"ColumnTransformer", "make_column_transformer", "FeatureUnion"}
ROLE_ORDER = (NodeKind.transform, NodeKind.source, NodeKind.split_step, NodeKind.train)


def _dataset_arg(call: ast.Call) -> str | None:
    return A.const_str(A.arg(call, 0, "path", "filepath_or_buffer", "path_or_buf", "url", "fname", "name"))


def _is_external(ref: str) -> bool:
    return ref.lower().startswith(EXTERNAL_PREFIXES) or ref.startswith(("sql:", "http:"))


def _is_model_ctor(call: ast.Call, aliases: dict[str, str]) -> bool:
    name = A.call_name(call, aliases) or ""
    return name.startswith(MODEL_MODULES) and A.last_segment(call) not in NOT_MODELS


def _promote(role: NodeKind, to: NodeKind) -> NodeKind:
    return max(role, to, key=ROLE_ORDER.index)


def _file_role_and_io(tree: ast.Module) -> tuple[NodeKind, list[str], list[str]]:
    aliases = A.import_aliases(tree)
    model_vars = {
        n.targets[0].id
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and isinstance(n.value, ast.Call)
        and _is_model_ctor(n.value, aliases)
    }
    reads: list[str] = []
    writes: list[str] = []
    role = NodeKind.transform
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        seg = A.last_segment(node)
        if seg in READERS:
            ref = _dataset_arg(node) or ("sql:query" if seg in SQL_READERS else None)
            if ref:
                reads.append(ref)
        elif A.call_name(node, aliases) in HTTP_GETTERS:
            reads.append(A.const_str(A.arg(node, 0, "url")) or "http:dynamic")
        elif seg in WRITERS and (ref := _dataset_arg(node)):
            writes.append(ref)
        if seg == "train_test_split" or seg in CV_FUNCS:
            role = _promote(role, NodeKind.split_step)
        elif seg == "fit" and isinstance(node.func, ast.Attribute):
            receiver = node.func.value
            if (isinstance(receiver, ast.Call) and _is_model_ctor(receiver, aliases)) or (
                isinstance(receiver, ast.Name) and receiver.id in model_vars
            ):
                role = NodeKind.train
    if role is NodeKind.transform and any(_is_external(r) for r in reads):
        role = NodeKind.source
    return role, reads, writes


def build_graph(sources: list[SourceFile]) -> PipelineGraph:
    nodes: dict[str, Node] = {}
    edges: dict[tuple[str, str], Edge] = {}
    for src in sources:
        fid = file_node_id(src.path)
        if not src.is_code:
            nodes[fid] = Node(id=fid, kind=NodeKind.config, label=src.path, file=src.path)
            continue
        if src.tree is None:
            continue
        role, reads, writes = _file_role_and_io(src.tree)
        nodes[fid] = Node(id=fid, kind=role, label=src.path, file=src.path)
        for ref in reads:
            did = f"data:{ref}"
            external = _is_external(ref)
            nodes.setdefault(did, Node(id=did, kind=NodeKind.dataset, label=ref, external=external))
            edges[(did, fid)] = Edge(src=did, dst=fid, trust_boundary=external)
        for ref in writes:
            did = f"data:{ref}"
            nodes.setdefault(did, Node(id=did, kind=NodeKind.dataset, label=ref, external=_is_external(ref)))
            edges[(fid, did)] = Edge(src=fid, dst=did)
    return PipelineGraph(nodes=list(nodes.values()), edges=list(edges.values()))
