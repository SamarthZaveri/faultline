from pathlib import Path

from faultline.discovery.graph import build_graph
from faultline.discovery.sources import iter_sources
from faultline.schema import NodeKind


def test_tiny_repo_lineage(tiny_repo: Path) -> None:
    g = build_graph(list(iter_sources(tiny_repo)))
    kinds = {n.id: n.kind for n in g.nodes}
    assert kinds["file:pipeline/ingest.py"] is NodeKind.source
    assert kinds["file:pipeline/clean.py"] is NodeKind.transform
    assert kinds["file:pipeline/train.py"] is NodeKind.train
    assert kinds["file:explore.ipynb"] is NodeKind.split_step
    assert kinds["data:https://example.com/orders.csv"] is NodeKind.dataset

    edges = {(e.src, e.dst): e.trust_boundary for e in g.edges}
    assert edges[("data:https://example.com/orders.csv", "file:pipeline/ingest.py")] is True
    assert edges[("file:pipeline/ingest.py", "data:data/raw_orders.parquet")] is False
    assert ("data:data/raw_orders.parquet", "file:pipeline/clean.py") in edges
    assert ("data:data/clean_orders.parquet", "file:pipeline/train.py") in edges
    assert ("data:data/clean_orders.parquet", "file:explore.ipynb") in edges
