"""Scan a pipeline repo: discover the graph and run every detector."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from faultline.detectors import run_detectors
from faultline.discovery.graph import build_graph
from faultline.discovery.sources import iter_sources
from faultline.schema import Finding, PipelineGraph


class ScanResult(BaseModel):
    root: str
    files_scanned: int
    parse_errors: list[str]
    graph: PipelineGraph
    findings: list[Finding]


def scan(root: Path) -> ScanResult:
    sources = list(iter_sources(root))
    findings = [f for src in sources for f in run_detectors(src)]
    findings.sort(key=lambda f: (f.severity.rank, f.location.file, f.location.line_start or 0))
    return ScanResult(
        root=str(root),
        files_scanned=len(sources),
        parse_errors=[s.path for s in sources if s.is_code and s.tree is None],
        graph=build_graph(sources),
        findings=findings,
    )
