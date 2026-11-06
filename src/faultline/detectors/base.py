from __future__ import annotations

from typing import Protocol

from faultline.discovery.sources import SourceFile
from faultline.schema import Evidence, Finding, Location, Severity, Tier


class Detector(Protocol):
    name: str
    version: str

    def detect(self, src: SourceFile) -> list[Finding]: ...


def file_node_id(path: str) -> str:
    return f"file:{path}"


def make_finding(
    det: Detector,
    src: SourceFile,
    *,
    type: str,
    title: str,
    severity: Severity,
    confidence: float,
    line: int,
    rule_id: str,
    explanation: str,
    snippet: str | None = None,
    fix: str | None = None,
    line_end: int | None = None,
    metrics: dict[str, float | int | str] | None = None,
    tier: Tier = Tier.unknown,
) -> Finding:
    if snippet is None:
        snippet = src.line(line).strip()[:200] or None
    return Finding(
        type=type,
        title=title,
        severity=severity,
        confidence=confidence,
        location=Location(
            file=src.path,
            line_start=line,
            line_end=line_end or line,
            cell=src.cell_of(line),
            node_id=file_node_id(src.path),
        ),
        evidence=Evidence(rule_id=rule_id, explanation=explanation, snippet=snippet, metrics=metrics or {}),
        detector=f"{det.name}@{det.version}",
        suggested_fix=fix,
        tier=tier,
    )
