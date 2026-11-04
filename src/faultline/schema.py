"""The core contract. Every detector returns Finding; discovery returns PipelineGraph.

Scorer, check generators, agents and the dashboard consume only these types, so adding a
detector never requires touching anything downstream.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator

FINDING_TYPES: frozenset[str] = frozenset(
    {
        # data: poisoning
        "data.poisoning.label_flip",
        "data.poisoning.backdoor",
        "data.poisoning.clean_label",
        "data.poisoning.injected_rows",
        "data.poisoning.source_flooding",
        # data: drift and quality
        "data.drift.covariate",
        "data.drift.label_prior",
        "data.drift.concept",
        "data.drift.schema",
        "data.drift.new_category",
        "data.drift.missing_spike",
        # data: leakage (data seen by the model when it should not be)
        "data.leakage.preprocessing_before_split",
        "data.leakage.feature_selection_full",
        "data.leakage.test_in_training",
        "data.leakage.duplicate_across_split",
        "data.leakage.target_derived",
        "data.leakage.temporal_split",
        "data.leakage.pii_exposure",
        "data.leakage.resampling_before_split",
        "data.leakage.tuning_on_test",
        # pipeline vulnerabilities
        "pipeline.secret.hardcoded",
        "pipeline.injection.sql",
        "pipeline.injection.shell",
        "pipeline.deserialization.unsafe",
        "pipeline.dependency.unpinned",
        "pipeline.dependency.vulnerable",
        "pipeline.dependency.suspicious_package",
        "pipeline.supply_chain.unverified_download",
        "pipeline.supply_chain.trust_remote_code",
        "pipeline.transport.tls_verification_disabled",
        "pipeline.config.public_bucket",
        "pipeline.config.overbroad_iam",
        "pipeline.boundary.unvalidated",
    }
)


def _now() -> datetime:
    return datetime.now(UTC)


class Severity(StrEnum):
    critical = "critical"
    high = "high"
    medium = "medium"
    low = "low"
    info = "info"

    @property
    def rank(self) -> int:
        return list(Severity).index(self)  # 0 = most severe


class Category(StrEnum):
    data = "data"
    pipeline = "pipeline"


class Environment(StrEnum):
    preprod = "preprod"
    prod = "prod"


class Tier(StrEnum):
    tier1 = "tier1"  # simple, localized: agents rewrite the code
    tier2 = "tier2"  # needs judgment: agents escalate
    unknown = "unknown"


class Status(StrEnum):
    detected = "detected"
    triaged = "triaged"
    fix_proposed = "fix_proposed"
    fixed = "fixed"
    escalated = "escalated"
    false_positive = "false_positive"
    suppressed = "suppressed"


class Location(BaseModel):
    file: str
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)
    cell: int | None = Field(default=None, ge=0)  # notebook code-cell index
    node_id: str | None = None
    columns: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ordered(self) -> Location:
        if self.line_start and self.line_end and self.line_end < self.line_start:
            raise ValueError("line_end must be >= line_start")
        return self


class Evidence(BaseModel):
    rule_id: str
    explanation: str
    snippet: str | None = None  # secrets are masked before they get here
    metrics: dict[str, float | int | str] = Field(default_factory=dict)


class Finding(BaseModel):
    id: str = ""
    type: str
    title: str
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    location: Location
    evidence: Evidence
    detector: str  # name@version
    suggested_fix: str | None = None
    tier: Tier = Tier.unknown
    status: Status = Status.detected
    environment: Environment = Environment.preprod
    created_at: datetime = Field(default_factory=_now)

    @field_validator("type")
    @classmethod
    def _known_type(cls, v: str) -> str:
        if v not in FINDING_TYPES:
            raise ValueError(f"unknown finding type {v!r}; add it to FINDING_TYPES first")
        return v

    @computed_field  # type: ignore[prop-decorator]
    @property
    def category(self) -> Category:
        return Category(self.type.split(".", 1)[0])

    @model_validator(mode="after")
    def _stable_id(self) -> Finding:
        # Stable across re-scans so findings dedupe.
        if not self.id:
            loc = self.location
            key = f"{self.type}|{loc.file}|{loc.line_start}|{loc.cell}|{self.evidence.rule_id}"
            if loc.columns:  # data findings: same dataset and rule, one finding per column
                key += "|" + ",".join(loc.columns)
            self.id = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self


class NodeKind(StrEnum):
    dataset = "dataset"
    source = "source"  # code that pulls data in from outside
    transform = "transform"
    split_step = "split"  # code that splits train/test (member name avoids str.split)
    train = "train"  # code that fits a model
    config = "config"  # dependency or config file


class Node(BaseModel):
    id: str
    kind: NodeKind
    label: str
    file: str | None = None
    external: bool = False  # dataset outside the pipeline's trust boundary (URL, bucket, API)


class Edge(BaseModel):
    src: str
    dst: str
    trust_boundary: bool = False  # data crosses from external into the pipeline


class PipelineGraph(BaseModel):
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
