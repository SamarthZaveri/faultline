"""Finding and Check: the core contract. See docs/FINDING_SCHEMA.md.

Every detector returns Finding objects. Scorer, agents, API and dashboard consume only these.
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

TYPE_PATTERN = re.compile(r"^(data|pipeline)\.[a-z_]+\.[a-z_]+$")


def _now() -> datetime:
    return datetime.now(UTC)


class Severity(str, Enum):
    critical = "critical"
    high = "high"
    medium = "medium"
    low = "low"
    info = "info"


class Category(str, Enum):
    data = "data"
    pipeline = "pipeline"


class Environment(str, Enum):
    preprod = "preprod"
    prod = "prod"


class Complexity(str, Enum):
    tier1 = "tier1"
    tier2 = "tier2"
    unknown = "unknown"


class FindingStatus(str, Enum):
    detected = "detected"
    triaged = "triaged"
    fix_pr_open = "fix_pr_open"
    fixed = "fixed"
    escalated = "escalated"
    resolved = "resolved"
    false_positive = "false_positive"
    suppressed = "suppressed"


class Location(BaseModel):
    node_id: str | None = None
    file: str | None = None
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)
    dataset: str | None = None
    columns: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _has_some_anchor(self) -> Location:
        if not (self.node_id or self.file or self.dataset):
            raise ValueError("Location needs at least one of node_id, file, dataset")
        if self.line_start and self.line_end and self.line_end < self.line_start:
            raise ValueError("line_end must be >= line_start")
        return self


class Evidence(BaseModel):
    rule_id: str | None = None
    explanation: str
    metrics: dict[str, float | int | str] = Field(default_factory=dict)
    # MUST be redacted (hashed/masked). Never raw PII or secret values.
    samples: list[str] = Field(default_factory=list)
    code_snippet: str | None = None


class SuggestedFix(BaseModel):
    description: str
    diff: str | None = None  # unified diff
    check_spec: dict | None = None  # spec for a Great Expectations/Pandera check


class Finding(BaseModel):
    id: str | None = None
    type: str
    title: str
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    environment: Environment = Environment.preprod
    category: Category
    location: Location
    evidence: Evidence
    suggested_fix: SuggestedFix | None = None
    complexity: Complexity = Complexity.unknown
    status: FindingStatus = FindingStatus.detected
    detector: str  # name@version
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    refs: list[str] = Field(default_factory=list)

    @field_validator("type")
    @classmethod
    def _valid_type(cls, v: str) -> str:
        if not TYPE_PATTERN.match(v):
            raise ValueError(f"type must look like 'data.drift.covariate', got {v!r}")
        return v

    @model_validator(mode="after")
    def _derive(self) -> Finding:
        if self.type.split(".")[0] != self.category.value:
            raise ValueError("category must match the first segment of type")
        if self.id is None:
            self.id = self.stable_id()
        return self

    def stable_id(self) -> str:
        """Stable across re-scans so findings dedupe: hash of type + location + rule."""
        loc = self.location
        key = "|".join(
            str(x)
            for x in (
                self.type,
                loc.node_id,
                loc.file,
                loc.line_start,
                loc.dataset,
                ",".join(sorted(loc.columns)),
                self.evidence.rule_id,
            )
        )
        return hashlib.sha256(key.encode()).hexdigest()[:16]


class CheckKind(str, Enum):
    expectation = "expectation"
    pandera = "pandera"
    custom = "custom"


class CheckStatus(str, Enum):
    proposed = "proposed"
    approved = "approved"
    installed = "installed"
    failing = "failing"
    passing = "passing"
    disabled = "disabled"


class Check(BaseModel):
    id: str
    node_id: str
    kind: CheckKind
    spec: dict
    guards: list[str] = Field(default_factory=list)  # finding types this check protects against
    cost_estimate: float = Field(default=0.0, ge=0.0)
    status: CheckStatus = CheckStatus.proposed
    history: list[dict] = Field(default_factory=list)
