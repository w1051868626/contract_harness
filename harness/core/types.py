from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RiskLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class ContractDocument:
    id: str
    title: str
    content: str
    file_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Clause:
    clause_type: str
    content: str
    start_pos: int | None = None
    end_pos: int | None = None
    risk: RiskLevel = RiskLevel.INFO
    comment: str = ""


@dataclass
class RiskAssessment:
    clause: Clause
    risk_level: RiskLevel
    reason: str
    suggestion: str = ""


@dataclass
class ComplianceCheck:
    regulation: str
    status: bool
    detail: str = ""


@dataclass
class ReviewReport:
    document_id: str
    document_title: str
    reviewed_at: str
    summary: str
    clauses: list[Clause] = field(default_factory=list)
    risks: list[RiskAssessment] = field(default_factory=list)
    compliance_checks: list[ComplianceCheck] = field(default_factory=list)
    overall_risk: RiskLevel = RiskLevel.INFO
    raw_output: str = ""
    agent_version: str = ""


@dataclass
class ToolCall:
    tool_name: str
    input: dict[str, Any]
    output: Any = None
    started_at: str = ""
    finished_at: str = ""
    duration_ms: float = 0.0


@dataclass
class AgentStep:
    step_index: int
    agent_message: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    timestamp: str = ""


@dataclass
class AgentSession:
    session_id: str
    document: ContractDocument
    started_at: str
    finished_at: str | None = None
    steps: list[AgentStep] = field(default_factory=list)
    report: ReviewReport | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalMetric:
    name: str
    value: float
    weight: float = 1.0


@dataclass
class EvalResult:
    dataset_name: str
    agent_version: str
    metrics: list[EvalMetric] = field(default_factory=list)
    detailed_results: list[dict[str, Any]] = field(default_factory=list)
    timestamp: str = ""


@dataclass
class RegressionResult:
    baseline_version: str
    current_version: str
    metrics_diff: dict[str, float] = field(default_factory=dict)
    regressions: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    passed: bool = True
