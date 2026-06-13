"""核心数据模型与类型定义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RiskLevel(str, Enum):
    """风险等级枚举。"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class ContractDocument:
    """合同文档。"""

    id: str
    title: str
    content: str
    file_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Clause:
    """合同条款识别结果。"""

    clause_type: str
    content: str
    start_pos: int | None = None
    end_pos: int | None = None
    risk: RiskLevel = RiskLevel.INFO
    comment: str = ""


@dataclass
class RiskAssessment:
    """单个条款的风险评估。"""

    clause: Clause
    risk_level: RiskLevel
    reason: str
    suggestion: str = ""


@dataclass
class ComplianceCheck:
    """合规检查结果。"""

    regulation: str
    status: bool
    detail: str = ""


@dataclass
class ReviewReport:
    """合同审查报告，包含条款、风险、合规检查等。"""

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
    """Agent 工具调用记录。"""

    tool_name: str
    input: dict[str, Any]
    output: Any = None
    started_at: str = ""
    finished_at: str = ""
    duration_ms: float = 0.0


@dataclass
class AgentStep:
    """Agent 单步执行记录。"""

    step_index: int
    agent_message: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    timestamp: str = ""


@dataclass
class AgentSession:
    """Agent 会话记录，包含文档、步骤与审查报告。"""

    session_id: str
    document: ContractDocument
    started_at: str
    finished_at: str | None = None
    steps: list[AgentStep] = field(default_factory=list)
    report: ReviewReport | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalMetric:
    """评测指标（名称、数值、权重）。"""

    name: str
    value: float
    weight: float = 1.0


@dataclass
class EvalResult:
    """单次评测的结果数据。"""

    dataset_name: str
    agent_version: str
    metrics: list[EvalMetric] = field(default_factory=list)
    detailed_results: list[dict[str, Any]] = field(default_factory=list)
    timestamp: str = ""


@dataclass
class RegressionResult:
    """回归测试对比结果。"""

    baseline_version: str
    current_version: str
    metrics_diff: dict[str, float] = field(default_factory=dict)
    regressions: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    passed: bool = True
