# Multi-Agent 协同审查实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 `MULTI_AGENT` 模式，使用 Supervisor + Worker 多 Agent 协同完成合同审查，支持专业分工与交叉验证。

**Architecture:** `MultiAgentCoordinator` 作为第四种 Agent 运行模式，内含 `SupervisorAgent`（任务分配+分歧检测+报告合成）、3 个领域 `WorkerAgent`（ClauseExpert/RiskExpert/ComplianceExpert）、`CrossValidator`（规则优先+LLM 兜底仲裁）。

**Tech Stack:** Python 3.11+, LLMClient, pytest, MockLLMClient

---

## Task 1: 新增数据模型

**Files:**
- Modify: `harness/core/types.py` — 新增 WorkerTask/WorkerOutput/Disagreement，AgentMode 追加 MULTI_AGENT

**Interfaces:**
- Consumes: 现有 Clause/RiskAssessment/ComplianceCheck/ReviewReport/AgentMode
- Produces: `WorkerTask`, `WorkerOutput`, `Disagreement`, `AgentMode.MULTI_AGENT`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_multi_agent.py
"""Multi-Agent 数据模型测试。"""

from __future__ import annotations

from harness.core.types import (
    AgentMode,
    Disagreement,
    WorkerOutput,
    WorkerTask,
)


class TestDataModels:
    """Multi-Agent 新增数据模型测试。"""

    def test_agent_mode_has_multi_agent(self):
        assert AgentMode.MULTI_AGENT == "multi_agent"

    def test_worker_task_creation(self):
        task = WorkerTask(worker_role="ClauseExpert", prompt="提取条款", input_data={})
        assert task.worker_role == "ClauseExpert"
        assert task.prompt == "提取条款"

    def test_worker_output_creation(self):
        output = WorkerOutput(worker_role="RiskExpert", content="高风险", structured={"level": "high"})
        assert output.worker_role == "RiskExpert"
        assert output.structured["level"] == "high"

    def test_disagreement_creation(self):
        d = Disagreement(
            item_id="clause-0",
            field="risk_level",
            value_a="high",
            value_b="low",
            worker_a="RiskExpert",
            worker_b="ComplianceExpert",
        )
        assert d.field == "risk_level"
        assert d.worker_a == "RiskExpert"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_multi_agent.py::TestDataModels -v`
Expected: FAIL (types not yet defined)

- [ ] **Step 3: Add data models to types.py**

```python
# In harness/core/types.py, add after existing dataclasses:

@dataclass
class WorkerTask:
    """分派给 Worker Agent 的任务。"""
    worker_role: str
    prompt: str
    input_data: Any
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkerOutput:
    """Worker Agent 的执行输出。"""
    worker_role: str
    content: str
    structured: Any = None
    confidence: float = 1.0


@dataclass
class Disagreement:
    """两个 Worker 之间的分歧记录。"""
    item_id: str
    field: str
    value_a: Any
    value_b: Any
    worker_a: str
    worker_b: str
```

Update `AgentMode`:

```python
class AgentMode(str, Enum):
    PIPELINE = "pipeline"
    REACT = "react"
    REFLECTION = "reflection"
    MULTI_AGENT = "multi_agent"  # 新增
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_multi_agent.py::TestDataModels -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_multi_agent.py harness/core/types.py
git commit -m "feat: add multi-agent data models and AgentMode.MULTI_AGENT"
```

---

## Task 2: WorkerAgent 实现

**Files:**
- Create: `harness/agent/multi_agent/__init__.py`
- Create: `harness/agent/multi_agent/worker.py`

**Interfaces:**
- Consumes: `LLMClient`, `WorkerTask`, `WorkerOutput`, `ClauseExtractor`, `RiskAnalyzer`, `ComplianceChecker`, `KnowledgeRetriever`
- Produces: `WorkerAgent.execute(task, peer_results)` → `WorkerOutput`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_multi_agent.py (append)
import json
from harness.agent.llm import LLMResponse
from harness.agent.multi_agent.worker import WorkerAgent, WORKER_PROMPTS
from tests.conftest import MockLLMClient


class TestWorkerAgent:
    """WorkerAgent 单元测试。"""

    def test_worker_has_role_prompt(self):
        assert "ClauseExpert" in WORKER_PROMPTS
        assert "RiskExpert" in WORKER_PROMPTS
        assert "ComplianceExpert" in WORKER_PROMPTS

    def test_worker_execute_returns_output(self):
        llm = MockLLMClient([LLMResponse(content='{"clauses": []}', model="mock")])
        worker = WorkerAgent(role="ClauseExpert", llm=llm)
        output = worker.execute("测试合同内容")
        assert output.worker_role == "ClauseExpert"
        assert output.content

    def test_worker_calls_llm_with_correct_messages(self):
        llm = MockLLMClient([LLMResponse(content="ok", model="mock")])
        worker = WorkerAgent(role="RiskExpert", llm=llm)
        worker.execute("审核内容")
        assert len(llm.calls) == 1
        messages = llm.calls[0]["messages"]
        # should have system prompt + user message
        assert any(m["role"] == "system" for m in messages)
        assert any("审核" in m.get("content", "") for m in messages if m["role"] == "user")

    def test_worker_with_peer_results(self):
        llm = MockLLMClient([LLMResponse(content="交叉验证完成", model="mock")])
        worker = WorkerAgent(role="ComplianceExpert", llm=llm)
        peer = {"RiskExpert": "高风险发现"}
        output = worker.execute("审核内容", peer_results=peer)
        assert output.content
        # peer_results should appear in messages
        all_text = " ".join(m.get("content", "") for m in llm.calls[0]["messages"])
        assert "高风险发现" in all_text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_multi_agent.py::TestWorkerAgent -v`
Expected: FAIL (worker.py not yet created)

- [ ] **Step 3: Implement WorkerAgent**

```python
# harness/agent/multi_agent/__init__.py
"""Multi-Agent 协同审查包。"""


# harness/agent/multi_agent/worker.py
"""领域专业 Worker Agent，独立 LLM + 专用 System Prompt。"""

from __future__ import annotations

from typing import Any

from harness.agent.llm import LLMClient
from harness.core.types import WorkerOutput

WORKER_PROMPTS: dict[str, str] = {
    "ClauseExpert": (
        "你是一位专业的合同条款提取专家。你的职责是从合同中准确提取所有关键条款，"
        "包括条款类型、具体内容。请输出结构化 JSON。"
    ),
    "RiskExpert": (
        "你是一位专业的法律风险评估专家。你的职责是对已提取的合同条款进行风险分析，"
        "评估风险等级（critical/high/medium/low/info）并提供理由和建议。请输出结构化 JSON。"
    ),
    "ComplianceExpert": (
        "你是一位专业的合规审查专家。你的职责是检查合同条款是否符合中国法律法规，"
        "包括民法典、劳动合同法、数据安全法、个人信息保护法、反垄断法等。请输出结构化 JSON。"
    ),
}


class WorkerAgent:
    """领域专业 Worker Agent，可独立调用 LLM 执行任务。"""

    def __init__(self, role: str, llm: LLMClient):
        if role not in WORKER_PROMPTS:
            raise ValueError(f"未知 Worker 角色: {role}，可用: {list(WORKER_PROMPTS.keys())}")
        self.role = role
        self._llm = llm

    def execute(
        self,
        content: str,
        peer_results: dict[str, Any] | None = None,
    ) -> WorkerOutput:
        """执行任务，可选传入同行结果供交叉验证。"""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": WORKER_PROMPTS[self.role]},
        ]
        if peer_results:
            peer_text = "\n".join(f"## {k} 的分析\n{v}" for k, v in peer_results.items())
            messages.append({
                "role": "user",
                "content": f"请审阅以下其他专家的分析意见，检查是否存在分歧或遗漏：\n\n{peer_text}",
            })
        messages.append({"role": "user", "content": content})

        resp = self._llm.chat(messages)
        return WorkerOutput(
            worker_role=self.role,
            content=resp.content,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_multi_agent.py::TestWorkerAgent -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add harness/agent/multi_agent/ tests/unit/test_multi_agent.py
git commit -m "feat: add WorkerAgent with domain-specific prompts"
```

---

## Task 3: CrossValidator 实现

**Files:**
- Create: `harness/agent/multi_agent/validator.py`

**Interfaces:**
- Consumes: `Disagreement`, `LLMClient`, `ReviewReport`
- Produces: `CrossValidator.arbitrate(disagreements, report)` → `list[dict]`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_multi_agent.py (append)
from harness.agent.llm import LLMResponse
from harness.agent.multi_agent.validator import CrossValidator
from harness.core.types import (
    Clause, ComplianceCheck, Disagreement, ReviewReport, RiskAssessment, RiskLevel,
)
from tests.conftest import MockLLMClient


class TestCrossValidator:
    """CrossValidator 分歧仲裁测试。"""

    def test_same_level_no_disagreement(self):
        validator = CrossValidator()
        disc: list[Disagreement] = []
        result = validator.arbitrate(disc)
        assert len(result) == 0

    def test_one_level_diff_uses_stricter(self):
        """风险等级差 1 级时取更严格侧。"""
        d = Disagreement(
            item_id="clause-0", field="risk_level",
            value_a="medium", value_b="low",
            worker_a="A", worker_b="B",
        )
        validator = CrossValidator()
        result = validator.arbitrate([d])
        assert result[0]["resolved"] == "medium"  # 更严格

    def test_two_level_diff_triggers_llm(self):
        """风险等级差 ≥2 级时需 LLM 仲裁。"""
        d = Disagreement(
            item_id="clause-0", field="risk_level",
            value_a="critical", value_b="low",
            worker_a="A", worker_b="B",
        )
        llm = MockLLMClient([LLMResponse(
            content='{"resolution": "critical", "explanation": "条款涉及核心义务"}',
            model="mock",
        )])
        validator = CrossValidator(llm=llm)
        result = validator.arbitrate([d])
        assert result[0]["resolved"] == "critical"

    def test_compliance_status_disagreement(self):
        """合规状态分歧标记为待人工审核。"""
        d = Disagreement(
            item_id="clause-0", field="compliance_status",
            value_a=True, value_b=False,
            worker_a="A", worker_b="B",
        )
        validator = CrossValidator()
        result = validator.arbitrate([d])
        assert result[0]["needs_human_review"] is True

    def test_llm_fallback_keeps_majority(self):
        """LLM 调用失败时用多数决保留。"""
        d1 = Disagreement(
            item_id="c0", field="risk_level",
            value_a="medium", value_b="high",
            worker_a="A", worker_b="B",
        )
        d2 = Disagreement(
            item_id="c1", field="clause_type",
            value_a="保密", value_b="违约责任",
            worker_a="A", worker_b="B",
        )
        validator = CrossValidator()
        result = validator.arbitrate([d1, d2])
        # clause_type 分歧 → 都保留
        assert "c1" in [r["item_id"] for r in result]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_multi_agent.py::TestCrossValidator -v`
Expected: FAIL (validator.py not yet created)

- [ ] **Step 3: Implement CrossValidator**

```python
# harness/agent/multi_agent/validator.py
"""分歧仲裁器：规则优先，LLM 兜底。"""

from __future__ import annotations

from typing import Any

from harness.agent.llm import LLMClient
from harness.core.types import Disagreement
from harness.utils.log import logger

RISK_RANK = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}


class CrossValidator:
    """分歧仲裁：按分歧类型分层处理。"""

    def __init__(self, llm: LLMClient | None = None):
        self._llm = llm

    def arbitrate(
        self,
        disagreements: list[Disagreement],
    ) -> list[dict[str, Any]]:
        """对分歧列表逐条仲裁，返回仲裁结果。"""
        results: list[dict[str, Any]] = []
        for d in disagreements:
            result = self._arbitrate_one(d)
            results.append(result)
        return results

    def _arbitrate_one(self, d: Disagreement) -> dict[str, Any]:
        """仲裁单条分歧。"""
        if d.field == "risk_level":
            return self._arbitrate_risk(d)
        if d.field == "compliance_status":
            return {
                "item_id": d.item_id,
                "field": d.field,
                "resolved": d.value_a,
                "needs_human_review": True,
                "explanation": "合规状态分歧，需人工审核确认",
            }
        # 其他分歧（条款类型等）：都保留
        return {
            "item_id": d.item_id,
            "field": d.field,
            "resolved": f"{d.value_a} / {d.value_b}",
            "needs_review": True,
            "explanation": f"分歧: {d.worker_a}={d.value_a}, {d.worker_b}={d.value_b}",
        }

    def _arbitrate_risk(self, d: Disagreement) -> dict[str, Any]:
        """仲裁风险等级分歧。"""
        rank_a = RISK_RANK.get(str(d.value_a), 0)
        rank_b = RISK_RANK.get(str(d.value_b), 0)
        diff = abs(rank_a - rank_b)

        if diff <= 1:
            # 差 1 级以内，取更严格侧
            resolved = d.value_a if rank_a >= rank_b else d.value_b
            return {
                "item_id": d.item_id,
                "field": "risk_level",
                "resolved": resolved,
                "explanation": f"等级差 {diff} 级，取更严格侧: {resolved}",
            }

        # 差 ≥2 级，需要 LLM 仲裁
        return self._llm_arbitrate_risk(d)

    def _llm_arbitrate_risk(self, d: Disagreement) -> dict[str, Any]:
        """用 LLM 仲裁严重分歧的风险等级。"""
        if not self._llm:
            # 无 LLM 时取更严格侧
            rank_a = RISK_RANK.get(str(d.value_a), 0)
            rank_b = RISK_RANK.get(str(d.value_b), 0)
            resolved = d.value_a if rank_a >= rank_b else d.value_b
            return {
                "item_id": d.item_id,
                "field": "risk_level",
                "resolved": resolved,
                "explanation": "LLM 不可用，取更严格侧",
            }

        prompt = (
            f"两位专家对同一条款的风险等级判断出现严重分歧：\n"
            f"- {d.worker_a} 认为: {d.value_a}\n"
            f"- {d.worker_b} 认为: {d.value_b}\n\n"
            f"请根据风险管理原则，裁定最终风险等级。"
            f"输出 JSON: {{\"resolution\": \"等级\", \"explanation\": \"原因\"}}"
        )
        try:
            resp = self._llm.chat([
                {"role": "system", "content": "你是一位风险管理仲裁专家。"},
                {"role": "user", "content": prompt},
            ])
            import json
            data = json.loads(resp.content.strip().removeprefix("```json").removesuffix("```").strip())
            resolved = data.get("resolution", d.value_a)
            return {
                "item_id": d.item_id,
                "field": "risk_level",
                "resolved": resolved,
                "explanation": data.get("explanation", "LLM 仲裁"),
            }
        except Exception as e:
            logger.warning("LLM 风险仲裁失败: {}", e)
            rank_a = RISK_RANK.get(str(d.value_a), 0)
            rank_b = RISK_RANK.get(str(d.value_b), 0)
            resolved = d.value_a if rank_a >= rank_b else d.value_b
            return {
                "item_id": d.item_id,
                "field": "risk_level",
                "resolved": resolved,
                "explanation": f"LLM 仲裁失败，取更严格侧: {resolved}",
            }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_multi_agent.py::TestCrossValidator -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add harness/agent/multi_agent/validator.py tests/unit/test_multi_agent.py
git commit -m "feat: add CrossValidator with rule-based + LLM arbitration"
```

---

## Task 4: SupervisorAgent 实现

**Files:**
- Create: `harness/agent/multi_agent/supervisor.py`

**Interfaces:**
- Consumes: `ContractDocument`, `WorkerOutput`, `Disagreement`, `Clause`, `RiskAssessment`, `ComplianceCheck`, `ReviewReport`
- Produces: `SupervisorAgent.assign_tasks(document)` → `dict[str, WorkerTask]`?, `SupervisorAgent.validate_consensus(outputs)` → `list[Disagreement]`, `SupervisorAgent.synthesize_report(outputs, arbitration_results)` → `ReviewReport`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_multi_agent.py (append)
from harness.agent.llm import LLMResponse
from harness.agent.multi_agent.supervisor import SupervisorAgent
from harness.agent.multi_agent.worker import WorkerOutput
from harness.core.types import (
    Clause, ComplianceCheck, ContractDocument, ReviewReport, RiskAssessment, RiskLevel,
)
from tests.conftest import MockLLMClient


SAMPLE_CLAUSES = [
    {"type": "保密", "content": "双方应保守商业秘密", "risk": "low"},
    {"type": "违约责任", "content": "违约方应赔偿损失", "risk": "medium"},
]


class TestSupervisorAgent:
    """SupervisorAgent 单元测试。"""

    def test_validate_consensus_no_disagreement(self):
        """相同输出不应产生分歧。"""
        outputs = {
            "RiskExpert": WorkerOutput(
                worker_role="RiskExpert",
                content="风险分析完成",
                structured=[
                    {"clause_type": "保密", "risk_level": "low"},
                    {"clause_type": "违约责任", "risk_level": "medium"},
                ],
            ),
            "ComplianceExpert": WorkerOutput(
                worker_role="ComplianceExpert",
                content="合规检查完成",
                structured=[
                    {"clause_index": 0, "status": True},
                    {"clause_index": 1, "status": True},
                ],
            ),
        }
        supervisor = SupervisorAgent()
        disagreements = supervisor.validate_consensus(outputs)
        assert len(disagreements) == 0

    def test_validate_consensus_finds_disagreement(self):
        """不同风险等级应被检测为分歧。"""
        outputs = {
            "RiskExpert": WorkerOutput(
                worker_role="RiskExpert",
                content="风险分析完成",
                structured=[
                    {"clause_type": "保密", "risk_level": "low"},
                ],
            ),
            "ComplianceExpert": WorkerOutput(
                worker_role="ComplianceExpert",
                content="合规检查完成",
                structured=[
                    {"clause_index": 0, "status": True, "risk_level_note": "high"},
                ],
            ),
        }
        supervisor = SupervisorAgent()
        disagreements = supervisor.validate_consensus(outputs)
        assert len(disagreements) > 0

    def test_synthesize_report_returns_report(self, sample_document):
        """合成报告应返回完整的 ReviewReport。"""
        outputs = {
            "ClauseExpert": WorkerOutput(
                worker_role="ClauseExpert",
                content="条款提取完成",
                structured=SAMPLE_CLAUSES,
            ),
            "RiskExpert": WorkerOutput(
                worker_role="RiskExpert",
                content="风险分析完成",
                structured=[
                    {"clause_type": "保密", "risk_level": "low", "reason": "标准条款"},
                    {"clause_type": "违约责任", "risk_level": "medium", "reason": "赔偿范围模糊"},
                ],
            ),
            "ComplianceExpert": WorkerOutput(
                worker_role="ComplianceExpert",
                content="合规检查完成",
                structured=[],
            ),
        }
        supervisor = SupervisorAgent()
        report = supervisor.synthesize_report(sample_document, outputs, [])
        assert isinstance(report, ReviewReport)
        assert report.document_id == sample_document.id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_multi_agent.py::TestSupervisorAgent -v`
Expected: FAIL (supervisor.py not yet created)

- [ ] **Step 3: Implement SupervisorAgent**

```python
# harness/agent/multi_agent/supervisor.py
"""Supervisor Agent：任务分配 + 分歧检测 + 报告合成。"""

from __future__ import annotations

from typing import Any

from harness.core.types import (
    Clause,
    ComplianceCheck,
    ContractDocument,
    Disagreement,
    ReviewReport,
    RiskAssessment,
    RiskLevel,
    WorkerOutput,
)
from harness.utils.log import logger


class SupervisorAgent:
    """协调 Agent，负责任务分配、分歧检测和报告合成。"""

    def assign_tasks(self, document: ContractDocument) -> dict[str, str]:
        """根据合同内容决定任务列表。
        当前实现：始终分配 ClauseExpert → RiskExpert → ComplianceExpert。
        返回 {worker_role: prompt} 字典。
        """
        return {
            "ClauseExpert": f"请提取以下合同中的所有关键条款，输出 JSON 数组：\n\n{document.content}",
            "RiskExpert": "",
            "ComplianceExpert": "",
        }

    def validate_consensus(
        self, outputs: dict[str, WorkerOutput]
    ) -> list[Disagreement]:
        """比对各 Worker 输出，找出分歧项。"""
        disagreements: list[Disagreement] = []

        clause_results: list[dict[str, Any]] = []
        for role, out in outputs.items():
            if out.structured and isinstance(out.structured, list):
                clause_results.append({"role": role, "data": out.structured})

        if len(clause_results) < 2:
            return disagreements

        # 按条款位置对比风险等级
        for i in range(max(len(r["data"]) for r in clause_results)):
            values: dict[str, Any] = {}
            for r in clause_results:
                if i < len(r["data"]):
                    item = r["data"][i]
                    if isinstance(item, dict):
                        rl = item.get("risk_level") or item.get("risk_level_note")
                        if rl:
                            values[r["role"]] = rl

            if len(values) >= 2:
                unique = set(values.values())
                if len(unique) > 1:
                    items = list(values.items())
                    for j in range(len(items)):
                        for k in range(j + 1, len(items)):
                            if items[j][1] != items[k][1]:
                                disagreements.append(Disagreement(
                                    item_id=f"clause-{i}",
                                    field="risk_level",
                                    value_a=items[j][1],
                                    value_b=items[k][1],
                                    worker_a=items[j][0],
                                    worker_b=items[k][0],
                                ))

        logger.info("Supervisor found {} disagreements", len(disagreements))
        return disagreements

    def synthesize_report(
        self,
        document: ContractDocument,
        outputs: dict[str, WorkerOutput],
        arbitration_results: list[dict[str, Any]],
    ) -> ReviewReport:
        """汇总 Worker 输出为 ReviewReport。"""
        clauses: list[Clause] = []
        risks: list[RiskAssessment] = []
        compliance: list[ComplianceCheck] = []

        clause_out = outputs.get("ClauseExpert")
        if clause_out and clause_out.structured:
            for c in clause_out.structured:
                if isinstance(c, dict):
                    clauses.append(Clause(
                        clause_type=c.get("type", "未知"),
                        content=c.get("content", ""),
                    ))

        risk_out = outputs.get("RiskExpert")
        if risk_out and risk_out.structured:
            for i, r in enumerate(risk_out.structured):
                if isinstance(r, dict):
                    clause = clauses[i] if i < len(clauses) else Clause(clause_type="未知", content="")
                    risks.append(RiskAssessment(
                        clause=clause,
                        risk_level=RiskLevel(r.get("risk_level", "info")),
                        reason=r.get("reason", ""),
                        suggestion=r.get("suggestion", ""),
                    ))

        compliance_out = outputs.get("ComplianceExpert")
        if compliance_out and compliance_out.structured:
            for c in compliance_out.structured:
                if isinstance(c, dict):
                    compliance.append(ComplianceCheck(
                        regulation=c.get("regulation", ""),
                        status=c.get("status", False),
                        detail=c.get("detail", ""),
                    ))

        overall_risk = self._compute_overall_risk(risks)
        summary = self._build_summary(clauses, risks, compliance)
        return ReviewReport(
            document_id=document.id,
            document_title=document.title,
            reviewed_at="",
            summary=summary,
            clauses=clauses,
            risks=risks,
            compliance_checks=compliance,
            overall_risk=overall_risk,
        )

    @staticmethod
    def _compute_overall_risk(risks: list[RiskAssessment]) -> RiskLevel:
        if not risks:
            return RiskLevel.INFO
        levels = [r.risk_level for r in risks]
        for lv in (RiskLevel.CRITICAL, RiskLevel.HIGH, RiskLevel.MEDIUM, RiskLevel.LOW):
            if lv in levels:
                return lv
        return RiskLevel.INFO

    @staticmethod
    def _build_summary(
        clauses: list[Clause], risks: list[RiskAssessment], compliance: list[ComplianceCheck]
    ) -> str:
        parts = [f"共发现 {len(clauses)} 个条款"]
        high = [r for r in risks if r.risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH)]
        parts.append(f"高风险项: {len(high)} 个")
        bad = [c for c in compliance if not c.status]
        parts.append(f"不合规项: {len(bad)} 个")
        return "；".join(parts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_multi_agent.py::TestSupervisorAgent -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add harness/agent/multi_agent/supervisor.py tests/unit/test_multi_agent.py
git commit -m "feat: add SupervisorAgent with consensus validation and report synthesis"
```

---

## Task 5: MultiAgentCoordinator 实现

**Files:**
- Create: `harness/agent/multi_agent/coordinator.py`

**Interfaces:**
- Consumes: `ContractDocument`, `LLMClient`, `SupervisorAgent`, `WorkerAgent`, `CrossValidator`, `KnowledgeRetriever`, `MemoryStore`
- Produces: `MultiAgentCoordinator.run(document)` → `tuple[ReviewReport, AgentSession]`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_multi_agent.py (append)
import json
from harness.agent.llm import LLMResponse
from harness.agent.multi_agent.coordinator import MultiAgentCoordinator
from harness.core.types import (
    Clause, ComplianceCheck, ContractDocument, RiskAssessment, RiskLevel,
)
from tests.conftest import MockLLMClient


class TestMultiAgentCoordinator:
    """MultiAgentCoordinator 集成测试。"""

    def test_run_returns_report_and_session(self, sample_document):
        """全流程应返回报告和会话。"""
        # 需要足够 mock 响应: ClauseExpert + RiskExpert + ComplianceExpert + 交叉验证(3次)
        llm = MockLLMClient([
            LLMResponse(content=json.dumps([
                {"type": "保密", "content": "双方应保守商业秘密"},
                {"type": "违约责任", "content": "违约方应赔偿损失"},
            ], ensure_ascii=False), model="mock"),
            LLMResponse(content=json.dumps([
                {"clause_type": "保密", "risk_level": "low", "reason": "标准", "suggestion": ""},
                {"clause_type": "违约责任", "risk_level": "medium", "reason": "模糊", "suggestion": "明确"},
            ], ensure_ascii=False), model="mock"),
            LLMResponse(content=json.dumps([], ensure_ascii=False), model="mock"),
            # 3 次交叉验证 LLM 调用
            LLMResponse(content="无分歧", model="mock"),
            LLMResponse(content="无分歧", model="mock"),
            LLMResponse(content="无分歧", model="mock"),
        ])
        coordinator = MultiAgentCoordinator(llm=llm)
        report, session = coordinator.run(sample_document)
        assert report.document_id == sample_document.id
        assert session.session_id
        assert len(session.steps) >= 3  # 至少三阶段

    def test_run_with_worker_failure_falls_back(self, sample_document):
        """单个 Worker 失败时不影响整体流程。"""
        llm = MockLLMClient([
            LLMResponse(content=json.dumps([
                {"type": "保密", "content": "保密内容"},
            ], ensure_ascii=False), model="mock"),
            # RiskExpert 返回空（模拟失败）
            LLMResponse(content="{}", model="mock"),
            LLMResponse(content=json.dumps([], ensure_ascii=False), model="mock"),
            # 交叉验证（跳过失败的 RiskExpert）
            LLMResponse(content="ok", model="mock"),
        ])
        coordinator = MultiAgentCoordinator(llm=llm)
        report, session = coordinator.run(sample_document)
        assert report.document_id == sample_document.id
        # 即使 RiskExpert 失败，也能得到报告
        assert report.summary

    def test_all_workers_fail_fallsback_to_pipeline(self, sample_document):
        """全部 Worker 失败时降级到 Pipeline。"""
        llm = MockLLMClient([
            LLMResponse(content="{}", model="mock"),
        ])
        coordinator = MultiAgentCoordinator(llm=llm)
        report, session = coordinator.run(sample_document)
        assert report is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_multi_agent.py::TestMultiAgentCoordinator -v`
Expected: FAIL (coordinator.py not yet created)

- [ ] **Step 3: Implement MultiAgentCoordinator**

```python
# harness/agent/multi_agent/coordinator.py
"""Multi-Agent 协同审查编排器。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from harness.agent.llm import LLMClient
from harness.agent.memory import MemoryStore
from harness.agent.multi_agent.supervisor import SupervisorAgent
from harness.agent.multi_agent.validator import CrossValidator
from harness.agent.multi_agent.worker import WorkerAgent
from harness.core.types import AgentSession, AgentStep, ContractDocument, ReviewReport, ToolCall
from harness.utils.io import make_id
from harness.utils.log import logger


class MultiAgentCoordinator:
    """编排 Supervisor + Worker + CrossValidator 多 Agent 审查流程。"""

    def __init__(
        self,
        llm: LLMClient,
        memory_store: MemoryStore | None = None,
    ):
        self._llm = llm
        self._memory = memory_store
        self._supervisor = SupervisorAgent()
        self._validator = CrossValidator(llm=self._llm)
        # Workers 共用同一个 LLMClient（实例隔离但底层复用）
        self._workers: dict[str, WorkerAgent] = {
            role: WorkerAgent(role=role, llm=LLMClient(
                # 复用主 LLM 的配置
                config=self._llm.config,
            ))
            for role in ("ClauseExpert", "RiskExpert", "ComplianceExpert")
        }

    def run(self, document: ContractDocument) -> tuple[ReviewReport, AgentSession]:
        """执行多 Agent 协同审查。"""
        session = AgentSession(
            session_id=make_id(),
            document=document,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        logger.info("Starting multi-agent review for document_id={}", document.id)

        # Phase 1: 分配任务
        tasks = self._supervisor.assign_tasks(document)

        # Phase 2-3: 串行执行（ClauseExpert → RiskExpert → ComplianceExpert）
        outputs: dict[str, Any] = {}
        phase_order = ["ClauseExpert", "RiskExpert", "ComplianceExpert"]

        for phase_idx, role in enumerate(phase_order):
            step = AgentStep(
                step_index=phase_idx,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            step.agent_message = f"正在执行 {role}..."
            tc = ToolCall(
                tool_name=role,
                input={"document_id": document.id},
                started_at=datetime.now(timezone.utc).isoformat(),
            )

            try:
                # 构造输入。RiskExpert 和 ComplianceExpert 需要已提取的条款
                if role == "ClauseExpert":
                    content = tasks.get(role, document.content)
                else:
                    clause_out = outputs.get("ClauseExpert")
                    if clause_out and clause_out.structured:
                        content = f"合同条款：\n{clause_out.content}"
                    else:
                        content = document.content

                worker_out = self._workers[role].execute(content)
                outputs[role] = worker_out
                tc.output = f"{role} 执行完成"
                logger.info("{} completed successfully", role)
            except Exception as e:
                logger.warning("{} failed: {}", role, e)
                tc.output = str(e)
            finally:
                tc.finished_at = datetime.now(timezone.utc).isoformat()
                step.tool_calls.append(tc)
                session.steps.append(step)

        # 检查是否全部失败 → 降级
        if not any(out for out in outputs.values()):
            logger.warning("All workers failed, synthesizing partial report")
            return self._build_partial_report(document, session)

        # Phase 4: 交叉验证
        step = AgentStep(
            step_index=len(session.steps),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        step.agent_message = "正在交叉验证..."
        tc = ToolCall(
            tool_name="cross_validation",
            input={},
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        for role, worker in self._workers.items():
            if role not in outputs:
                continue
            try:
                peer = {k: v.content for k, v in outputs.items() if k != role}
                if peer:
                    self._workers[role].execute("确认审查结果", peer_results=peer)
            except Exception as e:
                logger.warning("Cross-validation for {} failed: {}", role, e)
        tc.output = "交叉验证完成"
        tc.finished_at = datetime.now(timezone.utc).isoformat()
        step.tool_calls.append(tc)
        session.steps.append(step)

        # Phase 5: 分歧检测
        disagreements = self._supervisor.validate_consensus(
            {k: v for k, v in outputs.items() if v}
        )

        # Phase 6: 仲裁
        arbitration = self._validator.arbitrate(disagreements)

        # Phase 7: 合成报告
        report = self._supervisor.synthesize_report(document, outputs, arbitration)
        report.reviewed_at = datetime.now(timezone.utc).isoformat()
        session.report = report
        session.finished_at = datetime.now(timezone.utc).isoformat()

        if disagreements:
            logger.info("Multi-agent review found {} disagreements", len(disagreements))

        logger.info("Multi-agent review completed for document_id={}", document.id)
        return report, session

    def _build_partial_report(
        self, document: ContractDocument, session: AgentSession
    ) -> tuple[ReviewReport, AgentSession]:
        """全部 Worker 失败时生成简化报告。"""
        report = ReviewReport(
            document_id=document.id,
            document_title=document.title,
            reviewed_at=datetime.now(timezone.utc).isoformat(),
            summary="多 Agent 审查全部失败，请重试或切换为 Pipeline 模式",
            clauses=[],
            risks=[],
            compliance_checks=[],
            overall_risk=None,  # type: ignore[arg-type]
        )
        session.report = report
        session.finished_at = datetime.now(timezone.utc).isoformat()
        return report, session
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_multi_agent.py::TestMultiAgentCoordinator -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add harness/agent/multi_agent/coordinator.py tests/unit/test_multi_agent.py
git commit -m "feat: add MultiAgentCoordinator with full orchestration"
```

---

## Task 6: ContractAgent 集成

**Files:**
- Modify: `harness/agent/contract_agent.py` — 新增 MULTI_AGENT 分支
- Modify: `harness/agent/__init__.py` — 导出（可选）

**Interfaces:**
- Consumes: `AgentMode.MULTI_AGENT`, `MultiAgentCoordinator`
- Produces: `ContractAgent.review()` 新增 `MULTI_AGENT` 调度

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_multi_agent.py (append)
import json
from harness.agent.contract_agent import ContractAgent
from harness.agent.llm import LLMResponse
from harness.core.types import AgentMode, ContractDocument
from tests.conftest import MockLLMClient


class TestContractAgentMultiAgent:
    """ContractAgent Multi-Agent 模式集成测试。"""

    def test_multi_agent_mode_dispatch(self, sample_document):
        """MULTI_AGENT 模式应正确分发到 MultiAgentCoordinator。"""
        llm = MockLLMClient([
            LLMResponse(content=json.dumps([
                {"type": "保密", "content": "双方应保守商业秘密"},
                {"type": "违约责任", "content": "违约方应赔偿损失"},
            ], ensure_ascii=False), model="mock"),
            LLMResponse(content=json.dumps([
                {"clause_type": "保密", "risk_level": "low", "reason": "标准", "suggestion": ""},
                {"clause_type": "违约责任", "risk_level": "medium", "reason": "模糊", "suggestion": "明确"},
            ], ensure_ascii=False), model="mock"),
            LLMResponse(content=json.dumps([], ensure_ascii=False), model="mock"),
            LLMResponse(content="无分歧", model="mock"),
            LLMResponse(content="无分歧", model="mock"),
            LLMResponse(content="无分歧", model="mock"),
        ])
        agent = ContractAgent(llm=llm, mode=AgentMode.MULTI_AGENT)
        report, session = agent.review(sample_document)
        assert report.document_id == sample_document.id
        assert session.session_id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_multi_agent.py::TestContractAgentMultiAgent -v`
Expected: FAIL (ContractAgent not yet wired)

- [ ] **Step 3: Update ContractAgent**

```python
# In harness/agent/contract_agent.py, add imports at top:
from harness.agent.multi_agent.coordinator import MultiAgentCoordinator

# In review() method, add branch:
def review(self, document: ContractDocument) -> tuple[ReviewReport, AgentSession]:
    if self._mode == AgentMode.REACT:
        return self._review_react(document)
    if self._mode == AgentMode.REFLECTION:
        return self._review_reflection(document)
    if self._mode == AgentMode.MULTI_AGENT:
        return self._review_multi_agent(document)
    return self._review_pipeline(document)

# Add new method after _review_reflection:
def _review_multi_agent(self, document: ContractDocument) -> tuple[ReviewReport, AgentSession]:
    """Multi-Agent 模式：多 Agent 协同审查。"""
    coordinator = MultiAgentCoordinator(
        llm=self._llm,
        memory_store=self._memory,
    )
    return coordinator.run(document)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_multi_agent.py::TestContractAgentMultiAgent -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add harness/agent/contract_agent.py tests/unit/test_multi_agent.py
git commit -m "feat: integrate MultiAgentCoordinator into ContractAgent as MULTI_AGENT mode"
```

---

## Task 7: Config 与 CLI 集成（可选，小变更）

**Files:**
- Modify: `harness/core/config.py` — `HarnessConfig.agent_mode` 默认值保持不变，但支持 multi_agent
- (CLI 已有 `--mode` 参数，无需修改)

- [ ] **Step 1: Verify existing CLI handles multi_agent**

Run: `pytest tests/unit/test_cli.py -v | grep -i mode`
Expected: CLI test passes with no changes needed

- [ ] **Step 2: Commit any necessary config update (if any)**

```bash
# Only if changes were needed:
git add harness/core/config.py
git commit -m "chore: enable agent_mode multi_agent in config"
```
