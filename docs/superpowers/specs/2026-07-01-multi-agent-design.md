# Multi-Agent 协同审查设计

## 背景

现有 `ContractAgent` 以单体方式运行（Pipeline/ReAct/Reflection 三种模式）。为满足专业分工 + 交叉验证的需求，新增第四种 `MULTI_AGENT` 模式，引入多 Agent 协同架构。

## 架构

```
ContractAgent.review()
  ├─ PIPELINE    → _review_pipeline()
  ├─ REACT       → ReActLoop.run()
  ├─ REFLECTION  → _review_reflection()
  └─ MULTI_AGENT → MultiAgentCoordinator.run()
```

### 核心组件

#### 1. AgentMode 枚举新增

```python
class AgentMode(str, Enum):
    PIPELINE = "pipeline"
    REACT = "react"
    REFLECTION = "reflection"
    MULTI_AGENT = "multi_agent"  # 新增
```

#### 2. MultiAgentCoordinator

职责：编排多 Agent 协同审查全流程。

```
MultiAgentCoordinator
├── SupervisorAgent       # 任务分配 + 分歧检测 + 报告合成
├── WorkerAgent × 3      # 领域专业 Agent
│   ├── ClauseExpert     # 条款提取
│   ├── RiskExpert       # 风险评估
│   └── ComplianceExpert # 合规检查
└── CrossValidator       # 分歧仲裁（规则优先，LLM 兜底）
```

#### 3. SupervisorAgent

轻量协调 Agent，不执行具体审查。

- `assign_tasks(document)` → 根据合同类型/内容决定分派策略
- `validate_consensus(results)` → 纯逻辑对比各 Worker 输出，标记分歧
- `synthesize_report(results, disagreements)` → 汇总最终报告，分歧项由 CrossValidator 仲裁后注入注释

#### 4. WorkerAgent

每个 Worker 自持独立 `LLMClient` + 专用 system prompt + 领域工具：

| Worker | 工具 | 上游依赖 |
|---|---|---|
| ClauseExpert | ClauseExtractor | 合同原文 |
| RiskExpert | RiskAnalyzer | ClauseExpert 的输出 |
| ComplianceExpert | ComplianceChecker | ClauseExpert 的输出 |

```python
class WorkerAgent:
    role: str
    _llm: LLMClient
    _system_prompt: str
    _tools: dict[str, Callable]

    def execute(task: WorkerTask, peer_results: dict | None = None) -> WorkerOutput
```

- `peer_results` 参数支持交叉验证：Worker 审阅同行输出，标记分歧点
- 独立 LLM 实例避免上下文污染
- 共享同一个 KnowledgeRetriever（只读无状态）

#### 5. CrossValidator

分层分歧处理规则：

| 分歧类型 | 处理方式 |
|---|---|
| 风险等级差 1 级 | 取更严格侧 |
| 风险等级差 ≥2 级 | LLM 仲裁 |
| 合规状态不同 | 标记"待人工审核" |
| 条款归类不同 | 都保留 |

## 流程

### 四阶段执行

```
Phase 1: Supervisor.assign_tasks(document)
Phase 2: ClauseExpert.extract() → clauses
Phase 3: RiskExpert.analyze(clauses) → risks
         ComplianceExpert.check(clauses) → compliance
Phase 4: 并行交叉验证
         ├── ClauseExpert 审阅 RiskExpert
         ├── RiskExpert 审阅 ComplianceExpert
         └── ComplianceExpert 审阅 RiskExpert
Phase 5: Supervisor.validate_consensus() → disagreements
Phase 6: CrossValidator.arbitrate(disagreements) → 仲裁结果
Phase 7: Supervisor.synthesize_report() → ReviewReport
```

### 异常处理

- 单个 Worker LLM 调用失败 → 用其他 Worker 结果兜底
- 全部 Worker 失败 → 降级到 Pipeline 模式
- 交叉验证阶段单个 Worker 失败 → 跳过该次验证

## 数据模型

### 新增类型（`harness/core/types.py`）

```python
@dataclass
class WorkerTask:
    worker_role: str
    prompt: str
    input_data: Any
    context: dict[str, Any]

@dataclass
class WorkerOutput:
    worker_role: str
    content: str
    structured: Any
    confidence: float = 1.0

@dataclass
class Disagreement:
    item_id: str
    field: str
    value_a: Any
    value_b: Any
    worker_a: str
    worker_b: str
```

## 文件清单（新增）

| 文件 | 内容 |
|---|---|
| `harness/agent/multi_agent/__init__.py` | 导出 |
| `harness/agent/multi_agent/coordinator.py` | MultiAgentCoordinator |
| `harness/agent/multi_agent/supervisor.py` | SupervisorAgent |
| `harness/agent/multi_agent/worker.py` | WorkerAgent |
| `harness/agent/multi_agent/validator.py` | CrossValidator |

## 变更清单（修改）

| 文件 | 变更 |
|---|---|
| `harness/agent/__init__.py` | 导出 MultiAgentCoordinator |
| `harness/agent/contract_agent.py` | 新增 MULTI_AGENT 分支 |
| `harness/core/types.py` | 新增 WorkerTask/WorkerOutput/Disagreement；AgentMode 新增 MULTI_AGENT |
| `harness/core/config.py` | HarnessConfig 可选新增 multi_agent 配置 |
| `tests/unit/test_multi_agent.py` | 新增测试 |

## 测试策略

1. **单元测试**: WorkerAgent.execute（mock LLM）、CrossValidator.arbitrate（纯规则路径 vs LLM 路径）、Supervisor.validate_consensus
2. **集成测试**: MultiAgentCoordinator.run 全流程（mock 所有 Worker LLM 调用）
3. **异常测试**: Worker 失败降级、全部 Worker 失败降级 Pipeline
4. **交叉验证测试**: 有分歧 / 无分歧场景
