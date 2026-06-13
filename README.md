# contract-harness

可回放、可评测、可回归的**法律合同审查 Agent** 系统。

基于自研 Agent Loop 框架，集成了 LLM 编排、工具调用、会话回放、评测评分和回归对比能力。

## 安装

### Conda（推荐）

```bash
conda env create -f environment.yml
conda activate contract-harness
pip install -e .
```

### pip

```bash
pip install -e .
```

## 快速使用

### 审查合同

```bash
harness review examples/contracts/sample_nda.json
```

审查流程：条款提取 → 风险分析 → 合规检查 → 生成报告

### 回放审查会话

```bash
harness replay <session_id>
harness sessions          # 列出所有会话
harness replay <id> --json  # JSON 格式输出
```

### 运行评测

```bash
harness eval run examples/contracts/
harness eval report
```

### 回归测试

```bash
harness regression run examples/contracts/
harness regression diff <session_a> <session_b>
```

### Web 界面

```bash
harness serve
```

访问 http://127.0.0.1:8000 查看 Web 界面，支持合同上传审查、会话回放等功能。

## 架构

```
harness/
├── agent/        合同审查 Agent（LLM 编排 + 工具调用）
├── replay/       回放系统（录制 + 回放 + 存储管理）
├── eval/         评测系统（数据集 + 指标 + 评分流水线）
├── regression/   回归系统（测试套件 + 对比器）
├── rag/          知识库（Embedding + 向量存储 + 检索）
├── cli/          命令行入口（click + rich）
├── core/         核心类型（pydantic）、配置、异常
└── utils/        工具函数
```

### 流程图

```mermaid
graph TB
    subgraph CLI["CLI 入口"]
        CMD["harness review &lt;file&gt;<br/>harness replay &lt;session&gt;<br/>harness eval run<br/>harness regression run"]
    end

    subgraph AGENT["Agent 流水线"]
        direction TB
        S0["Step 0<br/>知识库检索"]:::step --> S1["Step 1<br/>条款提取<br/>(ClauseExtractor)"]:::step
        S1 --> S2["Step 2<br/>风险分析<br/>(RiskAnalyzer)"]:::step
        S2 --> S3["Step 3<br/>合规检查<br/>(ComplianceChecker)"]:::step
        S3 --> S4["Step 4<br/>生成摘要<br/>(LLM)"]:::step
        S0 -.->|可选| KB[("SQLite 向量库<br/>KnowledgeBase")]
    end

    subgraph LLM["LLM 层"]
        LLMC["LLMClient"] -->|OpenAI 协议| APIS["OpenAI / Ollama<br/>vLLM / Azure ..."]
        LLMC -->|httpx proxy| PROXY["代理"]
    end

    subgraph OUTPUT["输出"]
        REPORT["ReviewReport<br/>(clauses + risks + compliance)"]
        SESSION["AgentSession<br/>(4 步全量 Trace + ToolCall)"]
    end

    subgraph HARNESS["Harness 系统"]
        REPLAY["ReplaySystem<br/>录制 → JSON → 回放"]
        EVAL["EvalSystem<br/>Ground Truth → 4 项指标"]
        REGR["RegressionSystem<br/>基线对比 → pass/fail"]
    end

    AGENT --> LLM
    AGENT --> OUTPUT
    SESSION --> REPLAY
    SESSION --> EVAL
    SESSION --> REGR
    CMD --> AGENT

    classDef step fill:#4a90d9,color:#fff
```

## 环境变量

| 变量 | 说明 | 默认值 |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API 密钥（必填） | - |

## 开发

```bash
conda activate contract-harness
pip install -e ".[dev]"
pytest tests/ -v             # 运行测试
ruff check harness/ tests/   # 代码检查
```
