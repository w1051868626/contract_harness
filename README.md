# contract-harness

可回放、可评测、可回归的合同审查 Agent 系统。

## 安装

```bash
pip install -e .
```

## 快速使用

### 审查合同

```bash
harness review examples/contracts/sample_nda.json
```

### 回放审查会话

```bash
harness replay <session_id>
harness sessions  # 列出所有会话
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

## 架构

```
harness/
├── agent/        合同审查 Agent（LLM 编排 + 工具调用）
├── replay/       回放系统（录制 + 回放 + 存储管理）
├── eval/         评测系统（数据集 + 指标 + 评分流水线）
├── regression/   回归系统（测试套件 + 对比器）
├── cli/          命令行入口
├── core/         核心类型、配置、异常
└── utils/        工具函数
```

## 环境变量

| 变量 | 说明 | 默认值 |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API 密钥 | - |
| `HARNESS_DATA_DIR` | 数据存储目录 | `~/.harness/data` |

## 开发

```bash
pip install -e ".[dev]"
pytest tests/
ruff check harness/
```
