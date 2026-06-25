# GitHub Actions 流水线

项目包含两条 GitHub Actions 流水线：

| 文件 | 触发条件 | 说明 |
|------|----------|------|
| `.github/workflows/ci.yml` | push / PR → main | 代码质量检查（lint + typecheck + test） |
| `.github/workflows/regression.yml` | 每周一 02:00 + workflow_dispatch + push 到 main 且 harness/ 变更 | 回归测试 |

---

## CI — `ci.yml`

### 触发

- `push` 到 `main`
- `pull_request` 到 `main`

### Jobs

#### lint

- **OS:** ubuntu-latest | **Python:** 3.11
- 安装 `ruff` → `ruff check harness/ tests/` → `ruff format --check harness/ tests/`

#### type-check

- **OS:** ubuntu-latest | **Python:** 3.11
- `pip install -e ".[dev]"` → `pyright harness/`

#### test

- **OS:** ubuntu-latest | **Matrix:** Python 3.11 / 3.12 / 3.13
- `pip install -e ".[dev]"` → `pytest tests/ --cov=harness --cov-report=xml --cov-report=term-missing:skip-covered`
- 上传覆盖率到 Codecov
- 上传 `coverage.xml` 为 artifact（无论成败）

---

## 回归测试 — `regression.yml`

### 触发

- **定时:** 每周一 02:00 UTC（`cron: "0 2 * * 1"`）
- **手动:** `workflow_dispatch`
- **Push 到 main** 且 `harness/**` 有变更

### 并发

同一 ref 排队，新任务取消旧任务。

### Jobs

#### regression

- **OS:** ubuntu-latest | **Python:** 3.11
- `pip install -e ".[dev]"`
- `OPENAI_API_KEY` 从 secrets 注入（缺失时使用 Mock LLM）
- 运行 `harness regression run examples/contracts/ --version "ci-<YYYYMMDD>-<SHA7>"`
- 上传 `~/.harness/reports/` 为 artifact，保留 30 天

---

## 本地模拟

```bash
pip install -e ".[dev]"

ruff check harness/ tests/
ruff format --check harness/ tests/
pyright harness/
pytest tests/ -v --cov=harness --cov-report=term-missing:skip-covered

# 回归测试（需 OPENAI_API_KEY）
harness regression run examples/contracts/ --version "local-$(date +'%Y%m%d')"
```
