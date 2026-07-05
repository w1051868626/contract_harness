---
name: sync-deps
description: 校验 pyproject.toml / environment.yml / requirements.txt 三者依赖一致性，输出差异并提示修复。AGENTS.md 规定三者必须保持一致。
---

# /sync-deps

校验 contract_harness 项目的依赖三方一致性。

## 执行步骤

1. 解析三个文件的依赖列表：
   - `pyproject.toml` 的 `[project].dependencies` + `[project.optional-dependencies]`
   - `environment.yml` 的 `dependencies:` 列表
   - `requirements.txt` 的逐行依赖

2. 归一化对比：
   - 包名统一小写
   - 版本约束保留（`>=X`、`==X`、`~=X`）
   - 忽略注释行与空行
   - 注明每个包来自哪个 optional group（dev/local/docling）

3. 输出三向 diff 表：

```
依赖一致性检查
==============
                  pyproject.toml    environment.yml    requirements.txt
click             >=8.1             >=8.1              >=8.1            ✓
chromadb          >=0.5.0           —                  >=0.5.0          ⚠ environment.yml 缺失
docling           [docling] >=2.0   —                  —                ⚠ 三者均未列入主依赖（optional）
sentence-transformers [local] >=3.0 —                 —                ⚠ requirements.txt 缺失
new-package       >=1.0             >=1.0              >=1.0            ✓

汇总：
  ✓ 一致：N 个
  ⚠ 不一致：M 个
    1. chromadb — environment.yml 缺失
    2. ...
```

4. 若发现不一致，给出修复建议（在哪个文件加哪一行），但不自动修改。

## 规约参考

来自 `AGENTS.md`：
> 新增依赖时同步更新 `pyproject.toml`、`environment.yml`、`requirements.txt`（三者必须保持一致）

## 边界

- 不处理依赖版本冲突（如 `pyproject.toml` 写 `>=1.0` 而 `requirements.txt` 写 `==0.9`）——只报告，不求解
- 不解析 conda 渠道（`pip::`、`conda-forge::`）——按包名匹配即可
