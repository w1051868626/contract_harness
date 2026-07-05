#!/usr/bin/env node
/**
 * block-sensitive.js — AtomCode PreToolUse hook
 *
 * 拦截对敏感文件/目录的 Edit|Write，避免误改业务资产或破坏依赖三方一致性。
 *
 * 拦截规则：
 *  1. .env / .env.local / *.env          — 凭据文件（example.env 除外）
 *  2. examples/contracts-XXX/ 下所有 .json/.md/.txt — 评测数据集（业务资产）
 *  3. environment.yml / pyproject.toml / requirements.txt 单独修改
 *                                         — AGENTS.md 规定三者必须同步
 *  4. .harness/**                         — 运行时数据（向量库/记忆/回放）
 *
 * 退出码：
 *   0  — 放行
 *   2  — 阻止工具调用（AtomCode 会把 stderr 反馈给模型）
 */

const path = require("path");

// AtomCode 通过 stdin 传入 JSON：{ tool_name, tool_input: { file_path, ... } }
let raw = "";
try {
  raw = require("fs").readFileSync(0, "utf8");
} catch (_) {
  process.exit(0); // 无输入则放行
}

let payload = {};
try {
  payload = JSON.parse(raw);
} catch (_) {
  process.exit(0); // 解析失败放行，不阻塞正常工作流
}

const toolName = payload.tool_name || "";
const filePath = (payload.tool_input && (payload.tool_input.file_path || payload.tool_input.path)) || "";

if (!filePath) {
  process.exit(0);
}

// 归一化为正斜杠相对路径
const projectDir = process.env.CLAUDE_PROJECT_DIR || process.cwd();
let rel = filePath;
try {
  rel = path.relative(projectDir, path.resolve(projectDir, filePath)).replace(/\\/g, "/");
} catch (_) {
  /* 保留原值 */
}

const rules = [
  {
    name: "凭据文件 .env",
    test: (p) => /(^|\/)\.env(\.[a-z]+)?$/.test(p) && !/example\.env$/i.test(p) && !/\.env\.example$/i.test(p),
    hint: "凭据文件禁止直接编辑。如需新增变量请改 .env.example 并让团队成员各自复制。",
  },
  {
    name: "评测数据集 examples/contracts*/",
    test: (p) => /^examples\/contracts[^/]*\/.*\.(json|md|txt)$/i.test(p),
    hint: "评测数据集是业务资产，禁止随意修改。如需新增用例请新建文件并同步更新数据集元信息。",
  },
  {
    name: "运行时数据 .harness/",
    test: (p) => /^\.harness\//.test(p),
    hint: ".harness/ 是运行时数据（向量库/记忆/回放/报告），禁止手工编辑。",
  },
  {
    name: "依赖清单单独修改",
    test: (p) =>
      /^(environment\.yml|pyproject\.toml|requirements\.txt)$/.test(p),
    hint: "AGENTS.md 规定 environment.yml / pyproject.toml / requirements.txt 必须同步修改。请使用 /sync-deps 命令或在一次变更里同时更新三者。",
  },
];

for (const r of rules) {
  if (r.test(rel)) {
    process.stderr.write(
      `⛔ 阻止 ${toolName} → ${rel}\n原因：${r.name}\n${r.hint}\n`
    );
    process.exit(2);
  }
}

process.exit(0);
