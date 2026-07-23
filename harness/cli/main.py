"""contract-harness 命令行入口，支持审查、回放、评测、回归与 Web 服务。"""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path

import click
import uvicorn

from harness.agent.contract_agent import ContractAgent
from harness.agent.llm import LLMClient
from harness.core.config import HarnessConfig
from harness.core.types import ContractDocument
from harness.eval.dataset import EvalDataset
from harness.eval.reporters import EvalReporter
from harness.eval.scorer import EvalScorer
from harness.rag.knowledge_base import KnowledgeBase
from harness.rag.parsing import enable_docling as _enable_docling
from harness.rag.parsing import parse_file as _parse_file
from harness.rag.seed_laws import get_seed_laws
from harness.regression.comparator import OutputComparator
from harness.regression.suite import RegressionSuite
from harness.replay.player import SessionPlayer
from harness.replay.recorder import SessionRecorder
from harness.replay.storage import ReplayStorage
from harness.utils.io import load_dotenv, read_text
from harness.utils.log import logger, setup_logging

load_dotenv()


def _get_config(ctx: click.Context) -> HarnessConfig:
    return ctx.obj["config"]


def _get_kb(ctx: click.Context) -> KnowledgeBase:
    config = _get_config(ctx)
    config.ensure_dirs()
    return KnowledgeBase.from_config(config)


def _get_player(ctx: click.Context) -> SessionPlayer:
    config = _get_config(ctx)
    return SessionPlayer(ReplayStorage(config.replay_dir))


def _load_yaml_run(path: str) -> dict | None:
    """从 YAML 配置文件中提取 ``run`` 字段（不触发完整配置加载）。"""
    try:
        from harness.core.config import load_config_yaml

        data = load_config_yaml(path)
        return data.get("run")
    except Exception:
        return None


@click.group(invoke_without_command=True)
@click.option("--verbose", "-v", is_flag=True, help="启用详细输出")
@click.option(
    "--config",
    "-c",
    default="",
    help="YAML 配置文件路径（支持 ${VAR_NAME} 环境变量引用）",
)
@click.pass_context
def cli(ctx: click.Context, verbose: bool, config: str) -> None:
    """合同审查 Agent 系统 CLI。"""
    yaml_run: dict | None = None
    if config:
        # 先提取 run 字段（CLI 自动执行用），再加载正式配置
        yaml_run = _load_yaml_run(config)
        cfg = HarnessConfig.from_yaml(config)
        cfg.verbose = verbose or cfg.verbose
    else:
        cfg = HarnessConfig()
        cfg.verbose = verbose
    if cfg.verbose:
        cfg.ensure_dirs()
    setup_logging(verbose=cfg.verbose, log_dir=cfg.log_dir)
    ctx.ensure_object(dict)
    ctx.obj["config"] = cfg
    logger.debug("CLI 启动 (verbose={})", cfg.verbose)

    # 无子命令 + YAML 定义了 run → 自动执行
    if ctx.invoked_subcommand is None and yaml_run and not os.getenv("_HARNESS_RUN_IN_PROGRESS"):
        cmd_str = yaml_run.get("command", "")
        if not cmd_str:
            return
        os.environ["_HARNESS_RUN_IN_PROGRESS"] = "1"
        cmd_parts = shlex.split(cmd_str)
        args_dict = yaml_run.get("arguments", {}) or {}

        # 遍历命令树找到目标命令，区分 Option（--key=value）和 Argument（位置参数）
        target: click.Command | click.Group | None = cli
        for part in cmd_parts:
            if isinstance(target, click.Group):
                target = target.get_command(ctx, part)
            else:
                target = None
                break
        option_names: set[str] = set()
        if target and hasattr(target, "params"):
            for p in target.params:
                if isinstance(p, click.Option):
                    for opt in p.opts:
                        option_names.add(opt.lstrip("-"))

        for k, v in args_dict.items():
            click_key = k.replace("_", "-")
            if click_key in option_names:
                if isinstance(v, bool):
                    cmd_parts.append(f"--{click_key}" if v else f"--no-{click_key}")
                else:
                    cmd_parts.append(f"--{click_key}={v}")
            else:
                # 位置参数（如 dataset），直接追加值
                cmd_parts.append(str(v))
        try:
            ctx.exit(cli.main(args=cmd_parts, standalone_mode=False))
        finally:
            os.environ.pop("_HARNESS_RUN_IN_PROGRESS", None)


@cli.command()
@click.argument("contract_file", type=click.Path(exists=True))
@click.option("--save/--no-save", default=True, help="是否保存回放记录")
@click.option("--model", default="", help="LLM 模型名称")
@click.option("--docling", is_flag=True, help="使用 Docling 解析（保留结构）")
@click.pass_context
def review(ctx: click.Context, contract_file: str, save: bool, model: str, docling: bool) -> None:
    """审查一份合同并展示结果。"""
    config = _get_config(ctx)
    filepath = Path(contract_file)
    if docling:
        config.use_docling = True
        _enable_docling()
        content = _parse_file(filepath)
    else:
        content = read_text(filepath)

    document = ContractDocument(
        id=filepath.stem,
        title=filepath.name,
        content=content,
        file_path=str(filepath.absolute()),
    )

    if model:
        config.llm.model = model

    logger.info("正在审查合同: {}", filepath.name)
    agent = ContractAgent(LLMClient(config.llm))
    report, session = agent.review(document)

    logger.info("审查完成: {}", filepath.name)
    logger.info("会话 ID: {}", session.session_id)
    logger.info("整体风险: {}", report.overall_risk.value.upper())
    logger.info("审查摘要:\n{}", report.summary[:500])

    logger.info("条款概览:")
    for c in report.clauses:
        comment = c.comment[:40] if c.comment else ""
        logger.info("  {} | {} | {}", c.clause_type, c.risk.value, comment)

    if save:
        recorder = SessionRecorder(config.replay_dir)
        path = recorder.record(session)
        logger.info("回放已保存: {} (会话 ID: {})", path, session.session_id)


@cli.command()
@click.argument("session_id")
@click.argument("query", nargs=-1, required=True)
@click.option("--model", default="", help="LLM 模型名称")
@click.pass_context
def converse(ctx: click.Context, session_id: str, query: tuple[str], model: str) -> None:
    """对已有审查会话继续追问。"""
    config = _get_config(ctx)
    if model:
        config.llm.model = model
    agent = ContractAgent(LLMClient(config.llm))
    answer = agent.converse(session_id, " ".join(query))
    logger.info(answer)


@cli.command()
@click.argument("session_id")
@click.option("--json", "as_json", is_flag=True, help="以 JSON 格式输出")
@click.pass_context
def replay(ctx: click.Context, session_id: str, as_json: bool) -> None:
    """回放指定审查会话。"""
    player = _get_player(ctx)
    session = player.load(session_id)

    if session is None:
        logger.warning("会话 {} 不存在", session_id)
        return

    if as_json:
        r = SessionRecorder()
        data = r._serialize(session)
        logger.info(json.dumps(data, ensure_ascii=False, indent=2))
        return

    logger.info("回放会话: {}", session.session_id)
    logger.info("合同: {}", session.document.title)
    logger.info("时间: {}", session.started_at)

    for step in session.steps:
        logger.info("Step {}: {}", step.step_index, step.agent_message)
        for tc in step.tool_calls:
            logger.info("  工具: {} ({} chars)", tc.tool_name, len(str(tc.input)))


@cli.command()
@click.option("--limit", default=20, help="显示最近的会话数量")
@click.pass_context
def sessions(ctx: click.Context, limit: int) -> None:
    """列出所有回放会话。"""
    player = _get_player(ctx)
    sessions_list = player.list_sessions()

    if not sessions_list:
        logger.info("暂无回放记录")
        return

    logger.info("回放会话列表:")
    for s in sessions_list[:limit]:
        logger.info("  {} | {} | {}", s["session_id"], s["document_title"], s["started_at"])


@cli.group()
def eval() -> None:
    """评测命令组."""


@eval.command()
@click.argument("dataset", type=click.Path(exists=True))
@click.pass_context
def run(ctx: click.Context, dataset: str) -> None:
    """在指定数据集上运行评测并生成报告。"""
    config = _get_config(ctx)
    ds = EvalDataset()
    ds.load(dataset)

    logger.info("正在运行评测...")
    scorer = EvalScorer()
    data = scorer.score(ds)

    logger.info("评测完成")
    for name, value in data.get("aggregated_metrics", {}).items():
        pct = value * 100
        logger.info("  {}: {:.1f}%", name, pct)

    reporter = EvalReporter(config.report_dir)
    md_path = reporter.report_markdown(data)
    html_path = reporter.report_html(data)
    logger.info("报告已生成:")
    logger.info("  Markdown: {}", md_path)
    logger.info("  HTML:     {}", html_path)


@eval.command()
@click.option("--output", default="eval_report", help="报告文件名")
@click.pass_context
def report(ctx: click.Context, output: str) -> None:
    """展示评测报告相关信息。"""
    config = _get_config(ctx)
    logger.info("报告目录: {}", config.report_dir)
    logger.info("使用 'eval run' 运行评测后会自动生成报告")


@cli.group()
def regression() -> None:
    """回归测试命令组."""


@regression.command(name="run")
@click.argument("dataset", type=click.Path(exists=True))
@click.option("--version", default="", help="当前版本标识")
@click.pass_context
def regression_run(ctx: click.Context, dataset: str, version: str) -> None:
    """运行回归测试并与基线对比。"""
    ds = EvalDataset()
    ds.load(dataset)

    logger.info("正在运行回归测试...")
    suite = RegressionSuite()
    result = suite.run(ds, version)

    if result.passed:
        logger.info("回归测试通过")
    else:
        logger.warning("回归测试失败")

    if result.improvements:
        logger.info("改进:")
        for imp in result.improvements:
            logger.info("  + {}", imp)
    if result.regressions:
        logger.warning("回归:")
        for reg in result.regressions:
            logger.warning("  - {}", reg)
    if result.metrics_diff:
        logger.info("指标变化:")
        for name, diff in result.metrics_diff.items():
            logger.info("  {}: {:+.2%}", name, diff)


@regression.command()
@click.argument("session_a")
@click.argument("session_b")
@click.pass_context
def diff(ctx: click.Context, session_a: str, session_b: str) -> None:
    """对比两个会话的审查差异。"""
    config = _get_config(ctx)
    comparator = OutputComparator(SessionPlayer(ReplayStorage(config.replay_dir)))

    logger.info("正在对比...")
    diff_result = comparator.compare_by_session_id(session_a, session_b)

    logger.info("对比结果:")
    if diff_result.get("risk_level_changed"):
        logger.warning("  风险等级发生变化")
    if diff_result.get("summary_changed"):
        logger.info("  摘要内容发生变化")
    if diff_result.get("clause_diffs"):
        logger.info("  条款变化: {} 处", len(diff_result["clause_diffs"]))
    if diff_result.get("risk_diffs"):
        logger.info("  风险评估变化: {} 处", len(diff_result["risk_diffs"]))
    if diff_result.get("compliance_diffs"):
        logger.info("  合规检查变化: {} 处", len(diff_result["compliance_diffs"]))


@cli.command()
@click.option("--host", default="127.0.0.1", help="监听地址")
@click.option("--port", default=8000, help="监听端口")
@click.option("--reload", is_flag=True, help="热重载")
def serve(host: str, port: int, reload: bool) -> None:
    """启动 FastAPI Web 界面。"""
    from harness.web.app import app

    setup_logging(log_dir=HarnessConfig().log_dir)
    logger.info("启动 Web 界面: http://{}:{}", host, port)
    if reload:
        logger.info("热重载已启用")
    uvicorn.run(app, host=host, port=port, reload=reload)


# ---- 知识库命令 ----


@cli.group()
def kb() -> None:
    """知识库管理命令组."""


@kb.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--docling", is_flag=True, help="使用 Docling 解析（保留结构）")
@click.option("--work-dir", default=None, help="临时文件目录（C 盘空间不足时指定）")
@click.pass_context
def import_file(ctx: click.Context, file_path: str, docling: bool, work_dir: str | None) -> None:
    """将单个文件导入知识库（zip 会自动解压分别导入）。"""
    config = _get_config(ctx)
    config.use_docling = docling
    kb_instance = _get_kb(ctx)
    logger.info("正在导入文件: {}", Path(file_path).name)
    if file_path.lower().endswith(".zip"):
        doc_ids = kb_instance.add_zip(Path(file_path), work_dir=work_dir)
        if doc_ids:
            logger.info("导入成功: {}", Path(file_path).name)
            for did in doc_ids:
                logger.info("  + {}", did)
        else:
            logger.warning("导入失败")
    else:
        doc_id = kb_instance.add_file(file_path)
        if doc_id:
            logger.info("导入成功: {} -> {}", Path(file_path).name, doc_id)
        else:
            logger.warning("导入失败")


@kb.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option("--docling", is_flag=True, help="使用 Docling 解析（保留结构）")
@click.option("--work-dir", default=None, help="临时文件目录（C 盘空间不足时指定）")
@click.pass_context
def import_dir(ctx: click.Context, directory: str, docling: bool, work_dir: str | None) -> None:
    """批量导入目录下所有支持的文件。"""
    config = _get_config(ctx)
    config.use_docling = docling
    kb_instance = _get_kb(ctx)
    supported = (".txt", ".md", ".json", ".pdf", ".docx", ".zip")
    files = [p for p in Path(directory).iterdir() if p.suffix.lower() in supported]
    if not files:
        logger.info("目录下没有支持的文件")
        return
    for f in files:
        logger.info("正在导入 {}...", f.name)
        if f.suffix.lower() == ".zip":
            doc_ids = kb_instance.add_zip(f, work_dir=work_dir)
            logger.info("  + {} ({} 篇)", f.name, len(doc_ids))
        else:
            doc_id = kb_instance.add_file(str(f))
            logger.info("  + {} -> {}", f.name, doc_id)


@kb.command()
@click.pass_context
def list_docs(ctx: click.Context) -> None:
    """列出知识库中的所有文档。"""
    kb_instance = _get_kb(ctx)
    docs = kb_instance.list_documents()
    if not docs:
        logger.info("知识库为空")
        return
    logger.info("知识库文档列表:")
    for d in docs:
        logger.info("  {} | {} | {}", d.id, d.title, d.source)


@kb.command()
@click.argument("query")
@click.option("--top-k", default=5, help="返回结果数")
@click.pass_context
def search(ctx: click.Context, query: str, top_k: int) -> None:
    """检索知识库。"""
    kb_instance = _get_kb(ctx)
    logger.info("正在检索: {} (top-k={})", query, top_k)
    chunks = kb_instance.query(query, top_k=top_k)
    if not chunks:
        logger.info("未找到相关结果")
        return
    logger.info("搜索结果 (top-{}):", top_k)
    for c in chunks:
        preview = c.content[:80].replace("\n", " ")
        logger.info("  {:.3f} | {} | {}", c.score, c.document_id, preview)


@kb.command()
@click.pass_context
def seed(ctx: click.Context) -> None:
    """导入内置法律条文种子数据。"""
    kb_instance = _get_kb(ctx)
    laws = get_seed_laws()
    imported = 0
    for law in laws:
        logger.info("正在导入 {}...", law.title)
        existing = kb_instance.list_documents()
        if any(d.title == law.title for d in existing):
            logger.info("跳过（已存在）: {}", law.title)
            continue
        kb_instance.add_text(title=law.title, content=law.content)
        logger.info("  + {}", law.title)
        imported += 1
    logger.info("导入完成: {} 部法律", imported)


@kb.group(name="eval")
def kb_eval() -> None:
    """RAG 检索质量评估。"""


@kb_eval.command()
@click.option("--queries-per-chunk", default=2, help="每个 chunk 生成的问题数")
@click.option(
    "--sample",
    default=100.0,
    type=float,
    help="chunk 采样百分比（0-100），小于 100 则随机采样快速生成小测试集",
)
@click.option("--seed", default=None, type=int, help="随机种子，相同 seed 保证采样可复现")
@click.option("--output", default=None, help="输出数据集路径（启用断点续增）")
@click.pass_context
def generate(
    ctx: click.Context,
    queries_per_chunk: int,
    sample: float,
    seed: int | None,
    output: str | None,
) -> None:
    """从知识库 chunk 自动生成评估数据集。
    支持断点继续：重复执行同一 --output 会自动跳过已处理 chunk。"""
    config = _get_config(ctx)
    kb_instance = _get_kb(ctx)
    llm = LLMClient(config.llm)
    from harness.eval_rag.generator import RagDatasetGenerator

    generator = RagDatasetGenerator()
    path = output or str(Path(config.data_dir) / "rag_eval_dataset.jsonl")
    generator.generate(
        kb_instance,
        llm,
        queries_per_chunk=queries_per_chunk,
        sample_percent=sample,
        seed=seed,
        output_path=path,
    )


@kb_eval.command(name="run")
@click.argument("dataset", type=click.Path(exists=True))
@click.option("--top-ks", default="1,3,5", help="评估的 K 值，逗号分隔")
@click.option(
    "--expansion-threshold",
    default=0.6,
    type=float,
    help="AI 扩展阈值（低于此值用 LLM 扩展同义查询，设 0 禁用扩展）",
)
@click.option(
    "--checkpoint",
    type=click.Path(),
    default=None,
    help="断点续跑文件路径（JSONL，每行一条 query→retrieved 记录）；"
    "提供时跳过已完成的 query，新结果增量追加",
)
@click.pass_context
def eval_run(
    ctx: click.Context,
    dataset: str,
    top_ks: str,
    expansion_threshold: float,
    checkpoint: str | None,
) -> None:
    """执行 RAG 检索质量评估。"""
    kb_instance = _get_kb(ctx)
    from harness.eval_rag.dataset import load_jsonl
    from harness.eval_rag.reporter import RagEvalReporter
    from harness.eval_rag.runner import RagEvalRunner

    items = load_jsonl(dataset)
    top_ks_list = [int(k.strip()) for k in top_ks.split(",")]
    runner = RagEvalRunner()
    result = runner.run(
        kb_instance,
        items,
        top_ks=top_ks_list,
        dataset_name=Path(dataset).stem,
        expansion_threshold=expansion_threshold,
        checkpoint_path=checkpoint,
    )
    reporter = RagEvalReporter()
    logger.info("RAG 评测结果:\n{}", reporter.to_markdown(result))
