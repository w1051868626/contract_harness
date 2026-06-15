"""contract-harness 命令行入口，支持审查、回放、评测、回归与 Web 服务。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import uvicorn
from rich.console import Console
from rich.table import Table

from harness.agent.contract_agent import ContractAgent
from harness.agent.llm import LLMClient
from harness.core.config import HarnessConfig
from harness.core.types import ContractDocument
from harness.eval.dataset import EvalDataset
from harness.eval.reporters import EvalReporter
from harness.eval.scorer import EvalScorer
from harness.rag.knowledge_base import KnowledgeBase
from harness.rag.seed_laws import get_seed_laws
from harness.regression.comparator import OutputComparator
from harness.regression.suite import RegressionSuite
from harness.replay.player import SessionPlayer
from harness.replay.recorder import SessionRecorder
from harness.replay.storage import ReplayStorage
from harness.utils.io import load_dotenv, read_text

load_dotenv()

console = Console()


def _existing_path(value: str) -> str:
    p = Path(value)
    if not p.exists():
        raise argparse.ArgumentTypeError(f"路径不存在: {value}")
    return value


def _existing_dir(value: str) -> str:
    p = Path(value)
    if not p.is_dir():
        raise argparse.ArgumentTypeError(f"目录不存在: {value}")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="harness", description="合同审查 Agent 系统 CLI")

    sub = parser.add_subparsers(dest="command")
    _parent = argparse.ArgumentParser(add_help=False)
    _parent.add_argument("--verbose", "-v", action="store_true", help="启用详细输出")

    # ---- review ----
    p = sub.add_parser("review", parents=[_parent], help="审查一份合同并展示结果")
    p.add_argument("contract_file", type=_existing_path, metavar="FILE")
    p.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否保存回放记录",
    )
    p.add_argument("--model", default="", help="LLM 模型名称")
    p.set_defaults(func=_cmd_review)

    # ---- replay ----
    p = sub.add_parser("replay", parents=[_parent], help="回放指定审查会话")
    p.add_argument("session_id")
    p.add_argument("--json", dest="as_json", action="store_true", help="以 JSON 格式输出")
    p.set_defaults(func=_cmd_replay)

    # ---- sessions ----
    p = sub.add_parser("sessions", parents=[_parent], help="列出所有回放会话")
    p.add_argument("--limit", type=int, default=20, help="显示最近的会话数量")
    p.set_defaults(func=_cmd_sessions)

    # ---- eval ----
    p = sub.add_parser("eval", parents=[_parent], help="评测命令组")
    eval_sub = p.add_subparsers(dest="eval_command")

    pe = eval_sub.add_parser("run", help="在指定数据集上运行评测并生成报告")
    pe.add_argument("dataset", type=_existing_path)
    pe.set_defaults(func=_cmd_eval_run)

    pe = eval_sub.add_parser("report", help="展示评测报告相关信息")
    pe.add_argument("--output", default="eval_report", help="报告文件名")
    pe.set_defaults(func=_cmd_eval_report)

    # ---- regression ----
    p = sub.add_parser("regression", parents=[_parent], help="回归测试命令组")
    reg_sub = p.add_subparsers(dest="regression_command")

    pr = reg_sub.add_parser("run", help="运行回归测试并与基线对比")
    pr.add_argument("dataset", type=_existing_path)
    pr.add_argument("--version", default="", help="当前版本标识")
    pr.set_defaults(func=_cmd_regression_run)

    pr = reg_sub.add_parser("diff", help="对比两个会话的审查差异")
    pr.add_argument("session_a")
    pr.add_argument("session_b")
    pr.set_defaults(func=_cmd_regression_diff)

    # ---- serve ----
    p = sub.add_parser("serve", help="启动 FastAPI Web 界面")
    p.add_argument("--host", default="127.0.0.1", help="监听地址")
    p.add_argument("--port", type=int, default=8000, help="监听端口")
    p.add_argument("--reload", action="store_true", help="热重载")
    p.set_defaults(func=_cmd_serve)

    # ---- kb ----
    p = sub.add_parser("kb", parents=[_parent], help="知识库管理命令组")
    kb_sub = p.add_subparsers(dest="kb_command")

    pk = kb_sub.add_parser("import-file", help="将单个文件导入知识库")
    pk.add_argument("file_path", type=_existing_path)
    pk.set_defaults(func=_cmd_kb_import_file)

    pk = kb_sub.add_parser("import-dir", help="批量导入目录下所有支持的文件")
    pk.add_argument("directory", type=_existing_dir)
    pk.set_defaults(func=_cmd_kb_import_dir)

    pk = kb_sub.add_parser("list", help="列出知识库中的所有文档")
    pk.set_defaults(func=_cmd_kb_list)

    pk = kb_sub.add_parser("search", help="检索知识库")
    pk.add_argument("query")
    pk.add_argument("--top-k", type=int, default=5, dest="top_k", help="返回结果数")
    pk.set_defaults(func=_cmd_kb_search)

    pk = kb_sub.add_parser("seed", help="导入内置法律条文种子数据")
    pk.set_defaults(func=_cmd_kb_seed)

    return parser


def _get_config(args: argparse.Namespace) -> HarnessConfig:
    config = HarnessConfig()
    verbose = getattr(args, "verbose", False)
    config.verbose = verbose
    if verbose:
        config.ensure_dirs()
    return config


def _cmd_review(args: argparse.Namespace) -> None:
    config = _get_config(args)
    filepath = Path(args.contract_file)
    content = read_text(filepath)
    document = ContractDocument(
        id=filepath.stem,
        title=filepath.name,
        content=content,
        file_path=str(filepath.absolute()),
    )
    if args.model:
        config.llm.model = args.model
    with console.status("正在审查合同..."):
        agent = ContractAgent(LLMClient(config.llm))
        report, session = agent.review(document)
    console.print(f"\n[bold green]审查完成:[/bold green] {filepath.name}")
    console.print(f"[bold]会话 ID:[/bold] {session.session_id}")
    console.print(f"[bold]整体风险:[/bold] [red]{report.overall_risk.value.upper()}[/red]")
    console.print(f"\n[bold]审查摘要:[/bold]\n{report.summary[:500]}...")
    table = Table(title="条款概览")
    table.add_column("类型", style="cyan")
    table.add_column("风险", style="magenta")
    table.add_column("备注")
    for c in report.clauses:
        table.add_row(c.clause_type, c.risk.value, c.comment[:40] if c.comment else "")
    console.print(table)
    if args.save:
        recorder = SessionRecorder(config.replay_dir)
        path = recorder.record(session)
        console.print(f"\n回放已保存: {path}")


def _cmd_replay(args: argparse.Namespace) -> None:
    config = _get_config(args)
    player = SessionPlayer(ReplayStorage(config.replay_dir))
    session = player.load(args.session_id)
    if session is None:
        console.print(f"[red]会话 {args.session_id} 不存在[/red]")
        return
    if args.as_json:
        r = SessionRecorder()
        data = r._serialize(session)
        console.print(json.dumps(data, ensure_ascii=False, indent=2))
        return
    console.print(f"[bold]回放会话:[/bold] {session.session_id}")
    console.print(f"[bold]合同:[/bold] {session.document.title}")
    console.print(f"[bold]时间:[/bold] {session.started_at}")
    for step in session.steps:
        console.print(f"\n[cyan]Step {step.step_index}[/cyan]: {step.agent_message}")
        for tc in step.tool_calls:
            console.print(f"  ⚙ {tc.tool_name} ({len(str(tc.input))} chars)")


def _cmd_sessions(args: argparse.Namespace) -> None:
    config = _get_config(args)
    player = SessionPlayer(ReplayStorage(config.replay_dir))
    sessions_list = player.list_sessions()
    if not sessions_list:
        console.print("[yellow]暂无回放记录[/yellow]")
        return
    table = Table(title="回放会话列表")
    table.add_column("会话 ID", style="cyan")
    table.add_column("合同", style="green")
    table.add_column("时间", style="white")
    for s in sessions_list[: args.limit]:
        table.add_row(s["session_id"], s["document_title"], s["started_at"])
    console.print(table)


def _cmd_eval_run(args: argparse.Namespace) -> None:
    config = _get_config(args)
    ds = EvalDataset()
    ds.load(args.dataset)
    with console.status("正在运行评测..."):
        scorer = EvalScorer()
        data = scorer.score(ds)
    console.print("\n[bold green]评测完成[/bold green]")
    for name, value in data.get("aggregated_metrics", {}).items():
        pct = value * 100
        color = "green" if pct >= 80 else "yellow" if pct >= 60 else "red"
        console.print(f"  {name}: [{color}]{pct:.1f}%[/{color}]")
    reporter = EvalReporter(config.report_dir)
    md_path = reporter.report_markdown(data)
    html_path = reporter.report_html(data)
    console.print("\n报告已生成:")
    console.print(f"  Markdown: {md_path}")
    console.print(f"  HTML:     {html_path}")


def _cmd_eval_report(args: argparse.Namespace) -> None:
    config = _get_config(args)
    console.print(f"[yellow]报告目录: {config.report_dir}[/yellow]")
    console.print("[yellow]使用 'eval run' 运行评测后会自动生成报告[/yellow]")


def _cmd_regression_run(args: argparse.Namespace) -> None:
    ds = EvalDataset()
    ds.load(args.dataset)
    with console.status("正在运行回归测试..."):
        suite = RegressionSuite()
        result = suite.run(ds, args.version)
    if result.passed:
        console.print("[bold green]回归测试通过 ✓[/bold green]")
    else:
        console.print("[bold red]回归测试失败 ✗[/bold red]")
    if result.improvements:
        console.print("\n[green]改进:[/green]")
        for imp in result.improvements:
            console.print(f"  ✓ {imp}")
    if result.regressions:
        console.print("\n[red]回归:[/red]")
        for reg in result.regressions:
            console.print(f"  ✗ {reg}")
    if result.metrics_diff:
        console.print("\n[bold]指标变化:[/bold]")
        for name, diff in result.metrics_diff.items():
            color = "green" if diff >= 0 else "red"
            console.print(f"  {name}: [{color}]{diff:+.2%}[/{color}]")


def _cmd_regression_diff(args: argparse.Namespace) -> None:
    config = _get_config(args)
    comparator = OutputComparator(SessionPlayer(ReplayStorage(config.replay_dir)))
    with console.status("正在对比..."):
        diff_result = comparator.compare_by_session_id(args.session_a, args.session_b)
    console.print("[bold]对比结果:[/bold]")
    if diff_result.get("risk_level_changed"):
        console.print("  [red]风险等级发生变化[/red]")
    if diff_result.get("summary_changed"):
        console.print("  [yellow]摘要内容发生变化[/yellow]")
    if diff_result.get("clause_diffs"):
        console.print(f"  条款变化: {len(diff_result['clause_diffs'])} 处")
    if diff_result.get("risk_diffs"):
        console.print(f"  风险评估变化: {len(diff_result['risk_diffs'])} 处")
    if diff_result.get("compliance_diffs"):
        console.print(f"  合规检查变化: {len(diff_result['compliance_diffs'])} 处")


def _cmd_serve(args: argparse.Namespace) -> None:
    from harness.web.app import app

    console.print(f"[green]正在启动 Web 界面:[/green] http://{args.host}:{args.port}")
    if args.reload:
        console.print("[yellow]热重载已启用[/yellow]")
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


def _cmd_kb_import_file(args: argparse.Namespace) -> None:
    config = _get_config(args)
    config.ensure_dirs()
    kb_instance = KnowledgeBase.from_config(config)
    file_path = args.file_path
    with console.status("正在导入文件..."):
        if file_path.lower().endswith(".zip"):
            doc_ids = kb_instance._add_zip(Path(file_path))
            if doc_ids:
                console.print(f"[green]导入成功:[/green] {Path(file_path).name}")
                for did in doc_ids:
                    console.print(f"  [green]✓[/green] {did}")
            else:
                console.print("[red]导入失败[/red]")
        else:
            doc_id = kb_instance.add_file(file_path)
            if doc_id:
                console.print(f"[green]导入成功:[/green] {Path(file_path).name} → {doc_id}")
            else:
                console.print("[red]导入失败[/red]")


def _cmd_kb_import_dir(args: argparse.Namespace) -> None:
    config = _get_config(args)
    config.ensure_dirs()
    kb_instance = KnowledgeBase.from_config(config)
    supported = (".txt", ".md", ".json", ".pdf", ".docx", ".zip")
    files = [p for p in Path(args.directory).iterdir() if p.suffix.lower() in supported]
    if not files:
        console.print("[yellow]目录下没有支持的文件[/yellow]")
        return
    for f in files:
        with console.status(f"正在导入 {f.name}..."):
            if f.suffix.lower() == ".zip":
                doc_ids = kb_instance._add_zip(f)
                console.print(f"  [green]✓[/green] {f.name} ({len(doc_ids)} 篇)")
            else:
                doc_id = kb_instance.add_file(str(f))
                console.print(f"  [green]✓[/green] {f.name} → {doc_id}")


def _cmd_kb_list(args: argparse.Namespace) -> None:
    config = _get_config(args)
    config.ensure_dirs()
    kb_instance = KnowledgeBase.from_config(config)
    docs = kb_instance.list_documents()
    if not docs:
        console.print("[yellow]知识库为空[/yellow]")
        return
    table = Table(title="知识库文档列表")
    table.add_column("ID", style="cyan")
    table.add_column("标题", style="green")
    table.add_column("来源")
    table.add_column("创建时间", style="white")
    for d in docs:
        table.add_row(d.id, d.title, d.source, "")
    console.print(table)


def _cmd_kb_search(args: argparse.Namespace) -> None:
    config = _get_config(args)
    config.ensure_dirs()
    kb_instance = KnowledgeBase.from_config(config)
    with console.status("正在检索..."):
        chunks = kb_instance.query(args.query, top_k=args.top_k)
    if not chunks:
        console.print("[yellow]未找到相关结果[/yellow]")
        return
    table = Table(title=f"搜索结果（top-{args.top_k}）")
    table.add_column("得分", style="cyan")
    table.add_column("文档 ID")
    table.add_column("内容")
    for c in chunks:
        preview = c.content[:80].replace("\n", " ")
        table.add_row(f"{c.score:.3f}", c.document_id, preview)
    console.print(table)


def _cmd_kb_seed(args: argparse.Namespace) -> None:
    config = _get_config(args)
    config.ensure_dirs()
    kb_instance = KnowledgeBase.from_config(config)
    laws = get_seed_laws()
    imported = 0
    for law in laws:
        with console.status(f"正在导入 {law.title}..."):
            existing = kb_instance.list_documents()
            if any(d.title == law.title for d in existing):
                console.print(f"  [yellow]跳过（已存在）[/yellow] {law.title}")
                continue
            kb_instance.add_text(title=law.title, content=law.content)
            console.print(f"  [green]✓[/green] {law.title}")
            imported += 1
    console.print(f"[bold green]导入完成:[/bold green] {imported} 部法律")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
