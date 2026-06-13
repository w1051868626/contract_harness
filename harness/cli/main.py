from __future__ import annotations

"""contract-harness 命令行入口，支持审查、回放、评测、回归与 Web 服务。"""

import json
from pathlib import Path

import click
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
from harness.regression.comparator import OutputComparator
from harness.regression.suite import RegressionSuite
from harness.replay.player import SessionPlayer
from harness.replay.recorder import SessionRecorder
from harness.replay.storage import ReplayStorage
from harness.utils.io import read_text

console = Console()


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="启用详细输出")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """合同审查 Agent 系统 CLI。"""
    ctx.ensure_object(dict)
    config = HarnessConfig()
    config.verbose = verbose
    if verbose:
        config.ensure_dirs()
    ctx.obj["config"] = config


@cli.command()
@click.argument("contract_file", type=click.Path(exists=True))
@click.option("--save/--no-save", default=True, help="是否保存回放记录")
@click.option("--model", default="", help="LLM 模型名称")
@click.pass_context
def review(ctx: click.Context, contract_file: str, save: bool, model: str) -> None:
    """审查一份合同并展示结果。"""
    config: HarnessConfig = ctx.obj["config"]
    filepath = Path(contract_file)
    content = read_text(filepath)

    document = ContractDocument(
        id=filepath.stem,
        title=filepath.name,
        content=content,
        file_path=str(filepath.absolute()),
    )

    if model:
        config.llm.model = model

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

    if save:
        recorder = SessionRecorder(config.replay_dir)
        path = recorder.record(session)
        console.print(f"\n回放已保存: {path}")


@cli.command()
@click.argument("session_id")
@click.option("--json", "as_json", is_flag=True, help="以 JSON 格式输出")
@click.pass_context
def replay(ctx: click.Context, session_id: str, as_json: bool) -> None:
    """回放指定审查会话。"""
    config: HarnessConfig = ctx.obj["config"]
    player = SessionPlayer(ReplayStorage(config.replay_dir))
    session = player.load(session_id)

    if session is None:
        console.print(f"[red]会话 {session_id} 不存在[/red]")
        return

    if as_json:
        from harness.replay.recorder import SessionRecorder

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


@cli.command()
@click.option("--limit", default=20, help="显示最近的会话数量")
@click.pass_context
def sessions(ctx: click.Context, limit: int) -> None:
    """列出所有回放会话。"""
    config: HarnessConfig = ctx.obj["config"]
    player = SessionPlayer(ReplayStorage(config.replay_dir))
    sessions_list = player.list_sessions()

    if not sessions_list:
        console.print("[yellow]暂无回放记录[/yellow]")
        return

    table = Table(title="回放会话列表")
    table.add_column("会话 ID", style="cyan")
    table.add_column("合同", style="green")
    table.add_column("时间", style="white")

    for s in sessions_list[:limit]:
        table.add_row(s["session_id"], s["document_title"], s["started_at"])
    console.print(table)


@cli.group()
def eval() -> None:
    """评测命令组。"""


@eval.command()
@click.argument("dataset", type=click.Path(exists=True))
@click.pass_context
def run(ctx: click.Context, dataset: str) -> None:
    """在指定数据集上运行评测并生成报告。"""
    config: HarnessConfig = ctx.obj["config"]
    ds = EvalDataset()
    ds.load(dataset)

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


@eval.command()
@click.option("--output", default="eval_report", help="报告文件名")
@click.pass_context
def report(ctx: click.Context, output: str) -> None:
    """展示评测报告相关信息。"""
    config: HarnessConfig = ctx.obj["config"]
    console.print(f"[yellow]报告目录: {config.report_dir}[/yellow]")
    console.print("[yellow]使用 'eval run' 运行评测后会自动生成报告[/yellow]")


@cli.group()
def regression() -> None:
    """回归测试命令组。"""


@regression.command(name="run")
@click.argument("dataset", type=click.Path(exists=True))
@click.option("--version", default="", help="当前版本标识")
@click.pass_context
def regression_run(ctx: click.Context, dataset: str, version: str) -> None:
    """运行回归测试并与基线对比。"""
    ds = EvalDataset()
    ds.load(dataset)

    with console.status("正在运行回归测试..."):
        suite = RegressionSuite()
        result = suite.run(ds, version)

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


@regression.command()
@click.argument("session_a")
@click.argument("session_b")
@click.pass_context
def diff(ctx: click.Context, session_a: str, session_b: str) -> None:
    """对比两个会话的审查差异。"""
    config: HarnessConfig = ctx.obj["config"]
    comparator = OutputComparator(SessionPlayer(ReplayStorage(config.replay_dir)))

    with console.status("正在对比..."):
        diff_result = comparator.compare_by_session_id(session_a, session_b)

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


@cli.command()
@click.option("--host", default="127.0.0.1", help="监听地址")
@click.option("--port", default=8000, help="监听端口")
@click.option("--reload", is_flag=True, help="热重载")
def serve(host: str, port: int, reload: bool) -> None:
    """启动 FastAPI Web 界面。"""
    from harness.web.app import app

    console.print(f"[green]正在启动 Web 界面:[/green] http://{host}:{port}")
    if reload:
        console.print("[yellow]热重载已启用[/yellow]")
    uvicorn.run(app, host=host, port=port, reload=reload)
