"""CLI 命令行入口单元测试。"""

from __future__ import annotations

from click.testing import CliRunner

from harness.cli.main import cli


class TestCLI:
    """CLI 基本命令测试。"""

    def test_cli_help(self):
        """cli --help 应正常输出。"""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "合同审查 Agent 系统 CLI" in result.output

    def test_serve_help(self):
        """serve --help 应正常输出。"""
        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--help"])
        assert result.exit_code == 0
        assert "Web" in result.output

    def test_eval_help(self):
        """eval --help 应显示子命令。"""
        runner = CliRunner()
        result = runner.invoke(cli, ["eval", "--help"])
        assert result.exit_code == 0
        assert "评测" in result.output

    def test_eval_report(self):
        """eval report 应显示报告目录信息。"""
        runner = CliRunner()
        result = runner.invoke(cli, ["eval", "report"])
        assert result.exit_code == 0
        assert "报告目录" in result.output

    def test_regression_help(self):
        """regression --help 应显示子命令。"""
        runner = CliRunner()
        result = runner.invoke(cli, ["regression", "--help"])
        assert result.exit_code == 0
        assert "回归" in result.output

    def test_kb_help(self):
        """kb --help 应显示子命令。"""
        runner = CliRunner()
        result = runner.invoke(cli, ["kb", "--help"])
        assert result.exit_code == 0
        assert "知识库" in result.output

    def test_kb_list_help(self):
        """kb list-docs --help 应显示帮助信息。"""
        runner = CliRunner()
        result = runner.invoke(cli, ["kb", "list-docs", "--help"])
        assert result.exit_code == 0
        assert "列出" in result.output

    def test_sessions_empty(self):
        """sessions 在无记录时应正常输出。"""
        runner = CliRunner()
        result = runner.invoke(cli, ["sessions"])
        assert result.exit_code == 0

    def test_replay_nonexistent(self):
        """replay 不存在的会话应输出警告。"""
        runner = CliRunner()
        result = runner.invoke(cli, ["replay", "nonexistent_session"])
        assert "不存在" in result.output or result.exit_code == 0
