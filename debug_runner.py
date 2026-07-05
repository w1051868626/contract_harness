#!/usr/bin/env python3
"""调试入口 — 在 PyCharm 中以此文件作为 Script path 运行。

Parameters 填入 CLI 参数即可下断点调试。
"""

import sys
from pathlib import Path
from typing import cast

import click

# 确保项目根目录在 Python 路径中
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from harness.cli.main import cli  # noqa: E402

if __name__ == "__main__":
    # cli 经 @click.group() 装饰后是 Group 实例，参数（ctx/verbose）由
    # click 运行时从 sys.argv 注入而非形参传递；用 cast 显式标注为 Group
    # 类型，绕过 IDE 对原函数签名 ``cli(ctx, verbose) -> None`` 的推断，
    # 避免「形参未填」/「找不到 main」静态告警。
    cli_group = cast(click.Group, cli)
    cli_group.main(prog_name="harness", standalone_mode=False)
