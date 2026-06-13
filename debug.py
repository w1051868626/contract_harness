#!/usr/bin/env python3
"""调试入口 — 在 PyCharm 中以此文件作为 Script path 运行，Parameters 填入 CLI 参数即可下断点调试。"""

import sys
from pathlib import Path

# 确保项目根目录在 Python 路径中
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from harness.cli.main import cli

if __name__ == "__main__":
    cli()
