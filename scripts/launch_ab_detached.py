"""以真正脱离 shell 的方式启动 A/B 对比脚本。

MSYS bash 的 nohup/cmd start 在退出时会杀子进程，用 Python subprocess
DETACHED_PROCESS 标志启动独立进程，日志写文件，主脚本立即退出。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / ".harness" / "reports" / "multiquery_ab_200.log"
SCRIPT = ROOT / "scripts" / "compare_rerank_pool.py"

# Windows DETACHED_PROCESS = 0x00000008; CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NEW_PROCESS_GROUP = 0x00000200
DETACHED_PROCESS = 0x00000008

log_fp = open(LOG, "w", encoding="utf-8")
proc = subprocess.Popen(
    [sys.executable, "-u", str(SCRIPT), "--mode", "multiquery", "--n", "200"],
    cwd=str(ROOT),
    stdout=log_fp,
    stderr=subprocess.STDOUT,
    stdin=subprocess.DEVNULL,
    creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
    close_fds=True,
)
log_fp.close()

# 写 PID 文件
pid_file = ROOT / ".harness" / "reports" / "multiquery_ab_200.pid"
pid_file.write_text(str(proc.pid), encoding="utf-8")

print(f"Launched detached A/B, PID={proc.pid}")
print(f"Log: {LOG}")
