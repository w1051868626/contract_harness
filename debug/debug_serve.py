"""调试：harness serve — 启动 Web 界面"""
import argparse
import uvicorn

from harness.core.config import HarnessConfig
from harness.utils.log import setup_logging
from harness.web.app import app

parser = argparse.ArgumentParser(description="启动 Web 界面")
parser.add_argument("--host", default="127.0.0.1", help="监听地址 (默认: 127.0.0.1)")
parser.add_argument("--port", type=int, default=8000, help="监听端口 (默认: 8000)")
args = parser.parse_args()

setup_logging(log_dir=HarnessConfig().log_dir)
print(f"启动 Web 界面: http://{args.host}:{args.port}")
uvicorn.run(app, host=args.host, port=args.port)
