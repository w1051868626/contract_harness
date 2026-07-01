"""调试：harness serve — 启动 Web 界面"""

import uvicorn

from harness.core.config import HarnessConfig
from harness.utils.log import setup_logging
from harness.web.app import app

setup_logging(log_dir=HarnessConfig().log_dir)
print("启动 Web 界面: http://127.0.0.1:8000")
uvicorn.run(app, host="127.0.0.1", port=8000)
