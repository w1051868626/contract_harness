"""调试：harness eval report — 查看评测报告目录"""

from harness.cli.main import load_dotenv
from harness.core.config import HarnessConfig

load_dotenv()
config = HarnessConfig()
print(f"报告目录: {config.report_dir}")
print("使用 debug_eval_run.py 运行评测后自动生成报告")
