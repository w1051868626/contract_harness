"""调试：harness regression run — 运行回归测试"""

import argparse
from datetime import datetime

from harness.cli.main import load_dotenv
from harness.eval.dataset import EvalDataset
from harness.regression.suite import RegressionSuite

load_dotenv()

parser = argparse.ArgumentParser(description="运行回归测试")
parser.add_argument("--dataset", required=True, help="数据集路径")
args = parser.parse_args()

ds = EvalDataset()
ds.load(args.dataset)

suite = RegressionSuite()
result = suite.run(ds, version=f"debug-{datetime.now():%Y%m%d}")

print(f"通过: {result.passed}")
if result.improvements:
    for imp in result.improvements:
        print(f"  改进: {imp}")
if result.regressions:
    for reg in result.regressions:
        print(f"  回归: {reg}")
if result.metrics_diff:
    for name, diff in result.metrics_diff.items():
        print(f"  {name}: {diff:+.2%}")
