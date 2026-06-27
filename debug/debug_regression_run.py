"""调试：harness regression run — 运行回归测试"""
from harness.cli.main import load_dotenv
from harness.eval.dataset import EvalDataset
from harness.regression.suite import RegressionSuite

load_dotenv()

dataset = input("数据集路径: ")
ds = EvalDataset()
ds.load(dataset)

suite = RegressionSuite()
result = suite.run(ds, version=f"debug-{__import__('datetime').datetime.now():%Y%m%d}")

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
