"""调试：harness eval run — 在数据集上运行评测"""

import argparse

from harness.cli.main import load_dotenv
from harness.core.config import HarnessConfig
from harness.eval.dataset import EvalDataset
from harness.eval.reporters import EvalReporter
from harness.eval.scorer import EvalScorer

load_dotenv()
config = HarnessConfig()

parser = argparse.ArgumentParser(description="在数据集上运行评测")
parser.add_argument("--dataset", required=True, help="数据集路径")
args = parser.parse_args()

ds = EvalDataset()
ds.load(args.dataset)

scorer = EvalScorer()
data = scorer.score(ds)

print("评测完成")
for name, value in data.get("aggregated_metrics", {}).items():
    print(f"  {name}: {value * 100:.1f}%")

reporter = EvalReporter(config.report_dir)
md = reporter.report_markdown(data)
html = reporter.report_html(data)
print(f"Markdown: {md}")
print(f"HTML:     {html}")
