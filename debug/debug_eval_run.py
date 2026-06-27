"""调试：harness eval run — 在数据集上运行评测"""
from harness.cli.main import load_dotenv
from harness.core.config import HarnessConfig
from harness.eval.dataset import EvalDataset
from harness.eval.scorer import EvalScorer
from harness.eval.reporters import EvalReporter

load_dotenv()
config = HarnessConfig()

dataset = input("数据集路径: ")
ds = EvalDataset()
ds.load(dataset)

scorer = EvalScorer()
data = scorer.score(ds)

print("评测完成")
for name, value in data.get("aggregated_metrics", {}).items():
    print(f"  {name}: {value*100:.1f}%")

reporter = EvalReporter(config.report_dir)
md = reporter.report_markdown(data)
html = reporter.report_html(data)
print(f"Markdown: {md}")
print(f"HTML:     {html}")
