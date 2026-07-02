"""调试：harness review — 审查一份合同"""
import argparse
from pathlib import Path

from harness.cli.main import load_dotenv
from harness.core.config import HarnessConfig
from harness.core.types import ContractDocument
from harness.agent.contract_agent import ContractAgent
from harness.agent.llm import LLMClient
from harness.utils.io import read_text

load_dotenv()
config = HarnessConfig()

parser = argparse.ArgumentParser(description="审查一份合同")
parser.add_argument("--file", default="examples/contracts/sample.md", help="合同文件路径")
parser.add_argument("--model", default="", help="LLM 模型名称")
args = parser.parse_args()

content = read_text(args.file)
doc = ContractDocument(id=Path(args.file).stem, title=Path(args.file).name, content=content)

agent = ContractAgent(LLMClient(config.llm))
report, session = agent.review(doc)

print(f"会话ID: {session.session_id}")
print(f"整体风险: {report.overall_risk.value}")
print(f"摘要: {report.summary[:200]}")
for c in report.clauses:
    print(f"  [{c.risk.value}] {c.clause_type}: {c.comment[:80]}")
