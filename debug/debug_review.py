"""调试：harness review — 审查一份合同"""
from pathlib import Path
from harness.cli.main import load_dotenv
from harness.core.config import HarnessConfig
from harness.core.types import ContractDocument
from harness.agent.contract_agent import ContractAgent
from harness.agent.llm import LLMClient
from harness.utils.io import read_text

load_dotenv()
config = HarnessConfig()

content = read_text("examples/contracts/sample.md")
doc = ContractDocument(id="sample", title="sample.md", content=content)

agent = ContractAgent(LLMClient(config.llm))
report, session = agent.review(doc)

print(f"会话ID: {session.session_id}")
print(f"整体风险: {report.overall_risk.value}")
print(f"摘要: {report.summary[:200]}")
for c in report.clauses:
    print(f"  [{c.risk.value}] {c.clause_type}: {c.comment[:80]}")
