"""调试：harness converse — 对已有会话继续追问"""
from harness.cli.main import load_dotenv
from harness.core.config import HarnessConfig
from harness.agent.contract_agent import ContractAgent
from harness.agent.llm import LLMClient

load_dotenv()
config = HarnessConfig()

agent = ContractAgent(LLMClient(config.llm))
session_id = input("会话ID: ")
query = input("追问内容: ")
answer = agent.converse(session_id, query)
print(f"回答: {answer}")
