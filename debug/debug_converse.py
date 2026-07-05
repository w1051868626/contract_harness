"""调试：harness converse — 对已有会话继续追问"""

import argparse

from harness.agent.contract_agent import ContractAgent
from harness.agent.llm import LLMClient
from harness.cli.main import load_dotenv
from harness.core.config import HarnessConfig

load_dotenv()
config = HarnessConfig()

parser = argparse.ArgumentParser(description="对已有会话继续追问")
parser.add_argument("--session-id", required=True, help="会话 ID")
parser.add_argument("--query", required=True, help="追问内容")
args = parser.parse_args()

agent = ContractAgent(LLMClient(config.llm))
answer = agent.converse(args.session_id, args.query)
print(f"回答: {answer}")
