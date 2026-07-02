"""调试：harness regression diff — 对比两个会话"""
import argparse

from harness.cli.main import load_dotenv
from harness.core.config import HarnessConfig
from harness.regression.comparator import OutputComparator
from harness.replay.player import SessionPlayer
from harness.replay.storage import ReplayStorage

load_dotenv()
config = HarnessConfig()

parser = argparse.ArgumentParser(description="对比两个会话")
parser.add_argument("--a", required=True, help="会话 A")
parser.add_argument("--b", required=True, help="会话 B")
args = parser.parse_args()

comparator = OutputComparator(SessionPlayer(ReplayStorage(config.replay_dir)))
result = comparator.compare_by_session_id(args.a, args.b)

if result.get("risk_level_changed"):
    print("  风险等级发生变化")
if result.get("summary_changed"):
    print("  摘要内容发生变化")
print(f"  条款变化: {len(result.get('clause_diffs', []))} 处")
print(f"  风险评估变化: {len(result.get('risk_diffs', []))} 处")
print(f"  合规检查变化: {len(result.get('compliance_diffs', []))} 处")
