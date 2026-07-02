"""调试：harness replay — 回放指定审查会话"""
import argparse
import json

from harness.cli.main import load_dotenv
from harness.core.config import HarnessConfig
from harness.replay.player import SessionPlayer
from harness.replay.storage import ReplayStorage

load_dotenv()
config = HarnessConfig()

parser = argparse.ArgumentParser(description="回放指定审查会话")
parser.add_argument("--session-id", required=True, help="会话 ID")
args = parser.parse_args()

player = SessionPlayer(ReplayStorage(config.replay_dir))
session = player.load(args.session_id)

if session is None:
    print("会话不存在")
else:
    print(f"合同: {session.document.title}")
    print(f"时间: {session.started_at}")
    print(f"步骤数: {len(session.steps)}")
    for step in session.steps:
        print(f"  Step {step.step_index}: {step.agent_message[:80]}")
