"""调试：harness sessions — 列出所有回放会话"""

from harness.cli.main import load_dotenv
from harness.core.config import HarnessConfig
from harness.replay.player import SessionPlayer
from harness.replay.storage import ReplayStorage

load_dotenv()
config = HarnessConfig()

player = SessionPlayer(ReplayStorage(config.replay_dir))
sessions = player.list_sessions()

if not sessions:
    print("暂无回放记录")
else:
    print(f"共 {len(sessions)} 个会话:")
    for s in sessions[:20]:
        print(f"  {s['session_id']} | {s['document_title']} | {s['started_at']}")
