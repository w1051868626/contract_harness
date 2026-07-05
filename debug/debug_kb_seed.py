"""调试：harness kb seed — 导入内置法律条文种子数据"""

from harness.cli.main import load_dotenv
from harness.core.config import HarnessConfig
from harness.rag.knowledge_base import KnowledgeBase
from harness.rag.seed_laws import get_seed_laws

load_dotenv()
config = HarnessConfig()
config.ensure_dirs()

kb = KnowledgeBase.from_config(config)
laws = get_seed_laws()
imported = 0

for law in laws:
    existing = kb.list_documents()
    if any(d.title == law.title for d in existing):
        print(f"跳过（已存在）: {law.title}")
        continue
    kb.add_text(title=law.title, content=law.content)
    print(f"  + {law.title}")
    imported += 1

print(f"导入完成: {imported} 部法律")
