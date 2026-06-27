"""调试：harness kb list — 列出知识库中所有文档"""
from harness.cli.main import load_dotenv
from harness.core.config import HarnessConfig
from harness.rag.knowledge_base import KnowledgeBase

load_dotenv()
config = HarnessConfig()
config.ensure_dirs()

kb = KnowledgeBase.from_config(config)
docs = kb.list_documents()

if not docs:
    print("知识库为空")
else:
    print(f"共 {len(docs)} 篇文档:")
    for d in docs:
        print(f"  {d.id} | {d.title} | {d.source}")
