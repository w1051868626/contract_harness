"""调试：harness kb search — 检索知识库"""
from harness.cli.main import load_dotenv
from harness.core.config import HarnessConfig
from harness.rag.knowledge_base import KnowledgeBase

load_dotenv()
config = HarnessConfig()
config.ensure_dirs()

kb = KnowledgeBase.from_config(config)
query = input("检索关键词: ")
top_k = int(input("返回结果数 (默认5): ") or "5")

chunks = kb.query(query, top_k=top_k)
if not chunks:
    print("未找到相关结果")
else:
    print(f"搜索结果 (top-{top_k}):")
    for c in chunks:
        preview = c.content[:80].replace("\n", " ")
        print(f"  {c.score:.3f} | {c.document_id} | {preview}")
