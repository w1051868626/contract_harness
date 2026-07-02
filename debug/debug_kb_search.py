"""调试：harness kb search — 检索知识库"""
import argparse

from harness.cli.main import load_dotenv
from harness.core.config import HarnessConfig
from harness.rag.knowledge_base import KnowledgeBase

load_dotenv()
config = HarnessConfig()
config.ensure_dirs()

parser = argparse.ArgumentParser(description="检索知识库")
parser.add_argument("--query", required=True, help="检索关键词")
parser.add_argument("--top-k", type=int, default=5, help="返回结果数 (默认: 5)")
args = parser.parse_args()

kb = KnowledgeBase.from_config(config)
chunks = kb.query(args.query, top_k=args.top_k)

if not chunks:
    print("未找到相关结果")
else:
    print(f"搜索结果 (top-{args.top_k}):")
    for c in chunks:
        preview = c.content[:80].replace("\n", " ")
        print(f"  {c.score:.3f} | {c.document_id} | {preview}")
