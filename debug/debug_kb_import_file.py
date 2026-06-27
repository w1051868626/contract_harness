"""调试：harness kb import-file — 导入单个文件到知识库"""
from pathlib import Path
from harness.cli.main import load_dotenv
from harness.core.config import HarnessConfig
from harness.rag.knowledge_base import KnowledgeBase

load_dotenv()
config = HarnessConfig()
config.ensure_dirs()

kb = KnowledgeBase.from_config(config)
file_path = input("文件路径: ")

if file_path.lower().endswith(".zip"):
    doc_ids = kb.add_zip(Path(file_path))
    print(f"导入 {len(doc_ids)} 篇文档" if doc_ids else "导入失败")
else:
    doc_id = kb.add_file(file_path)
    print(f"导入成功 -> {doc_id}" if doc_id else "导入失败")
