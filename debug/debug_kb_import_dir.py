"""调试：harness kb import-dir — 批量导入目录下所有文件"""
import argparse
from pathlib import Path

from harness.cli.main import load_dotenv
from harness.core.config import HarnessConfig
from harness.rag.knowledge_base import KnowledgeBase

load_dotenv()
config = HarnessConfig()
config.ensure_dirs()

parser = argparse.ArgumentParser(description="批量导入目录下所有文件")
parser.add_argument("--dir", required=True, help="目录路径")
args = parser.parse_args()

kb = KnowledgeBase.from_config(config)
directory = Path(args.dir)
supported = (".txt", ".md", ".json", ".pdf", ".docx", ".zip")

files = [p for p in directory.iterdir() if p.suffix.lower() in supported]
if not files:
    print("没有支持的文件")
else:
    for f in files:
        if f.suffix.lower() == ".zip":
            doc_ids = kb.add_zip(f)
            print(f"  {f.name}: {len(doc_ids)} 篇")
        else:
            doc_id = kb.add_file(str(f))
            print(f"  {f.name} -> {doc_id}")
