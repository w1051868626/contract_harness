"""使用 Chinese-Laws 数据集测试 RAG 切片效果

用法:
    conda activate contract-harness
    # 从项目根目录运行：
    python -m debug.test_chunking.demo_chinese_laws

首次运行会自动从 ModelScope 下载数据集（约 1.5MB）。
"""

import subprocess
import sys
import zipfile
from pathlib import Path

from harness.rag.knowledge_base import KnowledgeBase

# ---- 配置 ----
_SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = _SCRIPT_DIR / "chinese-laws"
CHUNK_SIZE = 200
OVERLAP = 30


def download_dataset():
    """从 ModelScope 下载 Chinese-Laws 数据集（仅首次需要）。"""
    if DATASET_DIR.exists():
        return DATASET_DIR / "chinese_laws.txt"

    import subprocess

    print("正在下载 Chinese-Laws 数据集...")
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://www.modelscope.cn/datasets/dengcao/Chinese-Laws.git",
            str(DATASET_DIR),
        ],
        check=True,
        capture_output=True,
    )

    zip_path = DATASET_DIR / "Chinese-Laws.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"下载失败：{zip_path} 不存在")

    # Git LFS：检查是否为指针文件（读前 200 字节，UTF-8 解码即可）
    header = zip_path.read_bytes()[:200]
    if header.startswith(b"version https://git-lfs"):
        subprocess.run(
            ["git", "lfs", "pull"],
            cwd=str(DATASET_DIR),
            check=True,
            capture_output=True,
        )

    # 解压（重读，LFS pull 后文件已被替换为真实 zip）
    out_path = DATASET_DIR / "chinese_laws.txt"
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        # 取第一个 .txt 文件
        txt_file = next((n for n in names if n.endswith(".txt")), names[0])
        out_path.write_bytes(z.read(txt_file))

    print(f"数据集已下载到 {out_path}")
    return out_path


def run_chunking_demo(file_path: Path):
    """对数据集中每部法律切片并统计。"""
    text = file_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    print(f"总行数: {len(lines)}")
    print(f"总字符: {len(text)}")

    # 数据集格式：每行是一条法律条文，如
    # 《中华人民共和国民法典》第八条规定，民事主体从事民事活动，不得违反法律，不得违背公序良俗。
    doc_id = "chinese-laws"

    # 整体文本切片（模拟整库导入）
    print(f"\n{'=' * 60}")
    print("  方式一：全文一次性切片 (_chunk_legal_text)")
    print(f"{'=' * 60}")
    legal = KnowledgeBase.chunk_legal_text(text, doc_id, CHUNK_SIZE, OVERLAP)
    if legal:
        _report_chunks(legal, "legal_text")
    else:
        print("  → _chunk_legal_text 未命中（无语条模式），降级 _chunk_text：")
        txt = KnowledgeBase.chunk_text(text, doc_id, CHUNK_SIZE, OVERLAP)
        _report_chunks(txt, "text")

    # 按行切片（每条法律独立 entry，模拟 kb.add_text）
    print(f"\n{'=' * 60}")
    print("  方式二：逐条切片 (模拟 add_text 逐个添加)")
    print(f"{'=' * 60}")
    all_chunks = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        chunks = KnowledgeBase.chunk_legal_text(line, doc_id, CHUNK_SIZE, OVERLAP)
        if chunks is None:
            chunks = KnowledgeBase.chunk_text(line, doc_id, CHUNK_SIZE, OVERLAP)
        all_chunks.extend(chunks)
    _report_chunks(all_chunks, "per-line")


def _report_chunks(chunks, label: str):
    """打印切片统计报告。"""
    if not chunks:
        print("  无切片")
        return

    total_chars = sum(len(c.content) for c in chunks)
    max_chars = max(len(c.content) for c in chunks)
    min_chars = min(len(c.content) for c in chunks)

    # 统计结尾对齐率
    end_aligned = sum(1 for c in chunks if c.content.rstrip()[-1:] in "。！？；.!?；")
    # 统计开头截断（以非句子开头开头，且不是元数据行）
    start_cut = sum(
        1
        for c in chunks
        if c.content and c.content[0] not in "《第\u201c\u2018" and not c.content[0].isalpha()
    )

    print(f"  切片数:      {len(chunks)}")
    print(f"  平均长度:    {total_chars // len(chunks)} chars")
    print(f"  最短/最长:   {min_chars} / {max_chars} chars")
    print(f"  结尾句柄对齐: {end_aligned}/{len(chunks)} ({end_aligned / len(chunks) * 100:.0f}%)")
    print(f"  可能开头截断: {start_cut}")

    # 展示前 3 个切片
    print(f"\n  前 3 个切片预览:")
    for i, c in enumerate(chunks[:3]):
        preview = c.content[:100].replace("\n", " ")
        print(f"    [{i}] ({len(c.content)} chars) {preview}...")

    # 检查重叠
    overlap_ok = 0
    for i in range(1, len(chunks)):
        prev_tail = (
            chunks[i - 1].content[-OVERLAP:]
            if len(chunks[i - 1].content) >= OVERLAP
            else chunks[i - 1].content
        )
        if prev_tail and prev_tail in chunks[i].content:
            overlap_ok += 1
    print(f"  overlap 携带: {overlap_ok}/{len(chunks) - 1}")


if __name__ == "__main__":
    data_file = download_dataset()
    run_chunking_demo(data_file)
