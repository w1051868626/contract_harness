"""文件切片测试：读取 zip/docx/txt/md/pdf 文件，解析并展示切片结果

用法:
    python -m debug.test_chunking.chunk_file <path> [--chunk-size 512] [--overlap 64]
    python -m debug.test_chunking.chunk_file contract.zip
    python -m debug.test_chunking.chunk_file contract.docx --chunk-size 256
"""
import argparse
from pathlib import Path

from harness.rag.knowledge_base import KnowledgeBase


def fmt_meta(meta: dict) -> str:
    if not meta:
        return "(无)"
    parts = []
    for k, v in meta.items():
        label = k.value if hasattr(k, "value") else k
        parts.append(f"{label}={v}")
    return ", ".join(parts)


def chunk_text(text: str, source: str, chunk_size: int, overlap: int):
    results = []
    src = Path(source)

    for chunker, tag in [
        (KnowledgeBase._chunk_markdown, "_chunk_markdown"),
        (KnowledgeBase._chunk_law_text, "_chunk_law_text"),
        (KnowledgeBase._chunk_legal_text, "_chunk_legal_text"),
    ]:
        ck = chunker(text, src.name, chunk_size, overlap)
        if ck:
            results.append((tag, ck))
            return results

    ck = KnowledgeBase._chunk_text(text, src.name, chunk_size, overlap)
    results.append(("_chunk_text", ck))
    return results


def display_chunks(tag: str, chunks, label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  分块器: {tag}  |  切片数: {len(chunks)}")
    print(f"{'='*60}")
    for i, c in enumerate(chunks):
        meta = fmt_meta(c.metadata or {})
        content_preview = c.content[:150].replace("\n", " ").replace("\r", "")
        print(f"\n  [{i:2d}] ({len(c.content):,} chars)  {meta}")
        print(f"        {content_preview}")
    end_ok = sum(1 for c in chunks if c.content.rstrip()[-1:] in "。！？；.!?；")
    print(f"\n  结尾句柄对齐: {end_ok}/{len(chunks)}")


def process_file(file_path: Path, chunk_size: int, overlap: int):
    ext = file_path.suffix.lower()
    source_name = file_path.name

    if ext == ".zip":
        print(f"\n{'#'*60}")
        print(f"  ZIP 文件: {file_path}")
        print(f"{'#'*60}")
        entries = KnowledgeBase._extract_zip_texts(file_path)
        print(f"  成功解析: {len(entries)} 个文件")
        for inner_name, text in entries:
            results = chunk_text(text, inner_name, chunk_size, overlap)
            for tag, chunks in results:
                shorten = inner_name[:80]
                display_chunks(tag, chunks, f"{shorten} ({len(text):,} chars)")

    elif ext == ".docx":
        text = KnowledgeBase._parse_file(file_path)
        if not text.strip():
            print("(空)")
            return
        results = chunk_text(text, source_name, chunk_size, overlap)
        for tag, chunks in results:
            display_chunks(tag, chunks, f"{source_name} ({len(text):,} chars)")

    elif ext in (".txt", ".md", ".json", ".pdf"):
        text = KnowledgeBase._parse_file(file_path)
        if not text.strip():
            print("(空)")
            return
        results = chunk_text(text, source_name, chunk_size, overlap)
        for tag, chunks in results:
            display_chunks(tag, chunks, f"{source_name} ({len(text):,} chars)")

    else:
        print(f"不支持的文件类型: {ext}")
        print(f"支持: .zip, .docx, .txt, .md, .json, .pdf")


def main():
    parser = argparse.ArgumentParser(
        description="文件切片测试工具 — 解析文件并展示切片结果"
    )
    parser.add_argument("file", type=Path, help="输入文件路径（zip/docx/txt/md/pdf）")
    parser.add_argument(
        "--chunk-size", type=int, default=512, help="切片大小（默认 512）"
    )
    parser.add_argument(
        "--overlap", type=int, default=64, help="切片重叠（默认 64）"
    )
    args = parser.parse_args()

    if not args.file.exists():
        print(f"文件不存在: {args.file}")
        raise SystemExit(1)

    process_file(args.file, args.chunk_size, args.overlap)


if __name__ == "__main__":
    main()
