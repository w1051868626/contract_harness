"""对比 reranker 候选池扩大前后 top-1 hit_rate。

旧策略：_search_single 候选池 top_k*2
新策略：_search_single 候选池 max(20, top_k*4)  (当前仓库已默认此实现)

从 hit1_miss_rank2.csv 抽样 N 条「差 1 位」的 query (hit@1=0 但 hit@3=1，正确 chunk 在第 2 位)，
分别用两种策略跑，看新策略能把多少条「差 1 位」救回 hit@1。
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.utils.io import load_dotenv  # type: ignore  # noqa: E402

load_dotenv()

from harness.core.config import HarnessConfig  # noqa: E402
from harness.rag.knowledge_base import KnowledgeBase  # noqa: E402
from harness.rag.sparse import rrf_fuse  # noqa: E402
from harness.rag.vector_store import Chunk  # noqa: E402

SAMPLE_N = 200
TOP_K = 5
# 优先用新一轮 eval 产物（.harness/reports/），回退到旧产物（.harness/evals/reports/）
_NEW = ROOT / ".harness" / "reports" / "analysis" / "hit1_miss_hit3_hit.csv"
_OLD = ROOT / ".harness" / "evals" / "reports" / "analysis" / "hit1_miss_rank2.csv"
RANK2_CSV = _NEW if _NEW.exists() else _OLD


def load_rank2_samples(n: int) -> list[dict]:
    rows = []
    with open(RANK2_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    rng = random.Random(42)
    rng.shuffle(rows)
    return rows[:n]


def make_search_single(kb: KnowledgeBase, pool_fn, inject_prefix: bool = True) -> None:
    """Monkeypatch kb._search_single 用 pool_fn(top_k) 计算候选池大小。

    inject_prefix: True 用 reranker 的标题前缀注入（新实现），
                   False monkeypatch _format_for_rerank 返裸 content（对照基线）。
    """
    embedding = kb._embedding
    store = kb._store
    sparse = kb._sparse_retriever
    rrf_k = kb._rrf_k
    reranker = kb._reranker

    # 对照组：让 reranker 喂裸 content，覆盖 _format_for_rerank
    if not inject_prefix and reranker is not None and hasattr(reranker, "_format_for_rerank"):
        reranker._format_for_rerank = staticmethod(lambda c: c.content)  # type: ignore

    def _search_single(text: str, top_k: int) -> list[Chunk]:
        pool = pool_fn(top_k)
        query_emb = embedding.embed(text)
        dense = store.search(query_emb, top_k=pool)
        if sparse is not None:
            sparse_c = sparse.search(text, top_k=pool)
            fuse_k = pool if reranker else top_k
            candidates = rrf_fuse(dense, sparse_c, fuse_k, k=rrf_k)
        else:
            candidates = dense[:pool] if reranker else dense[:top_k]
        if reranker and len(candidates) > 1:
            return reranker.rerank(text, candidates, top_k=top_k)
        return candidates[:top_k]

    kb._search_single = _search_single  # type: ignore


def run_eval(kb: KnowledgeBase, samples: list[dict], label: str = "") -> dict:
    hit1 = hit3 = hit5 = 0
    total = len(samples)
    desc = f"{label} eval" if label else "eval"
    pbar = tqdm(samples, desc=desc, unit="query")
    for s in pbar:
        try:
            results = kb.query(s["query"], top_k=TOP_K, expansion_threshold=0)
            retrieved = [r.id for r in results]
        except Exception as e:
            tqdm.write(f"ERROR: {e}")
            continue
        expected = s["expected"]
        if len(retrieved) > 0 and retrieved[0] == expected:
            hit1 += 1
        if expected in retrieved[:3]:
            hit3 += 1
        if expected in retrieved[:5]:
            hit5 += 1
        done = pbar.n
        pbar.set_postfix(
            h1=f"{hit1}/{done}={hit1 / done:.3f}" if done else "h1=0",
            h3=hit3,
            h5=hit5,
        )
    pbar.close()
    return {
        "n": total,
        "hit1": hit1,
        "hit3": hit3,
        "hit5": hit5,
        "hit1_rate": hit1 / total if total else 0,
        "hit3_rate": hit3 / total if total else 0,
        "hit5_rate": hit5 / total if total else 0,
    }


def main() -> None:
    samples = load_rank2_samples(SAMPLE_N)
    print(f"样本: {len(samples)} 条 (来自 hit1_miss_rank2.csv)")
    print("这些 query 原本都是 hit@1=0 但 hit@3=1，正确 chunk 在第 2 位")
    print()

    cfg = HarnessConfig()

    # 策略 A: 旧 (pool=20 + 裸 content，对照基线)
    print("=" * 70)
    print("策略 A (旧): 候选池 max(20, top_k*4) + 裸 content（无标题前缀）")
    print("=" * 70)
    kb_old = KnowledgeBase.from_config(cfg)
    make_search_single(kb_old, lambda tk: max(20, tk * 4), inject_prefix=False)
    res_old = run_eval(kb_old, samples, label="A-baseline")
    print()
    print(f"A 结果: hit@1={res_old['hit1']}/{res_old['n']} = {res_old['hit1_rate']:.3f}")
    print(f"        hit@3={res_old['hit3']}  hit@5={res_old['hit5']}")
    print()

    # 策略 B: 新 (pool=20 + 注入法条标题前缀)
    print("=" * 70)
    print("策略 B (新): 候选池 max(20, top_k*4) + 注入【法律名·第X条】前缀")
    print("=" * 70)
    kb_new = KnowledgeBase.from_config(cfg)
    make_search_single(kb_new, lambda tk: max(20, tk * 4), inject_prefix=True)
    res_new = run_eval(kb_new, samples, label="B-prefix")
    print()
    print(f"B 结果: hit@1={res_new['hit1']}/{res_new['n']} = {res_new['hit1_rate']:.3f}")
    print(f"        hit@3={res_new['hit3']}  hit@5={res_new['hit5']}")
    print()

    print("=" * 70)
    print("对比")
    print("=" * 70)
    print(f"{'metric':<12} {'A 裸content':<18} {'B 注入前缀':<18} {'delta':<12}")
    print("-" * 60)
    for k, label in [("hit1_rate", "hit@1"), ("hit3_rate", "hit@3"), ("hit5_rate", "hit@5")]:
        old_v = res_old[k]
        new_v = res_new[k]
        delta = new_v - old_v
        sign = "+" if delta >= 0 else ""
        print(f"{label:<12} {old_v:<18.3f} {new_v:<18.3f} {sign}{delta:<11.3f}")


if __name__ == "__main__":
    main()
