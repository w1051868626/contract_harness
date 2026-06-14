"""法律条文采集器，支持从公开来源爬取法律法规并导入知识库。"""

from __future__ import annotations

import json
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import httpx

from harness.rag.seed_laws import get_seed_laws

COLLECT_DIR = Path(__file__).resolve().parent / "collected"


class LawSource(ABC):
    """法律采集源抽象基类。"""

    @abstractmethod
    def fetch(self, query: str = "") -> list[dict[str, str]]:
        """根据查询词获取法律条文列表，每个元素包含 title 和 content。"""


class SeedLawSource(LawSource):
    """内置种子法律（离线，无需网络）。"""

    def fetch(self, query: str = "") -> list[dict[str, str]]:
        laws = get_seed_laws()
        queries = [q.strip() for q in query.split(",") if q.strip()] if query else []
        if queries:
            laws = [
                law
                for law in laws
                if any(q in law["title"] or q in law["content"] for q in queries)
            ]
        return laws


class NPCLawSource(LawSource):
    """国家法律法规数据库（flk.npc.gov.cn）在线采集。"""

    SEARCH_URL = "https://flk.npc.gov.cn/api/"
    DETAIL_URL = "https://flk.npc.gov.cn/api/detail"

    def __init__(
        self,
        page_size: int = 10,
        delay: float = 1.0,
        proxy: str | None = None,
    ):
        self.page_size = page_size
        self.delay = delay
        self._client = httpx.Client(
            proxy=proxy,
            timeout=30,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Referer": "https://flk.npc.gov.cn/",
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "zh-CN,zh;q=0.9",
            },
        )

    def fetch(self, query: str = "") -> list[dict[str, str]]:
        """从 NPC 数据库搜索并获取法律全文。"""
        queries = [q.strip() for q in query.split(",") if q.strip()] if query else [""]
        seen_titles: set[str] = set()
        results: list[dict[str, str]] = []
        for q in queries:
            page_num = 1
            while len(results) < self.page_size:
                try:
                    items = self._search_page(q or "民法典", page_num)
                    if not items:
                        break
                    for item in items:
                        title = item.get("title", "")
                        if title in seen_titles:
                            continue
                        seen_titles.add(title)
                        detail = self._fetch_detail(item.get("id", ""))
                        if detail:
                            results.append(detail)
                        if len(results) >= self.page_size:
                            break
                        time.sleep(self.delay)
                    page_num += 1
                except Exception as e:
                    print(f"  [采集异常] query={q!r}, 第 {page_num} 页: {e}")
                    break
        return results

    def _search_page(self, query: str, page: int) -> list[dict[str, Any]]:
        resp = self._client.get(
            self.SEARCH_URL,
            params={
                "page": page,
                "size": 10,
                "type": "flfg",
                "searchType": "title;accurate",
                "sortTr": "f_bbrq_s;desc",
                "gbrqStart": "",
                "gbrqEnd": "",
                "sxrqStart": "",
                "sxrqEnd": "",
                "sort": "true",
                "searchWord": query or "民法典",
            },
        )
        resp.raise_for_status()
        ct = resp.headers.get("content-type", "")
        if "text/html" in ct:
            raise RuntimeError(
                "国家法律法规数据库 API 需要浏览器环境，无法直接通过 HTTP 请求采集。"
                "建议使用 --source seed 使用内置种子数据。"
            )
        return resp.json().get("result", [])

    def _fetch_detail(self, law_id: str) -> dict[str, str] | None:
        resp = self._client.post(self.DETAIL_URL, data={"id": law_id})
        resp.raise_for_status()
        data = resp.json().get("result", {})
        title = data.get("title", "")
        # body/bodyText/content 字段名因 API 版本而异
        content = data.get("body", "") or data.get("bodyText", "") or data.get("content", "")
        if not title or not content:
            return None
        return {"title": title, "content": content}


class PlaywrightNPCSource(LawSource):
    """基于 Playwright 的国家法律法规数据库采集器，绕过反爬保护。"""

    SEARCH_URL = "https://flk.npc.gov.cn/api/"
    DETAIL_URL = "https://flk.npc.gov.cn/api/detail"

    def __init__(
        self,
        page_size: int = 10,
        delay: float = 1.0,
        headless: bool = True,
    ):
        self.page_size = page_size
        self.delay = delay
        self.headless = headless

    def _ensure_playwright(self):
        """检查 Playwright 是否可用。"""
        try:
            import playwright  # noqa: F401
        except ImportError:
            raise ImportError(
                "Playwright 未安装。运行: pip install playwright && playwright install chromium"
            )

    def fetch(self, query: str = "") -> list[dict[str, str]]:
        """通过浏览器环境访问 NPC API 采集法律条文。"""
        self._ensure_playwright()
        from playwright.sync_api import sync_playwright

        queries = [q.strip() for q in query.split(",") if q.strip()] if query else [""]
        seen_titles: set[str] = set()
        results: list[dict[str, str]] = []

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                locale="zh-CN",
            )
            api = context.request

            for q in queries:
                page_num = 1
                while len(results) < self.page_size:
                    try:
                        items = self._search_page_playwright(api, q or "民法典", page_num)
                        if not items:
                            break
                        for item in items:
                            title = item.get("title", "")
                            if title in seen_titles:
                                continue
                            seen_titles.add(title)
                            detail = self._fetch_detail_playwright(api, item.get("id", ""))
                            if detail:
                                results.append(detail)
                            if len(results) >= self.page_size:
                                break
                            time.sleep(self.delay)
                        page_num += 1
                    except Exception as e:
                        print(f"  [采集异常] query={q!r}, 第 {page_num} 页: {e}")
                        break

            browser.close()

        return results

    def _search_page_playwright(self, api, query: str, page_num: int) -> list[dict[str, Any]]:
        resp = api.get(
            self.SEARCH_URL,
            params={
                "page": str(page_num),
                "size": "10",
                "type": "flfg",
                "searchType": "title;accurate",
                "sortTr": "f_bbrq_s;desc",
                "gbrqStart": "",
                "gbrqEnd": "",
                "sxrqStart": "",
                "sxrqEnd": "",
                "sort": "true",
                "searchWord": query,
            },
        )
        data = resp.json()
        return data.get("result", [])

    def _fetch_detail_playwright(self, api, law_id: str) -> dict[str, str] | None:
        resp = api.post(self.DETAIL_URL, data={"id": law_id})
        data = resp.json().get("result", {})
        title = data.get("title", "")
        content = data.get("body", "") or data.get("bodyText", "") or data.get("content", "")
        if not title or not content:
            return None
        return {"title": title, "content": content}


class LocalFileSource(LawSource):
    """从本地 JSON 文件读取已采集的法律条文。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def fetch(self, query: str = "") -> list[dict[str, str]]:
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if query:
            data = [d for d in data if query in d.get("title", "") or query in d.get("content", "")]
        return data


def collect(
    source: str = "seed",
    query: str = "",
    output_dir: str | Path | None = None,
    proxy: str | None = None,
) -> list[dict[str, str]]:
    """采集法律条文并保存到 output_dir，返回采集结果列表。"""
    sources: dict[str, LawSource] = {
        "seed": SeedLawSource(),
        "npc": NPCLawSource(proxy=proxy),
        "pw-npc": PlaywrightNPCSource(),
    }
    src = sources.get(source)
    if src is None:
        raise ValueError(f"不支持的采集源: {source}，可选: {list(sources.keys())}")

    print(f"正在从 [{source}] 采集法律条文{' (q=' + query + ')' if query else ''}...")
    laws = src.fetch(query=query)
    if not laws:
        print("  [yellow]未采集到任何条文[/yellow]")
        return []

    print(f"  采集到 {len(laws)} 篇条文")

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        filepath = out / f"collected_{source}_{uuid.uuid4().hex[:8]}.json"
        filepath.write_text(json.dumps(laws, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  已保存: {filepath}")

    return laws
