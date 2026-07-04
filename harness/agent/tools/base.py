"""工具基类，封装 LLM 客户端注入。"""

from harness.agent.llm import LLMClient


class BaseTool:
    """工具基类，子类共享 LLM 客户端注入。"""

    def __init__(self, llm: LLMClient):
        self._llm = llm
