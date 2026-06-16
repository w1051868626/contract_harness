"""自定义异常类层次结构。"""


class HarnessError(Exception):
    """基础异常。"""

    pass


class AgentError(HarnessError):
    """Agent 相关异常。"""

    pass


class ReplayError(HarnessError):
    """回放系统相关异常。"""

    pass


class EvalError(HarnessError):
    """评测系统相关异常。"""

    pass


class RegressionError(HarnessError):
    """回归测试相关异常。"""

    pass


class ChunkingError(HarnessError):
    """文档分块相关异常。"""

    pass


class EmbeddingError(HarnessError):
    """嵌入向量生成相关异常。"""

    pass
