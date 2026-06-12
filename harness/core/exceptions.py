class HarnessError(Exception):
    pass


class AgentError(HarnessError):
    pass


class ReplayError(HarnessError):
    pass


class EvalError(HarnessError):
    pass


class RegressionError(HarnessError):
    pass
