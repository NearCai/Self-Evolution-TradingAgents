import pytest

from tradingagents.agents.utils.agent_utils import bind_tools_compat


class _LLM:
    def __init__(self):
        self.calls = []

    def bind_tools(self, tools, **kwargs):
        self.calls.append((tools, kwargs))
        return "bound"


class _NoParallelLLM:
    def __init__(self):
        self.calls = []

    def bind_tools(self, tools, **kwargs):
        self.calls.append((tools, kwargs))
        if "parallel_tool_calls" in kwargs:
            raise TypeError("parallel_tool_calls not supported")
        return "bound"


@pytest.mark.unit
def test_bind_tools_compat_disables_parallel_tool_calls():
    llm = _LLM()

    bound = bind_tools_compat(llm, ["tool"])

    assert bound == "bound"
    assert llm.calls == [(["tool"], {"parallel_tool_calls": False})]


@pytest.mark.unit
def test_bind_tools_compat_falls_back_when_arg_not_supported():
    llm = _NoParallelLLM()

    bound = bind_tools_compat(llm, ["tool"])

    assert bound == "bound"
    assert llm.calls == [
        (["tool"], {"parallel_tool_calls": False}),
        (["tool"], {}),
    ]
