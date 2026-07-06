import pytest
from langchain_core.messages import AIMessage, ToolMessage

from tradingagents.graph.tool_node_compat import ensure_tool_call_responses, has_tool_calls


@pytest.mark.unit
def test_ensure_tool_call_responses_pads_missing_tool_messages():
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "get_stock_data:0", "name": "get_stock_data", "args": {}},
                    {"id": "get_indicators:1", "name": "get_indicators", "args": {}},
                ],
            )
        ]
    }
    result = {
        "messages": [
            ToolMessage(
                content="stock rows",
                tool_call_id="get_stock_data:0",
                name="get_stock_data",
            )
        ]
    }

    fixed = ensure_tool_call_responses(state, result)

    returned = {message.tool_call_id for message in fixed["messages"]}
    assert returned == {"get_stock_data:0", "get_indicators:1"}
    missing_message = next(
        message for message in fixed["messages"] if message.tool_call_id == "get_indicators:1"
    )
    assert "<unavailable>" in missing_message.content
    assert missing_message.name == "get_indicators"


@pytest.mark.unit
def test_ensure_tool_call_responses_reraises_without_expected_tool_calls():
    with pytest.raises(RuntimeError, match="boom"):
        ensure_tool_call_responses(
            {"messages": [AIMessage(content="plain response")]},
            {"messages": []},
            execution_error=RuntimeError("boom"),
        )


@pytest.mark.unit
def test_has_tool_calls_detects_serialized_additional_kwargs():
    message = {
        "content": "",
        "additional_kwargs": {
            "tool_calls": [
                {
                    "id": "get_indicators:1",
                    "function": {"name": "get_indicators", "arguments": "{}"},
                    "type": "function",
                }
            ]
        },
    }

    assert has_tool_calls(message)
    fixed = ensure_tool_call_responses({"messages": [message]}, {"messages": []})
    assert fixed["messages"][0].tool_call_id == "get_indicators:1"
