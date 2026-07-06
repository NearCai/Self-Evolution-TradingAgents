"""Compatibility wrappers for LangGraph tool execution.

Some OpenAI-compatible providers validate chat history more strictly than
OpenAI. If an assistant message contains tool calls, the next request must
include one tool response for every ``tool_call_id``. LangGraph's ToolNode
normally guarantees this, but provider-specific tool-call payloads can still
leave gaps. The wrapper below preserves normal ToolNode execution and only
adds explicit unavailable ToolMessages for missing ids.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode

logger = logging.getLogger(__name__)


def _tool_call_id(call: Any) -> str | None:
    if isinstance(call, dict):
        return call.get("id") or call.get("tool_call_id")
    return getattr(call, "id", None) or getattr(call, "tool_call_id", None)


def _tool_call_name(call: Any) -> str | None:
    if isinstance(call, dict):
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        return call.get("name") or function.get("name")
    return getattr(call, "name", None)


def message_tool_calls(message: Any) -> list[Any]:
    """Return parsed or raw tool calls from a BaseMessage or serialized dict."""
    calls: list[Any] = []
    if isinstance(message, dict):
        calls.extend(message.get("tool_calls") or [])
        additional = message.get("additional_kwargs") or {}
        calls.extend(additional.get("tool_calls") or [])
        kwargs = message.get("kwargs") or {}
        calls.extend(kwargs.get("tool_calls") or [])
        kwargs_additional = kwargs.get("additional_kwargs") or {}
        calls.extend(kwargs_additional.get("tool_calls") or [])
        return calls

    calls.extend(getattr(message, "tool_calls", None) or [])
    additional = getattr(message, "additional_kwargs", {}) or {}
    calls.extend(additional.get("tool_calls") or [])
    return calls


def has_tool_calls(message: Any) -> bool:
    return bool(message_tool_calls(message))


def _expected_tool_calls(state: dict[str, Any]) -> list[tuple[str, str | None]]:
    messages = state.get("messages") or []
    if not messages:
        return []
    last_message = messages[-1]
    calls: list[tuple[str, str | None]] = []
    seen: set[str] = set()

    for call in message_tool_calls(last_message):
        call_id = _tool_call_id(call)
        if call_id and call_id not in seen:
            calls.append((call_id, _tool_call_name(call)))
            seen.add(call_id)
    return calls


def _returned_tool_call_ids(messages: Sequence[Any]) -> set[str]:
    returned: set[str] = set()
    for message in messages:
        if isinstance(message, dict):
            call_id = message.get("tool_call_id")
        else:
            call_id = getattr(message, "tool_call_id", None)
        if call_id:
            returned.add(call_id)
    return returned


def ensure_tool_call_responses(
    state: dict[str, Any],
    result: dict[str, Any] | None,
    *,
    execution_error: Exception | None = None,
) -> dict[str, Any]:
    """Return a ToolNode result with one response per requested tool call."""
    result = dict(result or {})
    messages = list(result.get("messages") or [])
    expected = _expected_tool_calls(state)
    if not expected:
        if execution_error is not None:
            raise execution_error
        result["messages"] = messages
        return result

    returned = _returned_tool_call_ids(messages)
    missing = [(call_id, name) for call_id, name in expected if call_id not in returned]
    if not missing:
        result["messages"] = messages
        return result

    detail = (
        f" Tool execution raised {type(execution_error).__name__}: {execution_error}"
        if execution_error is not None
        else ""
    )
    logger.warning(
        "ToolNode returned no response for %d/%d tool calls; adding compatibility placeholders.",
        len(missing),
        len(expected),
    )
    for call_id, name in missing:
        tool_name = name or "unknown_tool"
        messages.append(
            ToolMessage(
                content=(
                    f"<unavailable> Tool call `{tool_name}` did not return a usable "
                    f"response in this provider-compatible execution path.{detail} "
                    "Continue with the available tool outputs and explicitly note "
                    "any evidence gaps."
                ),
                tool_call_id=call_id,
                name=tool_name,
            )
        )
    result["messages"] = messages
    return result


def create_compatible_tool_node(
    tools: Sequence[BaseTool | Callable],
    *,
    name: str = "tools",
) -> Callable[[dict[str, Any], Any | None], dict[str, Any]]:
    """Create a ToolNode that pads missing tool responses for strict providers."""
    inner = ToolNode(tools, name=name)

    def compatible_tool_node(state: dict[str, Any]) -> dict[str, Any]:
        try:
            result = inner.invoke(state)
            return ensure_tool_call_responses(state, result)
        except Exception as exc:  # noqa: BLE001 - preserve graph progress with explicit tool gaps
            logger.warning("ToolNode execution failed; adding compatibility placeholders: %s", exc)
            return ensure_tool_call_responses(state, {"messages": []}, execution_error=exc)

    return compatible_tool_node
