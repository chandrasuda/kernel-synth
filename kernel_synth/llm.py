"""Tiny wrapper over Anthropic / OpenAI for tool-use agent loops.

The harness only needs *one* primitive: ``chat(messages, tools)`` that
returns either text or a list of tool calls. Both back-ends are normalized
to that surface so the harness code stays small.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Literal

try:
    import anthropic  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    anthropic = None  # type: ignore[assignment]

try:
    import openai  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    openai = None  # type: ignore[assignment]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResponse:
    """One model turn. Either text-only (terminal) or tool-use (continue)."""

    stop_reason: Literal["end", "tool_use"]
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None


class LLMUnavailable(RuntimeError):
    pass


class LLMClient:
    """Uniform chat-with-tools client.

    Provider selection: ``anthropic`` if ``ANTHROPIC_API_KEY`` is set,
    otherwise ``openai`` if ``OPENAI_API_KEY`` is set, otherwise raises.
    """

    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = provider or _auto_provider()
        if self.provider == "anthropic":
            if anthropic is None:
                raise LLMUnavailable("`anthropic` package is not installed.")
            self._client = anthropic.Anthropic()
            self.model = model or os.environ.get(
                "KERNEL_SYNTH_MODEL", "claude-sonnet-4-5"
            )
        elif self.provider == "openai":
            if openai is None:
                raise LLMUnavailable("`openai` package is not installed.")
            self._client = openai.OpenAI()
            self.model = model or os.environ.get(
                "KERNEL_SYNTH_MODEL", "gpt-4.1-mini"
            )
        else:
            raise LLMUnavailable(
                "No LLM provider available. Set ANTHROPIC_API_KEY or OPENAI_API_KEY."
            )

    def chat(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int = 4096,
    ) -> ChatResponse:
        if self.provider == "anthropic":
            return self._chat_anthropic(system, messages, tools, max_tokens)
        return self._chat_openai(system, messages, tools, max_tokens)

    def _chat_anthropic(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
    ) -> ChatResponse:
        anth_tools = [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["parameters"],
            }
            for t in tools
        ]
        resp = self._client.messages.create(
            model=self.model,
            system=system,
            messages=messages,
            tools=anth_tools,
            max_tokens=max_tokens,
        )
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)
            elif getattr(block, "type", None) == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=block.input or {})
                )
        stop = "tool_use" if tool_calls else "end"
        return ChatResponse(
            stop_reason=stop,
            text="".join(text_parts),
            tool_calls=tool_calls,
            raw=resp,
        )

    def _chat_openai(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
    ) -> ChatResponse:
        oai_msgs = [{"role": "system", "content": system}]
        for m in messages:
            oai_msgs.extend(_anthropic_msg_to_openai(m))
        oai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in tools
        ]
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=oai_msgs,
            tools=oai_tools,
            max_tokens=max_tokens,
        )
        choice = resp.choices[0]
        msg = choice.message
        tool_calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        stop = "tool_use" if tool_calls else "end"
        return ChatResponse(
            stop_reason=stop,
            text=msg.content or "",
            tool_calls=tool_calls,
            raw=resp,
        )


def _auto_provider() -> str:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    raise LLMUnavailable(
        "No LLM provider available. Set ANTHROPIC_API_KEY or OPENAI_API_KEY."
    )


def _anthropic_msg_to_openai(m: dict) -> list[dict]:
    """Translate the Anthropic-style messages we keep internally to OpenAI's."""
    role = m["role"]
    content = m["content"]
    if isinstance(content, str):
        return [{"role": role, "content": content}]

    if role == "assistant":
        text_parts: list[str] = []
        oai_tool_calls: list[dict] = []
        for block in content:
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "tool_use":
                oai_tool_calls.append(
                    {
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    }
                )
        out: dict = {"role": "assistant"}
        if text_parts:
            out["content"] = "".join(text_parts)
        if oai_tool_calls:
            out["tool_calls"] = oai_tool_calls
        return [out]

    if role == "user":
        results: list[dict] = []
        plain_parts: list[str] = []
        for block in content:
            if block.get("type") == "tool_result":
                content_val = block.get("content", "")
                if isinstance(content_val, list):
                    content_val = "".join(
                        c.get("text", "") for c in content_val if isinstance(c, dict)
                    )
                results.append(
                    {
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"],
                        "content": content_val or "",
                    }
                )
            elif block.get("type") == "text":
                plain_parts.append(block.get("text", ""))
        out_msgs: list[dict] = []
        if plain_parts:
            out_msgs.append({"role": "user", "content": "".join(plain_parts)})
        out_msgs.extend(results)
        return out_msgs

    return [{"role": role, "content": str(content)}]


def is_available() -> bool:
    """True if either provider has an API key in the environment."""
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"))
