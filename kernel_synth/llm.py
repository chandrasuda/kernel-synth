"""Tiny wrapper over Anthropic / OpenAI for tool-use agent loops.

The harness only needs *one* primitive: ``chat(messages, tools)`` that
returns either text or a list of tool calls. Both back-ends are normalized
to that surface so the harness code stays small.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

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

    # Retry knobs for transient 429 / 5xx / connection errors. Override via
    # env vars for ops without touching code.
    DEFAULT_MAX_RETRIES = int(os.environ.get("KERNEL_SYNTH_LLM_MAX_RETRIES", "5"))
    DEFAULT_BASE_DELAY_S = float(os.environ.get("KERNEL_SYNTH_LLM_BASE_DELAY_S", "0.75"))
    DEFAULT_MAX_DELAY_S = float(os.environ.get("KERNEL_SYNTH_LLM_MAX_DELAY_S", "20.0"))

    def chat(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int = 4096,
        temperature: float | None = None,
    ) -> ChatResponse:
        if self.provider == "anthropic":
            fn = lambda: self._chat_anthropic(
                system, messages, tools, max_tokens, temperature
            )
        else:
            fn = lambda: self._chat_openai(
                system, messages, tools, max_tokens, temperature
            )
        return self._with_retry(fn)

    def _with_retry(self, fn: Callable[[], ChatResponse]) -> ChatResponse:
        """Call ``fn`` with exponential backoff + jitter on transient errors.

        Retried: HTTP 429 and 5xx, connection / timeout errors. Non-retryable
        errors (4xx other than 429, validation errors, anything we can't
        classify as transient) propagate immediately so the caller sees them.
        """
        max_retries = self.DEFAULT_MAX_RETRIES
        delay = self.DEFAULT_BASE_DELAY_S
        for attempt in range(max_retries + 1):
            try:
                return fn()
            except Exception as e:  # noqa: BLE001
                if attempt >= max_retries or not _is_transient_llm_error(e):
                    raise
                sleep_for = min(delay, self.DEFAULT_MAX_DELAY_S)
                sleep_for += random.uniform(0, sleep_for * 0.25)
                time.sleep(sleep_for)
                delay *= 2.0
        # Unreachable — the loop either returns or raises.
        raise RuntimeError("LLMClient._with_retry exhausted without returning")

    def _chat_anthropic(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
        temperature: float | None = None,
    ) -> ChatResponse:
        anth_tools = [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["parameters"],
            }
            for t in tools
        ]
        extra: dict[str, Any] = {}
        if temperature is not None:
            extra["temperature"] = float(temperature)
        resp = self._client.messages.create(
            model=self.model,
            system=system,
            messages=messages,
            tools=anth_tools,
            max_tokens=max_tokens,
            **extra,
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
        temperature: float | None = None,
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
        extra: dict[str, Any] = {}
        if temperature is not None:
            extra["temperature"] = float(temperature)
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=oai_msgs,
            tools=oai_tools,
            max_tokens=max_tokens,
            **extra,
        )
        choice = resp.choices[0]
        msg = choice.message
        tool_calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            tool_calls.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=_parse_tool_arguments(tc.function.arguments),
                )
            )
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


def _parse_tool_arguments(raw: Any) -> dict[str, Any]:
    """Coerce a tool-call ``arguments`` payload into a Python dict.

    Provider quirks we handle:
      * OpenAI returns ``arguments`` as a JSON-encoded string (sometimes ``None``).
      * Anthropic returns a dict directly; pass it through unchanged.
      * Either provider can occasionally hand back invalid JSON when the
        model stops mid-stream — preserve the raw text under ``_raw`` so
        the tool dispatcher gets a clean dict but the upstream error is
        still observable for debugging.
    """
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"_raw": raw}
        if isinstance(parsed, dict):
            return parsed
        return {"_value": parsed}
    return {"_raw": str(raw)}


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


# Matches the Anthropic / OpenAI / generic ``sk-...`` API key shape — at
# least 20 url-safe characters after the ``sk-`` prefix. The trailing
# boundary is anchored on a non-word character (or string end) so we don't
# eat surrounding punctuation in the redacted log line.
_SECRET_RE = re.compile(r"sk-[A-Za-z0-9_\-]{20,}")


def _redact_secrets(text: str) -> str:
    """Scrub API-key-shaped tokens from ``text`` before it lands in a log.

    Returns the input unchanged when no match is found, so it's safe to
    apply to short / structured messages. Each match is replaced with
    ``"sk-***REDACTED***"`` regardless of the original length so the
    redacted form gives no hint about the secret length.
    """
    if not isinstance(text, str) or "sk-" not in text:
        return text
    return _SECRET_RE.sub("sk-***REDACTED***", text)


# Names of exception classes we always treat as transient regardless of
# whatever attributes the SDK happens to expose this week.
_TRANSIENT_EXC_NAMES = frozenset(
    {
        "RateLimitError",
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "ServiceUnavailableError",
        "Timeout",
        "ReadTimeout",
        "ConnectTimeout",
        "ConnectionError",
        "RemoteProtocolError",
    }
)


def _is_transient_llm_error(exc: BaseException) -> bool:
    """Best-effort classifier across anthropic / openai / httpx exception zoo.

    Both SDKs expose ``.status_code`` on ``APIStatusError``; we treat 429 and
    5xx as retryable. Connection / timeout-shaped exceptions are always
    retryable. Everything else is fatal so the caller surfaces it.
    """
    if exc.__class__.__name__ in _TRANSIENT_EXC_NAMES:
        return True
    status = getattr(exc, "status_code", None)
    if status is None:
        # OpenAI sometimes hangs status on `response.status_code`.
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    try:
        code = int(status) if status is not None else None
    except (TypeError, ValueError):
        code = None
    if code == 429 or (code is not None and 500 <= code < 600):
        return True
    return False
