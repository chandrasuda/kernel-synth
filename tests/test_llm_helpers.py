"""Unit tests for the small pure helpers in ``kernel_synth.llm``.

Both helpers are runtime-safe to test without any LLM provider configured
(neither imports the SDK clients).
"""

from __future__ import annotations

from kernel_synth.llm import _parse_tool_arguments, _redact_secrets


def test_redact_secrets_scrubs_anthropic_style_key() -> None:
    msg = "401 Unauthorized: invalid x-api-key sk-ant-api03-aBcDeFgHiJkLmN_o-Pqr-1234567"
    out = _redact_secrets(msg)
    assert "sk-ant-api03" not in out
    assert "sk-***REDACTED***" in out
    assert "401 Unauthorized" in out


def test_redact_secrets_scrubs_openai_style_key() -> None:
    msg = "Error: bad key sk-proj-ABCDEFGHIJKLMNOPQRSTUVWX12345 found in env"
    assert "sk-proj-ABCDEFGHIJKLMNOPQRSTUVWX12345" not in _redact_secrets(msg)


def test_redact_secrets_passes_clean_messages_through_unchanged() -> None:
    msg = "RateLimitError: please slow down"
    assert _redact_secrets(msg) is msg or _redact_secrets(msg) == msg


def test_redact_secrets_handles_non_strings() -> None:
    assert _redact_secrets(None) is None  # type: ignore[arg-type]
    assert _redact_secrets(123) == 123  # type: ignore[arg-type]


def test_parse_tool_arguments_accepts_dict() -> None:
    assert _parse_tool_arguments({"a": 1}) == {"a": 1}


def test_parse_tool_arguments_accepts_json_string() -> None:
    assert _parse_tool_arguments('{"runs": 3}') == {"runs": 3}


def test_parse_tool_arguments_treats_empty_as_dict() -> None:
    assert _parse_tool_arguments("") == {}
    assert _parse_tool_arguments(None) == {}


def test_parse_tool_arguments_preserves_invalid_json_as_raw() -> None:
    out = _parse_tool_arguments("not json {")
    assert out == {"_raw": "not json {"}


def test_parse_tool_arguments_handles_non_object_json() -> None:
    out = _parse_tool_arguments("42")
    assert out == {"_value": 42}
