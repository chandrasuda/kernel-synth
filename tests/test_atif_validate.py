"""Round-trip a TrajectoryBuilder-built trajectory through ``validate``.

This is the smallest possible regression test for the ATIF v1.7
validators: build a tiny but realistic trajectory (system + user +
agent step with a tool call and matching observation), persist it via
``write_json``, and re-validate from disk.
"""

from __future__ import annotations

import json
from pathlib import Path

from kernel_synth.rl.atif import (
    AtifAgent,
    Metrics,
    Observation,
    ObservationResult,
    ToolCall,
    TrajectoryBuilder,
    validate,
)


def _builder() -> TrajectoryBuilder:
    return TrajectoryBuilder(
        agent=AtifAgent(
            name="kernel-synth-test",
            version="0.1.0",
            model_name="test-model",
            tool_definitions=[
                {
                    "name": "run_benchmark",
                    "description": "run the bench",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        ),
        notes="round-trip test",
    )


def test_validate_dict_roundtrips_a_minimal_trajectory() -> None:
    b = _builder()
    b.add_step(source="system", message="you are a kernel agent")
    b.add_step(source="user", message="please run the benchmark")
    call = ToolCall(
        tool_call_id="call_abc123",
        function_name="run_benchmark",
        arguments={"runs": 4},
    )
    b.add_step(
        source="agent",
        model_name="test-model",
        message="running it now",
        tool_calls=[call],
        observation=Observation(
            results=[
                ObservationResult(
                    source_call_id=call.tool_call_id,
                    content=json.dumps({"correct": True, "eager_ms": 1.0}),
                )
            ]
        ),
        metrics=Metrics(prompt_tokens=10, completion_tokens=4),
    )
    payload = b.to_json_dict()
    ok, errs = validate(payload)
    assert ok, errs


def test_validate_file_roundtrips_a_persisted_trajectory(tmp_path: Path) -> None:
    b = _builder()
    b.add_step(source="user", message="hello")
    b.add_step(source="agent", model_name="test-model", message="hi")
    out = b.write_json(tmp_path / "trace.json")
    assert out.is_file()
    ok, errs = validate(out)
    assert ok, errs


def test_validate_rejects_non_sequential_step_ids() -> None:
    payload = {
        "schema_version": "ATIF-v1.7",
        "agent": {"name": "x", "version": "1"},
        "steps": [
            {"step_id": 1, "source": "user", "message": "a"},
            {"step_id": 3, "source": "user", "message": "b"},
        ],
    }
    ok, errs = validate(payload)
    assert not ok
    assert any("step_id" in e or "sequential" in e for e in errs)
