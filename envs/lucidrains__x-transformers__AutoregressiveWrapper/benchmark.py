"""Benchmark for the AutoregressiveWrapper kernel-engineering env.

Prints a single JSON object on stdout (with ``--json``) containing the
three timings + correctness:

    {
      "eager_ms":      <float | None>,
      "compile_ms":    <float | None>,
      "solution_ms":   <float | None>,
      "correct":       <bool>,
      "max_diff":      <float>,
      "eager_speedup": <float | None>,   # eager_ms / solution_ms
      "compile_ratio": <float | None>,   # compile_ms / solution_ms
      "device":        "cpu" | "cuda",
      "dtype":         "torch.float32",
      "runs":          <int>,
      "error":         <string, only on failure>,
    }

Even if any single piece fails, the script still emits a JSON object so
the agent harness can read it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch  # noqa: E402

WARMUP_RUNS = 3


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _to(obj, device, dtype):
    if isinstance(obj, torch.Tensor):
        if obj.dtype.is_floating_point:
            return obj.to(device=device, dtype=dtype)
        return obj.to(device=device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to(x, device, dtype) for x in obj)
    if isinstance(obj, dict):
        return {k: _to(v, device, dtype) for k, v in obj.items()}
    return obj


def _time(module, args, kwargs, runs: int) -> tuple[object, float]:
    """Warm + time. Returns (last output, avg ms per run)."""
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            out = module(*args, **kwargs)
    _sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            out = module(*args, **kwargs)
    _sync()
    return out, (time.perf_counter() - t0) * 1000.0 / runs


def _allclose(a, b, rtol=1e-3, atol=1e-4):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape:
            return False, float("inf")
        af = a.detach().float()
        bf = b.detach().float()
        diff = (af - bf).abs().max().item()
        ok = diff < atol + rtol * bf.abs().max().item()
        return ok, float(diff)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b):
        worst = 0.0
        ok = True
        for x, y in zip(a, b):
            okx, dx = _allclose(x, y, rtol, atol)
            ok = ok and okx
            worst = max(worst, dx)
        return ok, worst
    return a == b, 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Skip timing — only verify correctness against eager.",
    )
    args = parser.parse_args(argv)

    torch.manual_seed(0)

    result: dict = {
        "module": "AutoregressiveWrapper",
        "runs": args.runs,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "dtype": "torch.float32",
        "eager_ms": None,
        "compile_ms": None,
        "solution_ms": None,
        "correct": False,
        "max_diff": float("inf"),
        "eager_speedup": None,
        "compile_ratio": None,
        "warnings": [],
    }

    # ---- Imports ----
    try:
        import reference
        import solution as solution_mod
        from inputs import (
            BATCH, SEQ_LEN, HIDDEN, HEADS, HEAD_DIM, DEVICE, DTYPE,
            build_module_kwargs, build_forward_inputs,
        )
    except Exception as e:
        result["error"] = "import_failed"
        result["detail"] = repr(e)
        result["traceback"] = traceback.format_exc(limit=4)
        _emit(result, args.json)
        return 2

    try:
        kwargs = build_module_kwargs()
        fwd_args, fwd_kwargs = build_forward_inputs()
    except Exception as e:
        result["error"] = "input_build_failed"
        result["detail"] = repr(e)
        result["traceback"] = traceback.format_exc(limit=4)
        _emit(result, args.json)
        return 3

    # ---- Build eager reference (seed before every constructor so all three
    # modules share the same random weights) ----
    try:
        torch.manual_seed(0)
        eager_mod = reference.AutoregressiveWrapper(**kwargs).to(DEVICE, DTYPE)
        eager_mod.eval()
    except Exception as e:
        result["error"] = "reference_init_failed"
        result["detail"] = repr(e)
        result["traceback"] = traceback.format_exc(limit=4)
        _emit(result, args.json)
        return 4

    # Place inputs on device.
    try:
        fwd_args = _to(fwd_args, DEVICE, DTYPE)
        fwd_kwargs = _to(fwd_kwargs, DEVICE, DTYPE)
    except Exception as e:
        result["error"] = "input_to_device_failed"
        result["detail"] = repr(e)
        _emit(result, args.json)
        return 5

    # ---- Build solution ----
    try:
        torch.manual_seed(0)
        solution_mod_instance = solution_mod.build(**kwargs)
        if hasattr(solution_mod_instance, "to"):
            solution_mod_instance = solution_mod_instance.to(DEVICE, DTYPE)
        if hasattr(solution_mod_instance, "eval"):
            solution_mod_instance.eval()
        # If the solution exposes the underlying nn.Module state, mirror it
        # from the eager reference so weight-init randomness can't drive a
        # false negative.
        try:
            if hasattr(solution_mod_instance, "load_state_dict") and                     hasattr(eager_mod, "state_dict"):
                solution_mod_instance.load_state_dict(eager_mod.state_dict(), strict=False)
        except Exception:
            pass
    except Exception as e:
        result["error"] = "solution_build_failed"
        result["detail"] = repr(e)
        result["traceback"] = traceback.format_exc(limit=4)
        _emit(result, args.json)
        return 6

    # ---- Reference forward + timing ----
    try:
        eager_out, eager_ms = _time(eager_mod, fwd_args, fwd_kwargs, args.runs)
        result["eager_ms"] = eager_ms
    except Exception as e:
        result["error"] = "eager_forward_failed"
        result["detail"] = repr(e)
        result["traceback"] = traceback.format_exc(limit=4)
        _emit(result, args.json)
        return 7

    # ---- torch.compile baseline (best-effort) ----
    if not args.check_only:
        try:
            torch.manual_seed(0)
            compile_target = reference.AutoregressiveWrapper(**kwargs).to(DEVICE, DTYPE).eval()
            compile_target.load_state_dict(eager_mod.state_dict(), strict=False)
            compiled_mod = torch.compile(
                compile_target,
                dynamic=True,
                fullgraph=False,
            )
            _, compile_ms = _time(compiled_mod, fwd_args, fwd_kwargs, args.runs)
            result["compile_ms"] = compile_ms
        except Exception as e:
            result["compile_ms"] = None
            result["warnings"].append(f"torch.compile failed: {e!r}")

    # ---- Solution forward + timing ----
    try:
        if args.check_only:
            with torch.no_grad():
                sol_out = solution_mod_instance(*fwd_args, **fwd_kwargs)
        else:
            sol_out, sol_ms = _time(
                solution_mod_instance, fwd_args, fwd_kwargs, args.runs
            )
            result["solution_ms"] = sol_ms
    except Exception as e:
        result["error"] = "solution_forward_failed"
        result["detail"] = repr(e)
        result["traceback"] = traceback.format_exc(limit=4)
        _emit(result, args.json)
        return 8

    # ---- Correctness ----
    correct, diff = _allclose(eager_out, sol_out)
    result["correct"] = bool(correct)
    result["max_diff"] = float(diff)

    if result["solution_ms"] and result["solution_ms"] > 0:
        if result["eager_ms"]:
            result["eager_speedup"] = result["eager_ms"] / result["solution_ms"]
        if result["compile_ms"]:
            result["compile_ratio"] = result["compile_ms"] / result["solution_ms"]

    _emit(result, args.json)
    return 0 if result["correct"] else 1


def _emit(result: dict, as_json: bool) -> None:
    if as_json:
        # JSON strict spec disallows NaN/Infinity; replace before serializing
        # so JS consumers (the SPA) can parse the output.
        import math as _math
        def _clean(v):
            if isinstance(v, float):
                if _math.isnan(v) or _math.isinf(v):
                    return None
                return v
            if isinstance(v, dict):
                return {k: _clean(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [_clean(x) for x in v]
            return v
        print(json.dumps(_clean(result), indent=2))
        return
    print(f"== {result['module']} ==")
    for k in ("device", "dtype", "runs", "eager_ms", "compile_ms",
              "solution_ms", "eager_speedup", "compile_ratio",
              "correct", "max_diff"):
        if k in result:
            print(f"  {k:>16s}  {result[k]}")
    for w in result.get("warnings", []):
        print(f"  WARNING: {w}")
    if "error" in result:
        print(f"  ERROR: {result['error']}  ({result.get('detail', '')})")


if __name__ == "__main__":
    raise SystemExit(main())
