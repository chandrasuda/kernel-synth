"""Eval harness for the Attend kernel-engineering env.

Usage:
    python harness.py            # human-readable result
    python harness.py --json     # machine-readable JSON
    python harness.py --runs 50  # average over more timing samples
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch  # noqa: E402

import reference  # noqa: E402
import solution as solution_mod  # noqa: E402
from inputs import (  # noqa: E402
    BATCH, SEQ_LEN, HIDDEN, HEADS, HEAD_DIM, DEVICE, DTYPE,
    build_module_kwargs, build_forward_inputs,
)


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


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time(module, args, kwargs, runs: int) -> tuple[object, float]:
    with torch.no_grad():
        for _ in range(max(2, runs // 5)):
            out = module(*args, **kwargs)
    _sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            out = module(*args, **kwargs)
    _sync()
    return out, (time.perf_counter() - t0) / runs


def _allclose(a, b, rtol=1e-3, atol=1e-4):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False, float("inf")
        diff = (a - b).abs().max().item()
        return diff < atol + rtol * b.abs().max().item(), diff
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
    args = parser.parse_args(argv)

    torch.manual_seed(0)

    try:
        kwargs = build_module_kwargs()
        fwd_args, fwd_kwargs = build_forward_inputs()
    except Exception as e:  # noqa: BLE001
        msg = {"error": "input_build_failed", "detail": repr(e)}
        print(json.dumps(msg, indent=2))
        return 2

    try:
        ref_mod = reference.Attend(**kwargs).to(DEVICE, DTYPE)
        ref_mod.eval()
    except Exception as e:  # noqa: BLE001
        msg = {"error": "reference_init_failed", "detail": repr(e)}
        print(json.dumps(msg, indent=2))
        return 3

    try:
        cand_mod = solution_mod.build(**kwargs).to(DEVICE, DTYPE)
        cand_mod.eval()
    except Exception as e:  # noqa: BLE001
        msg = {"error": "solution_init_failed", "detail": repr(e)}
        print(json.dumps(msg, indent=2))
        return 4

    fwd_args = _to(fwd_args, DEVICE, DTYPE)
    fwd_kwargs = _to(fwd_kwargs, DEVICE, DTYPE)

    try:
        ref_out, ref_t = _time(ref_mod, fwd_args, fwd_kwargs, args.runs)
        cand_out, cand_t = _time(cand_mod, fwd_args, fwd_kwargs, args.runs)
    except Exception as e:  # noqa: BLE001
        msg = {"error": "forward_failed", "detail": repr(e)}
        print(json.dumps(msg, indent=2))
        return 5

    correct, diff = _allclose(ref_out, cand_out)
    speedup = ref_t / cand_t if cand_t > 0 else 0.0
    reward = float(correct) * max(0.0, min(speedup, 10.0)) / 10.0

    result = {
        "module": "Attend",
        "correct": bool(correct),
        "max_diff": float(diff),
        "ref_ms": ref_t * 1000,
        "cand_ms": cand_t * 1000,
        "speedup": speedup,
        "reward": reward,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "shapes": {
            "BATCH": BATCH, "SEQ_LEN": SEQ_LEN, "HIDDEN": HIDDEN,
            "HEADS": HEADS, "HEAD_DIM": HEAD_DIM,
        },
    }
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for k, v in result.items():
            print(f"{k:>10s}  {v}")
    return 0 if correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
