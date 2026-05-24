"""Reference Triton kernels the system prompt can point new agents at.

These are *teaching* kernels — small, well-commented, and self-contained.
Importing this module does NOT require Triton; we lazy-import inside the
helpers so the file stays cheap to import everywhere (CPU CI, doc builds,
the SPA backend). The actual kernel JIT-compiles on first use, only on
machines where Triton can run.

Each helper has the same shape::

    def triton_<op>(x, ...) -> torch.Tensor

so a kernel-engineering agent can copy any one into ``triton_kernels.py``
verbatim and start iterating from there.
"""

from __future__ import annotations

from typing import Any


__all__ = [
    "TRITON_SOFTMAX_SOURCE",
    "is_triton_available",
    "triton_softmax",
]


# Kept as a top-level string so the system prompt and unit tests can
# reference it without re-reading this file from disk.
TRITON_SOFTMAX_SOURCE = '''\
import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    OUT_ptr,
    IN_ptr,
    in_row_stride,
    out_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """One row per program: subtract row-max, exp, divide by row-sum."""
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row_start_in = IN_ptr + row_idx * in_row_stride
    row_start_out = OUT_ptr + row_idx * out_row_stride

    x = tl.load(row_start_in + col_offsets, mask=mask, other=-float("inf"))
    x_minus_max = x - tl.max(x, axis=0)
    numerator = tl.exp(x_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    tl.store(row_start_out + col_offsets, softmax_output, mask=mask)


def triton_softmax(x: "torch.Tensor") -> "torch.Tensor":
    """Numerically-stable rowwise softmax over the last dim."""
    assert x.is_cuda and x.dtype == torch.float32, "softmax kernel needs fp32 cuda"
    *leading, n_cols = x.shape
    x2 = x.reshape(-1, n_cols).contiguous()
    out = torch.empty_like(x2)
    n_rows = x2.shape[0]
    # Triton wants a power-of-two BLOCK_SIZE >= n_cols.
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    _softmax_kernel[(n_rows,)](
        out, x2,
        x2.stride(0), out.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return out.reshape(*leading, n_cols)
'''


def is_triton_available() -> bool:
    """``True`` iff ``triton`` imports AND a CUDA device is reachable."""
    try:
        import torch
        import triton  # noqa: F401
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def triton_softmax(x: Any) -> Any:
    """Lazy-import wrapper around the kernel in :data:`TRITON_SOFTMAX_SOURCE`.

    Built so the *signature* is importable everywhere; we only attempt to
    compile the kernel on first call. Raises ``RuntimeError`` (not
    ``ImportError``) on machines without Triton/CUDA so caller-side
    fallback logic can branch on a single exception type.
    """
    if not is_triton_available():
        raise RuntimeError(
            "triton_softmax requires triton + CUDA; install triton and run "
            "on a CUDA device."
        )
    namespace: dict[str, Any] = {}
    exec(TRITON_SOFTMAX_SOURCE, namespace)
    return namespace["triton_softmax"](x)
