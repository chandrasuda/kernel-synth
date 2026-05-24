"""Triton kernels for DistillWrapper.

Fill this in. Typical pattern::

    import triton
    import triton.language as tl


    @triton.jit
    def my_fused_kernel(
        X_ptr, Y_ptr, OUT_ptr,
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(X_ptr + offsets, mask=mask)
        y = tl.load(Y_ptr + offsets, mask=mask)
        tl.store(OUT_ptr + offsets, x + y, mask=mask)


    def fused_add(x, y):
        out = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        my_fused_kernel[grid](x, y, out, N, BLOCK)
        return out

Then import + call ``fused_add`` from ``solution.py``.

Tips
----
* Mark sizes / strides as ``tl.constexpr`` when they're fixed.
* On CUDA, prefer ``tl.dot`` over hand-rolled matmuls.
* Keep a Python fallback in solution.py for shapes your kernel
  doesn't support — better correct & slow than broken.
"""
from __future__ import annotations

# Optional — Triton isn't available on every machine. Import lazily so
# importing this module never explodes; the agent should add the
# ``import triton`` line right alongside its kernels.
