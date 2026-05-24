"""Task spec for the kernel-engineering agent.

Goal
----
Write **Triton** kernels (no raw CUDA) that match ``reference.TextDecoder``
numerically and are faster than PyTorch eager — ideally faster than
``torch.compile``. ``build(**kwargs)`` must keep returning something
callable like the reference module.

Files
-----
* ``reference.py``       — frozen target; read-only.
* ``inputs.py``          — drives the benchmark shapes/kwargs; read-only.
* ``triton_kernels.py``  — write your @triton.jit kernels here.
* ``solution.py``        — THIS FILE. Wire the kernels in below the marker.

Constraints
-----------
* Triton only — no raw CUDA, no .cu files, no cpp_extension.
* Restricted to writing files inside this folder via ``write_file``.
* Must keep ``build(**kwargs)`` callable and module-compatible.
* Must produce numerically-equivalent outputs (rtol=1e-3, atol=1e-4).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import reference  # noqa: E402

# The empty stub is here so ``solution.py`` is importable even before
# you've written any kernels. Once you add kernels, import + use them
# below the marker.
try:
    import triton_kernels  # noqa: F401,E402
except Exception:  # noqa: BLE001
    triton_kernels = None  # type: ignore[assignment]


# === REPLACE BELOW ===
# Baseline implementation: wraps the reference verbatim, speedup = 1.0.
# Replace this with a module that calls into your Triton kernels.

def build(**kwargs):
    """Return a module instance to be benchmarked.

    Must accept the same kwargs as ``reference.TextDecoder`` and
    return something callable like ``module(*args, **kwargs)`` that
    produces the same shape/dtype/values within tolerance.
    """
    return reference.TextDecoder(**kwargs)

# === REPLACE ABOVE ===
