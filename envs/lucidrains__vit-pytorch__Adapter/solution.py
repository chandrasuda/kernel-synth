"""Your optimized implementation of Adapter.

Replace the body of ``build`` with anything that:

* accepts the same constructor kwargs as the reference module
* is callable like ``module(*args, **kwargs)``
* returns the same shape & dtype, numerically equivalent

You may use: pure PyTorch, ``torch.compile``, Triton, custom CUDA, etc.
Don't touch ``reference.py`` or ``inputs.py``.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import reference  # noqa: E402


def build(**kwargs):
    """Return a module instance to be benchmarked.

    The default returns the reference verbatim — your starting baseline,
    with speedup = 1.0. Beat it.
    """
    return reference.Adapter(**kwargs)
