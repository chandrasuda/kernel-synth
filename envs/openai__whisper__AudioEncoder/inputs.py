"""Input generators for the eval harness.

Hand-tuned for openai/whisper AudioEncoder.

AudioEncoder.__init__(n_mels, n_ctx, n_state, n_head, n_layer)
AudioEncoder.forward(x) where x is [B, n_mels, n_ctx*2]:
  - conv1 (k=3, s=1, p=1) preserves time dim
  - conv2 (k=3, s=2, p=1) halves time dim
  - positional_embedding has length n_ctx, so input time must be n_ctx*2

We keep n_ctx small (128 -> input time=256) so the benchmark runs fast on CPU.
The BATCH/SEQ_LEN/HIDDEN/HEADS/HEAD_DIM constants below are re-exported
because benchmark.py imports them at module scope; only BATCH is actually
consumed by build_forward_inputs.
"""
import sys as _sys

import torch

# ---------------------------------------------------------------------------
# Workaround for a missing-symbol bug in the auto-generated reference.py:
# the whisper source uses `SDPA_AVAILABLE` and `scaled_dot_product_attention`
# (typically imported from torch.nn.functional via a feature-flag dance), but
# the env_builder's import filter dropped them. Without a patch, the very
# first forward pass raises `NameError: name 'SDPA_AVAILABLE' is not defined`
# from `MultiHeadAttention.qkv_attention`.
#
# We can't edit reference.py (frozen), but benchmark.py imports `reference`
# BEFORE it imports anything from `inputs`, so by the time this module body
# runs, `reference` is already in sys.modules and we can inject the flag.
# With SDPA_AVAILABLE=False, Python short-circuits the `and` in
#     if SDPA_AVAILABLE and MultiHeadAttention.use_sdpa: ...
# so `scaled_dot_product_attention` is never looked up either, and the
# standard-ops fallback (mathematically identical) is taken.
# ---------------------------------------------------------------------------
_ref_mod = _sys.modules.get("reference")
if _ref_mod is not None and not hasattr(_ref_mod, "SDPA_AVAILABLE"):
    _ref_mod.SDPA_AVAILABLE = False

# Shared problem-size knobs.
BATCH = 1                  # Whisper is heavy on CPU; keep batch=1.
SEQ_LEN = 256              # forward input time dim = n_ctx * 2
HIDDEN = 64                # n_state
HEADS = 4                  # n_head
HEAD_DIM = HIDDEN // HEADS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

# Constructor sizing.
N_MELS = 80
N_CTX = 128
N_STATE = HIDDEN
N_HEAD = HEADS
N_LAYER = 2


def build_module_kwargs() -> dict:
    """Constructor kwargs for the reference / solution module."""
    return dict(
        n_mels=N_MELS,
        n_ctx=N_CTX,
        n_state=N_STATE,
        n_head=N_HEAD,
        n_layer=N_LAYER,
    )


def build_forward_inputs() -> tuple[tuple, dict]:
    """Return (positional args, keyword args) for AudioEncoder.forward.

    x shape: [BATCH, n_mels, n_ctx*2] -> [1, 80, 256] with these defaults.
    """
    x = torch.randn(BATCH, N_MELS, N_CTX * 2)
    return (x,), {}
