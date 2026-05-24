"""Input generators for the eval harness.

EDIT these to match the module's expected shapes / dtypes.
Auto-inferred defaults are best-effort.
"""
import torch

# Shared problem-size knobs. Tune these for your benchmark.
BATCH = 2
SEQ_LEN = 64
HIDDEN = 64
HEADS = 4
HEAD_DIM = HIDDEN // HEADS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


def build_module_kwargs() -> dict:
    """Constructor kwargs for the reference / solution module."""
    return dict(
        forward_decoder=None  # TODO: pick a value for 'forward_decoder',
    )


def build_forward_inputs() -> tuple[tuple, dict]:
    """Return (positional args, keyword args) for the forward call."""
    seq = torch.randn(BATCH, SEQ_LEN, HIDDEN)  # TODO: shape for 'seq'
    return (seq,), {}
