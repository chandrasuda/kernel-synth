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
        num_coarse_quantizers=None  # TODO: pick a value for 'num_coarse_quantizers',
        num_fine_quantizers=None  # TODO: pick a value for 'num_fine_quantizers',
        codebook_size=None  # TODO: pick a value for 'codebook_size',
        dim=64,
        depth=2,
    )


def build_forward_inputs() -> tuple[tuple, dict]:
    """Return (positional args, keyword args) for the forward call."""
    coarse_token_ids = torch.randn(BATCH, SEQ_LEN, HIDDEN)  # TODO: shape for 'coarse_token_ids'
    fine_token_ids = torch.randn(BATCH, SEQ_LEN, HIDDEN)  # TODO: shape for 'fine_token_ids'
    return (coarse_token_ids, fine_token_ids,), {}
