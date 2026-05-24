"""Seed the buffer with five semi-popular repos that should yield non-standard
PyTorch modules for custom-kernel work.

Picked for diversity of mechanism, not for raw popularity:
    * state-spaces/mamba          — selective scan / SSM core
    * lucidrains/vit-pytorch      — ViT variants (lots of unusual mixers)
    * lucidrains/x-transformers   — long tail of attention / FFN variants
    * openai/whisper              — audio encoder/decoder w/ custom blocks
    * lucidrains/audiolm-pytorch  — discrete audio token models
"""

from __future__ import annotations

import sys

from .run_pipeline import main as run_main


SEED_REPOS = [
    "https://github.com/state-spaces/mamba",
    "https://github.com/lucidrains/vit-pytorch",
    "https://github.com/lucidrains/x-transformers",
    "https://github.com/openai/whisper",
    "https://github.com/lucidrains/audiolm-pytorch",
]


def main(argv: list[str] | None = None) -> int:
    extra = list(argv or sys.argv[1:])
    return run_main([*SEED_REPOS, *extra])


if __name__ == "__main__":
    sys.exit(main())
