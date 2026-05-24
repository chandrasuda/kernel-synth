"""Pydantic models shared across the pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ModuleCandidate(BaseModel):
    """One nn.Module the agent flagged as worth turning into an RL task."""

    file_path: str = Field(..., description="Path relative to the repo root.")
    class_name: str = Field(..., description="The nn.Module subclass name.")
    start_line: int = Field(..., ge=1)
    end_line: int = Field(..., ge=1)
    reason: str = Field(
        ...,
        description="Why this module is interesting for custom-kernel work.",
    )
    novelty_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="0 = boring/standard, 1 = highly unique.",
    )
    tags: list[str] = Field(default_factory=list)
    source_code: str = ""

    @property
    def loc(self) -> int:
        return max(self.end_line - self.start_line + 1, 1)


class RepoRecord(BaseModel):
    """One processed repository in the local buffer."""

    url: str
    name: str  # e.g. "state-spaces/mamba"
    local_path: str
    commit_sha: str | None = None
    # Remote default branch (typically ``main`` / ``master``) snapped at
    # clone time. Lets the SPA deep-link source files at the right ref.
    default_branch: str | None = None
    cloned_at: datetime
    n_python_files: int
    n_loc: int
    selection_mode: Literal["agent", "heuristic"]
    candidates: list[ModuleCandidate] = Field(default_factory=list)
    agent_log: list[dict] = Field(default_factory=list)
    notes: str = ""

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)

    def source_file_url(self, file_path: str) -> str | None:
        """Best-effort github.com link for ``file_path`` at this commit.

        Prefers the pinned commit SHA so links never rot; falls back to the
        default branch.
        """
        if not self.url:
            return None
        base = self.url.rstrip("/").removesuffix(".git")
        ref = self.commit_sha or self.default_branch
        if not ref:
            return None
        clean = file_path.lstrip("/")
        return f"{base}/blob/{ref}/{clean}"
