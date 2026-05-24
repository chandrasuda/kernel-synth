"""Top-level glue: ``Pipeline.run(repo_url)``."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .cloner import clone_repo, default_branch, repo_full_name
from .code_buffer import build_buffer
from .extractor import write_record
from .harness import AgentHarness
from .heuristics import select_candidates
from .llm import LLMClient, LLMUnavailable
from .models import RepoRecord


@dataclass
class PipelineConfig:
    data_dir: Path
    use_llm: bool = True
    max_agent_steps: int | None = None

    @property
    def clones_dir(self) -> Path:
        return self.data_dir / "clones"

    @property
    def extracted_dir(self) -> Path:
        return self.data_dir / "extracted"


class Pipeline:
    def __init__(self, config: PipelineConfig | None = None):
        if config is None:
            data_dir = Path(
                os.environ.get("KERNEL_SYNTH_DATA_DIR", "./data")
            ).resolve()
            config = PipelineConfig(data_dir=data_dir)
        self.config = config
        self.config.clones_dir.mkdir(parents=True, exist_ok=True)
        self.config.extracted_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------

    def run(self, url: str, *, force_reclone: bool = False) -> RepoRecord:
        full_name = repo_full_name(url)
        repo_path, sha = clone_repo(
            url, dest_root=self.config.clones_dir, force=force_reclone
        )
        buffer = build_buffer(repo_path)

        selection_mode = "heuristic"
        candidates = []
        agent_log: list[dict] = []
        notes = ""

        if self.config.use_llm:
            try:
                llm = LLMClient()
                harness = AgentHarness(
                    buffer, llm=llm, max_steps=self.config.max_agent_steps
                )
                hresult = harness.run()
                candidates = hresult.candidates
                agent_log = hresult.log
                notes = hresult.summary or hresult.stopped_reason
                selection_mode = "agent"
                if not candidates:
                    # The model produced nothing — fall back so we still
                    # have something to show in the viewer.
                    candidates = select_candidates(buffer, repo_root=repo_path)
                    selection_mode = "heuristic"
                    notes = (notes + " | empty agent result, using heuristic").strip(" |")
            except LLMUnavailable as e:
                notes = f"no LLM available ({e}); used heuristic"
                candidates = select_candidates(buffer, repo_root=repo_path)
                selection_mode = "heuristic"
        else:
            candidates = select_candidates(buffer, repo_root=repo_path)
            notes = "use_llm=False; heuristic mode"

        branch = default_branch(repo_path)
        if branch:
            branch_note = f"default_branch={branch}"
            notes = f"{notes} | {branch_note}".strip(" |") if notes else branch_note

        record = RepoRecord(
            url=url,
            name=full_name,
            local_path=str(repo_path),
            commit_sha=sha,
            default_branch=branch,
            cloned_at=datetime.now(timezone.utc),
            n_python_files=buffer.n_files,
            n_loc=buffer.n_loc,
            selection_mode=selection_mode,
            candidates=candidates,
            agent_log=agent_log,
            notes=notes,
        )
        write_record(record, out_root=self.config.extracted_dir)
        return record
