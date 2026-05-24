"""Persist selected modules to ``data/extracted/<owner>__<repo>/``.

Layout per repo:
    data/extracted/<owner>__<repo>/
        manifest.json           # full RepoRecord (Pydantic)
        modules/
            <ClassName>__<slug>.py   # extracted source for each candidate
"""

from __future__ import annotations

import re
from pathlib import Path

from .models import ModuleCandidate, RepoRecord


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def repo_slug(record: RepoRecord) -> str:
    return _safe_name(record.name.replace("/", "__"))


def write_record(record: RepoRecord, *, out_root: Path) -> Path:
    """Write ``record`` to ``out_root/<slug>/`` and return that dir."""
    out_dir = out_root / repo_slug(record)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "modules").mkdir(exist_ok=True)

    for cand in record.candidates:
        _write_module(cand, out_dir / "modules")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    return out_dir


def _write_module(cand: ModuleCandidate, modules_dir: Path) -> None:
    slug = _safe_name(cand.file_path.replace("/", "__").rsplit(".py", 1)[0])
    fname = f"{_safe_name(cand.class_name)}__{slug}.py"
    target = modules_dir / fname
    header = (
        f"# Extracted by kernel-synth\n"
        f"# Source: {cand.file_path} (lines {cand.start_line}-{cand.end_line})\n"
        f"# Class: {cand.class_name}\n"
        f"# Tags: {', '.join(cand.tags)}\n"
        f"# Novelty: {cand.novelty_score:.2f}\n"
        f"# Reason: {cand.reason}\n\n"
    )
    target.write_text(header + (cand.source_code or "") + "\n", encoding="utf-8")


def load_all(out_root: Path) -> list[RepoRecord]:
    """Load every ``manifest.json`` under ``out_root``."""
    if not out_root.exists():
        return []
    out: list[RepoRecord] = []
    for manifest in sorted(out_root.glob("*/manifest.json")):
        try:
            out.append(RepoRecord.model_validate_json(manifest.read_text("utf-8")))
        except Exception:  # noqa: BLE001
            continue
    return out
