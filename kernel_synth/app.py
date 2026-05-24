"""Local FastAPI viewer for the extracted module buffer.

Serves a single-page UI at ``/`` and a small JSON API:

    GET /api/repos                  -> list of RepoRecord summaries
    GET /api/repos/{slug}           -> full RepoRecord
    GET /api/modules                -> flat list of every ModuleCandidate
    GET /api/stats                  -> aggregate counts
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .extractor import load_all, repo_slug
from .models import RepoRecord


_STATIC_DIR = Path(__file__).resolve().parent / "static"


def _data_root() -> Path:
    return Path(os.environ.get("KERNEL_SYNTH_DATA_DIR", "./data")).resolve()


def _extracted_root() -> Path:
    return _data_root() / "extracted"


app = FastAPI(title="kernel-synth viewer", version="0.1.0")


app.mount(
    "/static",
    StaticFiles(directory=str(_STATIC_DIR)),
    name="static",
)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(_STATIC_DIR / "index.html"))


@app.get("/api/repos")
def list_repos() -> JSONResponse:
    records = load_all(_extracted_root())
    out: list[dict[str, Any]] = []
    for rec in records:
        avg_novelty = (
            sum(c.novelty_score for c in rec.candidates) / len(rec.candidates)
            if rec.candidates
            else 0.0
        )
        out.append(
            {
                "slug": repo_slug(rec),
                "name": rec.name,
                "url": rec.url,
                "commit_sha": rec.commit_sha,
                "cloned_at": rec.cloned_at.isoformat(),
                "n_python_files": rec.n_python_files,
                "n_loc": rec.n_loc,
                "n_candidates": len(rec.candidates),
                "selection_mode": rec.selection_mode,
                "avg_novelty": round(avg_novelty, 3),
                "notes": rec.notes,
            }
        )
    out.sort(key=lambda r: r["name"])
    return JSONResponse(out)


@app.get("/api/repos/{slug}")
def get_repo(slug: str) -> JSONResponse:
    record = _load_one(slug)
    return JSONResponse(record.model_dump(mode="json"))


@app.get("/api/modules")
def list_modules() -> JSONResponse:
    records = load_all(_extracted_root())
    out: list[dict[str, Any]] = []
    for rec in records:
        for cand in rec.candidates:
            out.append(
                {
                    "repo": rec.name,
                    "repo_slug": repo_slug(rec),
                    "file_path": cand.file_path,
                    "class_name": cand.class_name,
                    "start_line": cand.start_line,
                    "end_line": cand.end_line,
                    "reason": cand.reason,
                    "novelty_score": cand.novelty_score,
                    "tags": cand.tags,
                    "loc": cand.loc,
                }
            )
    out.sort(key=lambda r: (-r["novelty_score"], r["repo"]))
    return JSONResponse(out)


@app.get("/api/stats")
def stats() -> JSONResponse:
    records = load_all(_extracted_root())
    n_repos = len(records)
    n_modules = sum(len(r.candidates) for r in records)
    tags: dict[str, int] = {}
    for r in records:
        for c in r.candidates:
            for t in c.tags:
                tags[t] = tags.get(t, 0) + 1
    n_loc = sum(r.n_loc for r in records)
    avg_nov = (
        sum(c.novelty_score for r in records for c in r.candidates) / n_modules
        if n_modules
        else 0.0
    )
    top_tags = sorted(tags.items(), key=lambda kv: -kv[1])[:14]
    return JSONResponse(
        {
            "n_repos": n_repos,
            "n_modules": n_modules,
            "n_loc": n_loc,
            "avg_novelty": round(avg_nov, 3),
            "top_tags": [{"tag": t, "count": c} for t, c in top_tags],
        }
    )


def _load_one(slug: str) -> RepoRecord:
    if not re.fullmatch(r"[A-Za-z0-9._\-]+", slug):
        raise HTTPException(400, "bad slug")
    manifest = _extracted_root() / slug / "manifest.json"
    if not manifest.is_file():
        raise HTTPException(404, f"unknown repo slug: {slug}")
    return RepoRecord.model_validate_json(manifest.read_text("utf-8"))
