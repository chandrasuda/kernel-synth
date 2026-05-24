"""Local FastAPI viewer for the extracted module buffer.

Serves a single-page UI at ``/`` and a small JSON API:

    GET /api/repos                  -> list of RepoRecord summaries
    GET /api/repos/{slug}           -> full RepoRecord
    GET /api/modules                -> flat list of every ModuleCandidate
    GET /api/stats                  -> aggregate counts
    GET /api/envs                   -> one row per RL env folder
    GET /api/envs/{slug}/traces     -> trace files (rollouts) for an env
    GET /api/envs/{slug}/traces/{name}  -> raw trajectory JSON
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .extractor import load_all, repo_slug
from .models import RepoRecord


_BOOT_TIME = datetime.now(timezone.utc)


_STATIC_DIR = Path(__file__).resolve().parent / "static"
_SLUG_RE = re.compile(r"^[A-Za-z0-9._\-]+$")
_TRACE_NAME_RE = re.compile(r"^[A-Za-z0-9._\-]+\.json$")


def _data_root() -> Path:
    return Path(os.environ.get("KERNEL_SYNTH_DATA_DIR", "./data")).resolve()


def _extracted_root() -> Path:
    return _data_root() / "extracted"


def _envs_root() -> Path:
    return Path(os.environ.get("KERNEL_SYNTH_ENVS_DIR", "./envs")).resolve()


app = FastAPI(title="kernel-synth viewer", version="0.1.0")


app.mount(
    "/static",
    StaticFiles(directory=str(_STATIC_DIR)),
    name="static",
)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(_STATIC_DIR / "index.html"))


@app.get("/api/health")
def health() -> JSONResponse:
    """Lightweight liveness probe + tiny inventory summary.

    Surfaces enough state for the SPA to render a faint pulse dot and
    show the build timestamp without paying for the full ``/api/stats``
    walk.
    """
    records = load_all(_extracted_root())
    envs_root = _envs_root()
    n_envs = 0
    if envs_root.is_dir():
        n_envs = sum(
            1
            for d in envs_root.iterdir()
            if d.is_dir() and (d / "env.json").is_file()
        )
    return JSONResponse(
        {
            "ok": True,
            "version": __version__,
            "boot_time": _BOOT_TIME.isoformat(),
            "now": datetime.now(timezone.utc).isoformat(),
            "n_repos": len(records),
            "n_envs": n_envs,
        }
    )


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
                    "source_url": rec.source_file_url(cand.file_path),
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
    if not _SLUG_RE.fullmatch(slug):
        raise HTTPException(400, "bad slug")
    manifest = _extracted_root() / slug / "manifest.json"
    if not manifest.is_file():
        raise HTTPException(404, f"unknown repo slug: {slug}")
    return RepoRecord.model_validate_json(manifest.read_text("utf-8"))


# ---------------------------------------------------------------------------
# RL env / trace endpoints


@app.get("/api/envs")
def list_envs() -> JSONResponse:
    root = _envs_root()
    if not root.is_dir():
        return JSONResponse([])
    rows: list[dict[str, Any]] = []
    for env_dir in sorted(root.iterdir()):
        if not env_dir.is_dir():
            continue
        meta_path = env_dir / "env.json"
        if not meta_path.is_file():
            continue
        meta: dict[str, Any] = {}
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        traces = _trace_summaries(env_dir)
        best = max(
            (t for t in traces if t["reward"] is not None),
            key=lambda t: t["reward"],
            default=None,
        )
        latest = traces[0] if traces else None

        source = meta.get("source") or {}
        rows.append(
            {
                "slug": env_dir.name,
                "name": meta.get("name") or env_dir.name,
                "class_name": meta.get("class_name"),
                "repo": source.get("repo"),
                "repo_url": source.get("url"),
                "tags": meta.get("tags") or [],
                "novelty_score": meta.get("novelty_score"),
                "runnable": meta.get("runnable"),
                "runnable_error": meta.get("runnable_error"),
                "n_traces": len(traces),
                "latest_reward": latest.get("reward") if latest else None,
                "latest_mode": latest.get("mode") if latest else None,
                "best_reward": best.get("reward") if best else None,
                "best_mode": best.get("mode") if best else None,
                "best_trace": best.get("name") if best else None,
            }
        )
    rows.sort(key=lambda r: (-(r.get("best_reward") or float("-inf")), r["slug"]))
    return JSONResponse(rows)


@app.get("/api/envs/{slug}")
def get_env(slug: str) -> JSONResponse:
    env_dir = _resolve_env_dir(slug)
    meta_path = env_dir / "env.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["traces"] = _trace_summaries(env_dir)
    return JSONResponse(meta)


@app.get("/api/envs/{slug}/traces")
def list_env_traces(slug: str) -> JSONResponse:
    env_dir = _resolve_env_dir(slug)
    return JSONResponse(_trace_summaries(env_dir))


@app.get("/api/envs/{slug}/traces/{name}")
def get_env_trace(slug: str, name: str) -> JSONResponse:
    env_dir = _resolve_env_dir(slug)
    if not _TRACE_NAME_RE.fullmatch(name):
        raise HTTPException(400, "bad trace name")
    path = env_dir / "traces" / name
    if not path.is_file():
        raise HTTPException(404, f"trace not found: {name}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise HTTPException(500, f"trace unreadable: {e}") from e
    return JSONResponse(payload)


# ---------------------------------------------------------------------------
# helpers


def _resolve_env_dir(slug: str) -> Path:
    if not _SLUG_RE.fullmatch(slug):
        raise HTTPException(400, "bad slug")
    env_dir = _envs_root() / slug
    if not env_dir.is_dir() or not (env_dir / "env.json").is_file():
        raise HTTPException(404, f"unknown env: {slug}")
    return env_dir


def _trace_summaries(env_dir: Path) -> list[dict[str, Any]]:
    traces_dir = env_dir / "traces"
    if not traces_dir.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for p in sorted(traces_dir.glob("*.json"), reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        fm = data.get("final_metrics") or {}
        fm_extra = (fm.get("extra") or {}) if isinstance(fm, dict) else {}
        extra = data.get("extra") or {}
        reward = fm_extra.get("reward")
        if reward is None:
            # Fall back to the last agent step's metrics.extra.reward.
            for step in reversed(data.get("steps") or []):
                if step.get("source") == "agent":
                    m = step.get("metrics") or {}
                    me = m.get("extra") or {}
                    if "reward" in me:
                        reward = me["reward"]
                    break
        mode = fm_extra.get("mode") or extra.get("mode")
        out.append(
            {
                "name": p.name,
                "path": f"/api/envs/{env_dir.name}/traces/{p.name}",
                "size_bytes": p.stat().st_size,
                "mtime": p.stat().st_mtime,
                "schema_version": data.get("schema_version"),
                "trajectory_id": data.get("trajectory_id"),
                "session_id": data.get("session_id"),
                "n_steps": len(data.get("steps") or []),
                "mode": mode,
                "reward": reward,
            }
        )
    return out
