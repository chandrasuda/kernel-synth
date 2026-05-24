"""Read-only GitHub cloner.

Hard constraint: this module is allowed to CLONE only. It will never push,
fetch from arbitrary remotes, or otherwise mutate state on github.com.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

_GITHUB_URL_RE = re.compile(
    r"^(?:https?://github\.com/|git@github\.com:)([\w.\-]+)/([\w.\-]+?)(?:\.git)?/?$"
)


class CloneError(RuntimeError):
    """Raised when we cannot safely clone a repository."""


def parse_github_url(url: str) -> tuple[str, str]:
    """Return (owner, repo) for a GitHub URL.

    Accepts HTTPS, SSH, and bare ``owner/repo`` short-form. Anything else is
    refused so we never accidentally fetch from an unexpected host.
    """
    url = url.strip()
    short = re.fullmatch(r"([\w.\-]+)/([\w.\-]+?)(?:\.git)?", url)
    if short:
        return short.group(1), short.group(2)

    m = _GITHUB_URL_RE.match(url)
    if not m:
        raise CloneError(f"Not a recognized GitHub URL: {url!r}")
    return m.group(1), m.group(2)


def repo_full_name(url: str) -> str:
    owner, repo = parse_github_url(url)
    return f"{owner}/{repo}"


def clone_repo(
    url: str,
    *,
    dest_root: Path,
    depth: int = 1,
    force: bool = False,
    timeout: int = 180,
) -> tuple[Path, str | None]:
    """Shallow-clone ``url`` under ``dest_root`` and return (path, sha).

    The clone is forced to be HTTPS (no auth) regardless of the input style.
    """
    owner, repo = parse_github_url(url)
    safe_url = f"https://github.com/{owner}/{repo}.git"
    dest = dest_root / owner / repo

    if dest.exists():
        if not force and (dest / ".git").is_dir():
            return dest, _head_sha(dest)
        shutil.rmtree(dest)

    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                str(depth),
                "--filter=blob:none",
                "--single-branch",
                safe_url,
                str(dest),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise CloneError(f"Timed out cloning {safe_url}") from e
    except subprocess.CalledProcessError as e:
        raise CloneError(
            f"git clone failed for {safe_url}: {e.stderr.strip() or e.stdout.strip()}"
        ) from e

    return dest, _head_sha(dest)


def _head_sha(repo_path: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def assert_safe_remote(url: str) -> None:
    """Raise if ``url`` isn't a plain github.com URL."""
    parsed = urlparse(url)
    if parsed.scheme not in {"https", "http", ""} or (
        parsed.netloc and parsed.netloc != "github.com"
    ):
        raise CloneError(f"Refusing to touch non-github remote: {url!r}")
