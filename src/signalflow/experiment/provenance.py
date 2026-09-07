"""What code produced a run: package versions, git commits, platform, seed.

Research results are only reproducible if the run records the code that made
them. :func:`provenance` collects that as flat string tags; :func:`experiment_run`
attaches them to every MLflow run automatically. Anything that cannot be
determined is ``"unknown"`` - never an exception.
"""

import os
import platform
import subprocess
from importlib import metadata
from pathlib import Path

_PACKAGES = {
    "signalflow_trading": ("signalflow-trading", "signalflow"),
    "signalflow_ta": ("signalflow-ta", "signalflow.ta"),
    "signalflow_labs": ("signalflow-labs", "signalflow.labs"),
}


def _version(dist: str) -> str:
    try:
        return metadata.version(dist)
    except Exception:
        return "not installed"


def _repo_root(start: Path) -> "Path | None":
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def git_sha(path: "str | Path | None" = None) -> str:
    """``HEAD`` of the repository containing ``path`` (default: the working directory), ``+dirty`` when modified."""
    root = _repo_root(Path(path or os.getcwd()).resolve())
    if root is None:
        return "unknown"
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True, timeout=5
        )
        sha = head.stdout.strip()
        if not sha:
            return "unknown"
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return f"{sha}+dirty" if status.stdout.strip() else sha
    except Exception:
        return "unknown"


def _package_sha(module: str) -> str:
    """Commit of an editable install (the package's source tree lives in a git checkout)."""
    try:
        spec = __import__(module, fromlist=["__file__"])
        location = Path(getattr(spec, "__file__", "") or "").resolve()
    except Exception:
        return "unknown"
    if not location.exists() or "site-packages" in location.parts:
        return "n/a"
    return git_sha(location.parent)


def provenance(seed: "int | None" = None, cwd: "str | Path | None" = None) -> dict[str, str]:
    """Flat string tags that pin a run to its code: versions, commits, platform, seed."""
    tags: dict[str, str] = {}
    for key, (dist, module) in _PACKAGES.items():
        tags[f"{key}_version"] = _version(dist)
        tags[f"{key}_git"] = _package_sha(module)
    tags["git_sha"] = git_sha(cwd)
    tags["python"] = platform.python_version()
    tags["platform"] = platform.platform()
    tags["polars"] = _version("polars")
    if seed is not None:
        tags["seed"] = str(seed)
    return tags


__all__ = ["git_sha", "provenance"]
