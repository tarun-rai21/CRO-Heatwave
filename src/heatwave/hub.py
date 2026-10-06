"""Sync the local data mirror with the private Hugging Face dataset (reference.md §5.2).

The token comes from the HF_TOKEN environment variable (a GitHub Actions secret
in CI, or `hf auth login` locally). The repo id comes from HF_DATASET_REPO, or
configs/data.yaml hub.repo_id.
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, snapshot_download


def repo_id(cfg: dict) -> str:
    rid = os.environ.get("HF_DATASET_REPO") or cfg["hub"].get("repo_id")
    if not rid:
        raise RuntimeError("set HF_DATASET_REPO or hub.repo_id in configs/data.yaml")
    return rid


def pull(cfg: dict, root: Path, patterns: list[str]) -> None:
    """Download matching files into the local mirror (only what is missing or changed)."""
    snapshot_download(
        repo_id=repo_id(cfg),
        repo_type=cfg["hub"]["repo_type"],
        local_dir=root,
        allow_patterns=patterns,
    )


def push(cfg: dict, root: Path, files: list[Path], message: str) -> int:
    """Upload exactly these files (paths under root) in one commit."""
    files = [f for f in files if f.exists()]
    if not files:
        return 0
    ops = [
        CommitOperationAdd(path_in_repo=f.relative_to(root).as_posix(), path_or_fileobj=str(f))
        for f in files
    ]
    HfApi().create_commit(
        repo_id=repo_id(cfg),
        repo_type=cfg["hub"]["repo_type"],
        operations=ops,
        commit_message=message,
    )
    return len(ops)


def files_of_run(raw_power: Path, run_id: str) -> list[Path]:
    """The raw files a run created (one per observation year it touched)."""
    return sorted(raw_power.glob(f"obs_year=*/{run_id}.parquet"))
