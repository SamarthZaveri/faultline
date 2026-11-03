"""The corpus of real pipelines: manifest (committed) and pinned shallow checkouts (not committed)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

Split = Literal["train", "dev", "test"]
DEFAULT_MANIFEST = Path(__file__).resolve().parents[3] / "corpus" / "manifest.yaml"
DEFAULT_REPOS = DEFAULT_MANIFEST.parent / "repos"
# Only the file types detectors read; skips datasets, images and model weights.
SPARSE_PATTERNS = (
    "*.py",
    "*.ipynb",
    "requirements*.txt",
    "*.yaml",
    "*.yml",
    "*.toml",
    "*.cfg",
    "*.ini",
    "*.env",
)


class RepoEntry(BaseModel):
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    url: str
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    split: Split
    kinds: list[str] = Field(default_factory=list)
    notes: str = ""


class Manifest(BaseModel):
    repos: list[RepoEntry]

    @field_validator("repos")
    @classmethod
    def _unique(cls, repos: list[RepoEntry]) -> list[RepoEntry]:
        names = [r.name for r in repos]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate repo names: {sorted(dupes)}")
        return repos

    def in_split(self, split: Split) -> list[RepoEntry]:
        return [r for r in self.repos if r.split == split]


def load_manifest(path: Path = DEFAULT_MANIFEST) -> Manifest:
    return Manifest.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def _git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def fetch(entry: RepoEntry, repos_dir: Path = DEFAULT_REPOS) -> Path:
    """Shallow, sparse checkout of exactly `entry.commit`. Idempotent."""
    dest = repos_dir / entry.name
    if (dest / ".git").exists() and _git(["rev-parse", "HEAD"], dest) == entry.commit:
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    if not (dest / ".git").exists():
        _git(["init", "-q"], dest)
        _git(["config", "core.longpaths", "true"], dest)
        _git(["remote", "add", "origin", entry.url], dest)
    _git(["sparse-checkout", "set", "--no-cone", *SPARSE_PATTERNS], dest)
    _git(["fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", entry.commit], dest)
    _git(["checkout", "-q", "--force", "FETCH_HEAD"], dest)
    return dest
