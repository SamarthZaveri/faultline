"""Find new corpus entries on Kaggle (official API) and GitHub (REST search), and assign splits.

Splits come from a fixed hash of the group (competition or repo), so they are decided before
anyone looks at results and can't be tuned. Kaggle authors who appear in several splits are kept
only in the split where most of their notebooks are.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from faultline.eval.corpus import SPLITS, CorpusEntry, KaggleKernel, Manifest, Split

SPLIT_SALT = "faultline-split-v1"
MAX_REPO_KB = 500_000  # GitHub reports size in KB; skip monorepos and data dumps


def assign_split(group: str) -> Split:
    """70/15/15 by a salted hash. Changing the salt re-splits everything: don't."""
    bucket = int(hashlib.sha256(f"{SPLIT_SALT}:{group}".encode()).hexdigest(), 16) % 100
    return "train" if bucket < 70 else "dev" if bucket < 85 else "test"


# ---- Kaggle ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class KernelInfo:
    ref: str
    author: str
    votes: int = 0


class KernelSource(Protocol):
    def list_kernels(self, competition: str, n: int) -> list[KernelInfo]: ...

    def pull(self, ref: str, dest: Path) -> Path: ...


class KaggleApiSource:
    """Kaggle's official client. Needs ~/.kaggle/kaggle.json (or KAGGLE_USERNAME/KAGGLE_KEY)."""

    def __init__(self) -> None:
        module = importlib.import_module("kaggle.api.kaggle_api_extended")  # optional dependency
        self.api: Any = module.KaggleApi()
        self.api.authenticate()

    def list_kernels(self, competition: str, n: int) -> list[KernelInfo]:
        out: list[KernelInfo] = []
        page = 1
        while len(out) < n:
            batch = self.api.kernels_list(
                competition=competition,
                language="python",
                kernel_type="notebook",
                sort_by="voteCount",
                page=page,
                page_size=min(100, n),
            )
            if not batch:
                break
            for k in batch:
                ref = str(getattr(k, "ref", ""))
                votes = getattr(k, "total_votes", None) or getattr(k, "totalVotes", 0) or 0
                if "/" in ref:
                    out.append(KernelInfo(ref=ref, author=ref.split("/", 1)[0], votes=int(votes)))
            page += 1
        return out[:n]

    def pull(self, ref: str, dest: Path) -> Path:
        dest.mkdir(parents=True, exist_ok=True)
        self.api.kernels_pull(ref, path=str(dest))
        files = sorted(p for p in dest.iterdir() if p.suffix in (".ipynb", ".py"))
        if not files:
            raise FileNotFoundError(f"kaggle returned no notebook for {ref}")
        return files[0]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def discover_kaggle(
    source: KernelSource, competitions: list[str], per_competition: int
) -> tuple[list[CorpusEntry], list[str]]:
    entries, skipped = [], []
    for comp in competitions:
        try:
            infos = source.list_kernels(comp, per_competition)
        except (LookupError, OSError, ValueError, RuntimeError):
            skipped.append(comp)
            continue
        if not infos:
            skipped.append(comp)
            continue
        kernels = [KaggleKernel(ref=i.ref, author=i.author, votes=i.votes) for i in infos]
        entries.append(
            CorpusEntry(
                name=f"kaggle-{_slug(comp)}",
                source="kaggle",
                split=assign_split(f"kaggle:{comp}"),
                group=comp,
                competition=comp,
                kernels=kernels,
                kinds=["notebook"],
            )
        )
    return entries, skipped


def resolve_author_overlap(entries: list[CorpusEntry]) -> tuple[list[CorpusEntry], list[str]]:
    """Keep each author's notebooks only in the split holding most of them (ties: earlier split)."""
    per_author: dict[str, Counter[str]] = {}
    for e in entries:
        for k in e.kernels:
            per_author.setdefault(k.author, Counter())[e.split] += 1
    home = {a: max(SPLITS, key=lambda s: (c[s], -SPLITS.index(s))) for a, c in per_author.items()}
    fixed, dropped = [], []
    for e in entries:
        keep = [k for k in e.kernels if home[k.author] == e.split]
        dropped += [k.ref for k in e.kernels if home[k.author] != e.split]
        if e.source != "kaggle" or keep:
            fixed.append(e.model_copy(update={"kernels": keep}))
    return fixed, dropped


# ---- GitHub ----------------------------------------------------------------------------------
Search = Callable[[str, int], list[dict[str, Any]]]
Head = Callable[[str], str]


_last_search = 0.0


def github_search(query: str, per_page: int) -> list[dict[str, Any]]:
    """GitHub REST repository search. Uses GITHUB_TOKEN if set; anonymous search is paced to
    GitHub's 10 requests/minute and waits out a rate-limit reset once."""
    global _last_search
    url = "https://api.github.com/search/repositories?" + urllib.parse.urlencode(
        {"q": query, "sort": "stars", "order": "desc", "per_page": per_page}
    )
    token = os.environ.get("GITHUB_TOKEN")
    for attempt in range(3):
        wait = (2.1 if token else 6.5) - (time.monotonic() - _last_search)
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(
            url, headers={"Accept": "application/vnd.github+json", "User-Agent": "faultline"}
        )
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        try:
            _last_search = time.monotonic()
            with urllib.request.urlopen(req, timeout=30) as resp:
                items: list[dict[str, Any]] = json.load(resp).get("items", [])
            return items
        except urllib.error.HTTPError as e:
            if e.code not in (403, 429) or attempt == 2:
                raise
            retry_after = e.headers.get("Retry-After")
            reset = float(e.headers.get("X-RateLimit-Reset", time.time() + 60))
            delay = float(retry_after) if retry_after else reset - time.time()
            time.sleep(min(max(delay, 1.0) + 1.0, 90.0))
    return []


def git_head(url: str) -> str:
    out = subprocess.run(
        ["git", "ls-remote", url, "HEAD"], check=True, capture_output=True, text=True, timeout=60
    )
    return out.stdout.split()[0]


def discover_github(
    queries: list[str],
    existing: Manifest,
    limit: int,
    search: Search = github_search,
    head: Head = git_head,
    per_query: int = 5,
) -> list[CorpusEntry]:
    """Projects that *use* ML pipelines; at most `per_query` per query so no single query dominates."""
    known = {e.group.lower() for e in existing.repos} | {(e.url or "").lower() for e in existing.repos}
    entries: list[CorpusEntry] = []
    for q in queries:
        taken = 0
        for item in search(q, 50):
            if taken >= per_query:
                break
            full = str(item["full_name"]).lower()
            url = str(item["clone_url"])
            if full in known or url.lower() in known or item.get("fork") or item.get("archived"):
                continue
            if int(item.get("size", 0)) > MAX_REPO_KB or "awesome" in full:
                continue
            taken += 1
            known.add(full)
            entries.append(
                CorpusEntry(
                    name=_slug(full.replace("/", "-")),
                    url=url,
                    commit=head(url),
                    split=assign_split(f"github:{full}"),
                    group=full,
                    notes=f"stars={item.get('stargazers_count', 0)} query={q!r}",
                )
            )
            if len(entries) >= limit:
                return entries
    return entries
