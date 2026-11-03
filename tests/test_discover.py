from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from faultline.eval.corpus import CorpusEntry, KaggleKernel, Manifest, fetch_kaggle
from faultline.eval.discover import (
    KernelInfo,
    assign_split,
    discover_github,
    discover_kaggle,
    resolve_author_overlap,
)

SHA = "a" * 40


def gh(name: str, split: str, group: str | None = None) -> CorpusEntry:
    return CorpusEntry(name=name, url=f"https://x/{name}.git", commit=SHA, split=split, group=group or name)  # type: ignore[arg-type]


def kg(name: str, split: str, comp: str, authors: list[str]) -> CorpusEntry:
    kernels = [KaggleKernel(ref=f"{a}/nb-{i}", author=a) for i, a in enumerate(authors)]
    return CorpusEntry(name=name, source="kaggle", split=split, group=comp, competition=comp, kernels=kernels)  # type: ignore[arg-type]


# ---- split assignment ----
def test_hash_split_is_deterministic_and_roughly_70_15_15() -> None:
    groups = [f"group-{i}" for i in range(2000)]
    assert [assign_split(g) for g in groups] == [assign_split(g) for g in groups]
    counts = Counter(assign_split(g) for g in groups)
    assert 0.65 < counts["train"] / 2000 < 0.75
    assert 0.11 < counts["dev"] / 2000 < 0.19
    assert 0.11 < counts["test"] / 2000 < 0.19


# ---- manifest invariants ----
def test_group_cannot_span_splits() -> None:
    with pytest.raises(ValueError, match="group"):
        Manifest(repos=[gh("a", "train", "g"), gh("b", "test", "g")])


def test_author_cannot_span_splits() -> None:
    with pytest.raises(ValueError, match="author"):
        Manifest(repos=[kg("kaggle-x", "train", "x", ["alice"]), kg("kaggle-y", "test", "y", ["alice"])])


def test_github_entry_needs_commit_and_kaggle_entry_needs_kernels() -> None:
    with pytest.raises(ValueError):
        CorpusEntry(name="a", split="train", url="https://x/a.git")
    with pytest.raises(ValueError):
        CorpusEntry(name="kaggle-x", source="kaggle", split="train", competition="x")


def test_author_overlap_goes_to_majority_split() -> None:
    entries = [
        kg("kaggle-a", "train", "a", ["alice", "alice", "bob"]),
        kg("kaggle-b", "test", "b", ["alice", "carol"]),
    ]
    fixed, dropped = resolve_author_overlap(entries)
    assert [k.author for k in fixed[1].kernels] == ["carol"]
    assert dropped == ["alice/nb-0"]  # alice's test kernel moved out (she's mostly in train)
    Manifest(repos=fixed)  # now valid


# ---- kaggle discovery ----
class FakeKaggle:
    def __init__(self) -> None:
        self.pulled: list[str] = []

    def list_kernels(self, competition: str, n: int) -> list[KernelInfo]:
        if competition == "no-such-comp":
            raise LookupError("404")
        return [
            KernelInfo(ref=f"user{i}/{competition}-nb{i}", author=f"user{i}", votes=100 - i) for i in range(n)
        ]

    def pull(self, ref: str, dest: Path) -> Path:
        self.pulled.append(ref)
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / (ref.split("/")[1] + ".ipynb")
        path.write_text('{"cells": [], "nbformat": 4, "nbformat_minor": 5, "metadata": {}}')
        return path


def test_discover_kaggle_builds_one_entry_per_competition() -> None:
    entries, skipped = discover_kaggle(
        FakeKaggle(), ["titanic", "no-such-comp", "spaceship-titanic"], per_competition=3
    )
    assert [e.name for e in entries] == ["kaggle-titanic", "kaggle-spaceship-titanic"]
    assert skipped == ["no-such-comp"]
    e = entries[0]
    assert e.source == "kaggle" and e.group == "titanic" and e.split == assign_split("kaggle:titanic")
    assert [k.ref for k in e.kernels] == ["user0/titanic-nb0", "user1/titanic-nb1", "user2/titanic-nb2"]


def test_fetch_kaggle_records_content_hash(tmp_path: Path) -> None:
    src = FakeKaggle()
    entry = kg("kaggle-titanic", "train", "titanic", ["u1", "u2"])
    updated = fetch_kaggle(entry, src, tmp_path)
    assert all(k.sha256 and len(k.sha256) == 64 for k in updated.kernels)
    assert len(list((tmp_path / "kaggle-titanic").rglob("*.ipynb"))) == 2
    fetch_kaggle(updated, src, tmp_path)  # idempotent: already on disk, not pulled again
    assert len(src.pulled) == 2


# ---- github discovery ----
def fake_search(query: str, per_page: int) -> list[dict[str, Any]]:
    return [
        {"full_name": "org/good-ml", "clone_url": "https://github.com/org/good-ml.git", "fork": False, "size": 5_000,
         "stargazers_count": 900, "archived": False},
        {"full_name": "org/forked", "clone_url": "https://github.com/org/forked.git", "fork": True, "size": 100,
         "stargazers_count": 50, "archived": False},
        {"full_name": "org/huge", "clone_url": "https://github.com/org/huge.git", "fork": False, "size": 9_000_000,
         "stargazers_count": 5000, "archived": False},
        {"full_name": "GokuMohandas/Made-With-ML", "clone_url": "https://github.com/GokuMohandas/Made-With-ML.git",
         "fork": False, "size": 4000, "stargazers_count": 40000, "archived": False},
    ]  # fmt: skip


def test_discover_github_filters_and_pins() -> None:
    existing = Manifest(repos=[gh("made-with-ml", "train", "gokumohandas/made-with-ml")])
    entries = discover_github(["q"], existing, limit=10, search=fake_search, head=lambda url: SHA)
    assert [e.name for e in entries] == ["org-good-ml"]
    e = entries[0]
    assert e.commit == SHA and e.group == "org/good-ml" and e.split == assign_split("github:org/good-ml")


def test_dedupe_corpus_drops_cross_split_copies(tmp_path: Path, tiny_repo: Path) -> None:
    import shutil

    from faultline.eval.build import dedupe_corpus

    for name in ("a", "b"):
        shutil.copytree(tiny_repo, tmp_path / name)
    manifest = Manifest(repos=[gh("a", "train"), gh("b", "test")])
    report = dedupe_corpus(manifest, tmp_path)
    assert report.cross_split == 5  # every file in b copies a
    assert all(k.startswith("b/") for k in report.excluded)
    assert report.kept_by_split == {"train": 5}


def test_discover_github_caps_per_query_and_skips_awesome_lists() -> None:
    def many(query: str, per_page: int) -> list[dict[str, Any]]:
        names = ["org/awesome-ml", *[f"org/proj{i}" for i in range(10)]]
        return [{"full_name": n, "clone_url": f"https://github.com/{n}.git", "size": 10} for n in names]

    entries = discover_github(
        ["q1"], Manifest(repos=[gh("x", "train")]), limit=10, search=many, head=lambda u: SHA
    )
    assert [e.group for e in entries] == [f"org/proj{i}" for i in range(5)]
