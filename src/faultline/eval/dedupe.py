"""Exact and near-duplicate detection for corpus files (forks, translations, copy-paste).

Code is normalized (comments and whitespace dropped) and hashed for exact matches; near
matches use MinHash over token 5-grams with LSH banding, confirmed by the estimated Jaccard
similarity. Items are processed in order and the first occurrence wins, so pass train files
before dev before test: a test file that copies a train file is the one dropped.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

TOKEN = re.compile(r"[A-Za-z_]\w*|\d+(?:\.\d+)?|\S")
COMMENT = re.compile(r"#[^\n]*")
NUM_PERM = 64
BANDS = 16  # 16 bands x 4 rows: pairs with Jaccard ~0.6+ almost always collide in some band
ROWS = NUM_PERM // BANDS
MIN_SHINGLES = 20  # smaller files are only deduped exactly
SHINGLE = 5
_PRIME = (1 << 61) - 1
_PERMS = [
    (int.from_bytes(hashlib.blake2b(f"a{i}".encode(), digest_size=8).digest()) % _PRIME | 1,
     int.from_bytes(hashlib.blake2b(f"b{i}".encode(), digest_size=8).digest()) % _PRIME)
    for i in range(NUM_PERM)
]  # fmt: skip


def tokens(code: str) -> list[str]:
    return TOKEN.findall(COMMENT.sub("", code))


def normalized_hash(code: str) -> str:
    return hashlib.sha256(" ".join(tokens(code)).encode()).hexdigest()


def shingles(code: str, k: int = SHINGLE) -> set[int]:
    toks = tokens(code)
    return {
        int.from_bytes(hashlib.blake2b(" ".join(toks[i : i + k]).encode(), digest_size=8).digest())
        for i in range(max(len(toks) - k + 1, 0))
    }


def minhash(sh: Iterable[int]) -> tuple[int, ...]:
    values = list(sh)
    if not values:
        return tuple([_PRIME] * NUM_PERM)
    return tuple(min((a * x + b) % _PRIME for x in values) for a, b in _PERMS)


def jaccard_estimate(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a)


def find_duplicates(items: list[tuple[str, str]], threshold: float = 0.8) -> dict[str, tuple[str, str]]:
    """items: (key, code) in precedence order. Returns {duplicate_key: (kept_key, 'exact'|'near')}."""
    out: dict[str, tuple[str, str]] = {}
    exact: dict[str, str] = {}
    buckets: dict[tuple[int, tuple[int, ...]], list[str]] = {}
    sigs: dict[str, tuple[int, ...]] = {}  # insertion order = precedence
    order: dict[str, int] = {}
    for key, code in items:
        toks = tokens(code)
        if not toks:
            continue
        h = normalized_hash(code)
        if h in exact:
            out[key] = (exact[h], "exact")
            continue
        exact[h] = key
        sh = shingles(code)
        if len(sh) < MIN_SHINGLES:
            continue
        sig = minhash(sh)
        bands = [(b, sig[b * ROWS : (b + 1) * ROWS]) for b in range(BANDS)]
        candidates = {c for band in bands for c in buckets.get(band, [])}
        match = min(
            (c for c in candidates if jaccard_estimate(sig, sigs[c]) >= threshold),
            key=order.__getitem__,
            default=None,
        )
        if match is not None:
            out[key] = (match, "near")
            continue
        sigs[key] = sig
        order[key] = len(order)
        for band in bands:
            buckets.setdefault(band, []).append(key)
    return out
