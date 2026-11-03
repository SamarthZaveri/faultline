"""Find the files of a pipeline repo and normalize them for detectors.

Notebooks become one Python script (code cells concatenated, IPython magics blanked) so the
same AST detectors work on both, with `cell_starts` mapping lines back to cells.
"""

from __future__ import annotations

import ast
import json
import os
import re
from bisect import bisect_right
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from pathlib import Path

SKIP_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "site-packages",
        "__pycache__",
        ".tox",
        ".mypy_cache",
        ".ipynb_checkpoints",
        "build",
        "dist",
    }
)
MAX_BYTES = 1_000_000
CONFIG_SUFFIXES = frozenset(
    {".yaml", ".yml", ".ini", ".cfg", ".toml", ".env", ".json", ".properties", ".conf"}
)
REQUIREMENTS = re.compile(r"^requirements[\w.-]*\.txt$", re.IGNORECASE)
LOCKFILES = frozenset({"package-lock.json", "poetry.lock", "Pipfile.lock", "yarn.lock", "uv.lock"})
# Cell magics whose body is still Python.
PYTHON_CELL_MAGICS = frozenset({"time", "timeit", "capture", "prun"})


class SourceKind(StrEnum):
    python = "python"
    notebook = "notebook"
    requirements = "requirements"
    config = "config"


@dataclass(frozen=True)
class SourceFile:
    path: str  # posix, relative to repo root
    kind: SourceKind
    code: str  # for notebooks: the concatenated code cells
    cell_starts: tuple[int, ...] = ()  # 1-based line where each code cell starts

    @property
    def is_code(self) -> bool:
        return self.kind in (SourceKind.python, SourceKind.notebook)

    @cached_property
    def tree(self) -> ast.Module | None:
        if not self.is_code:
            return None
        try:
            return ast.parse(self.code)
        except (SyntaxError, ValueError):
            return None

    @cached_property
    def lines(self) -> list[str]:
        return self.code.splitlines()

    def line(self, n: int) -> str:
        return self.lines[n - 1] if 0 < n <= len(self.lines) else ""

    def cell_of(self, line: int) -> int | None:
        if not self.cell_starts:
            return None
        return max(bisect_right(self.cell_starts, line) - 1, 0)


def _is_magic(line: str) -> bool:
    s = line.lstrip()
    return s.startswith(("%", "!", "?")) or (s.endswith("?") and not s.startswith("#"))


def notebook_to_code(raw: str) -> tuple[str, tuple[int, ...]]:
    nb = json.loads(raw)
    cells = nb.get("cells")
    if cells is None:  # nbformat 3
        cells = [c for ws in nb.get("worksheets", []) for c in ws.get("cells", [])]
    out: list[str] = []
    starts: list[int] = []
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", cell.get("input", ""))
        text = "".join(src) if isinstance(src, list) else str(src)
        lines = text.splitlines()
        first = lines[0].lstrip() if lines else ""
        if first.startswith("%%") and first[2:].split(" ", 1)[0] not in PYTHON_CELL_MAGICS:
            lines = [""] * len(lines)  # %%bash, %%sql, ...: not Python
        else:
            lines = ["" if _is_magic(ln) else ln for ln in lines]
        try:
            ast.parse("\n".join(lines))
        except (SyntaxError, ValueError):
            lines = [""] * len(lines)  # e.g. Python 2 cells: drop the cell, keep the rest
        starts.append(len(out) + 1)
        out.extend(lines)
        out.append("")  # separator keeps cells from merging
    return "\n".join(out) + "\n", tuple(starts)


def _kind(name: str) -> SourceKind | None:
    suffix = Path(name).suffix.lower()
    if suffix == ".py":
        return SourceKind.python
    if suffix == ".ipynb":
        return SourceKind.notebook
    if REQUIREMENTS.match(name):
        return SourceKind.requirements
    if name in LOCKFILES:
        return None
    if suffix in CONFIG_SUFFIXES or name.startswith(".env"):
        return SourceKind.config
    return None


def load_source(path: Path, rel: str) -> SourceFile | None:
    kind = _kind(path.name)
    if kind is None or path.stat().st_size > MAX_BYTES:
        return None
    raw = path.read_text(encoding="utf-8", errors="replace")
    if kind is SourceKind.notebook:
        try:
            code, starts = notebook_to_code(raw)
        except (json.JSONDecodeError, AttributeError, TypeError):
            return None
        return SourceFile(path=rel, kind=kind, code=code, cell_starts=starts)
    return SourceFile(path=rel, kind=kind, code=raw)


def iter_sources(root: Path) -> Iterator[SourceFile]:
    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            path = Path(dirpath) / name
            src = load_source(path, path.relative_to(root).as_posix())
            if src is not None:
                yield src
