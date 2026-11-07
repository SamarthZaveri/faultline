"""Compact review extracts of Kaggle notebooks, for hand labeling (corpus/labels/notebooks/).

Each notebook becomes its code lines numbered exactly as detectors number them (code cells
concatenated, faultline.discovery.sources), minus what can't carry a label: blank lines, comments,
plots and display calls. Shell lines (`!pip install ...`) are kept with their text, since the code
view blanks them. Detector findings are left out on purpose so labels aren't anchored to rules.
Extracts hold third-party code, so they live under corpus/extracts/ (git-ignored).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

from faultline.discovery.sources import load_source
from faultline.eval.corpus import DEFAULT_MANIFEST, Manifest, Split

DEFAULT_EXTRACTS = DEFAULT_MANIFEST.parent / "extracts"
MAX_LINES = 400

DISPLAY = re.compile(
    r"^\s*(plt|sns|px|go|ff|msno|ax|axes|axs|fig|wordcloud|plotly|py)\b[\w\[\]\-, ]*[.(]"
    r"|^\s*(print|display|pprint|show)\s*\("
    # bare expressions only: `df.head()` is display, `train = df.head(800)` is a split
    r"|^\s*[\w.\[\]'\"]+\.(head|tail|describe|info|sample|plot|hist|style|value_counts)\([^=]*\)\s*$"
    r"|^\s*[\w.\[\]'\"]+\.(shape|dtypes|columns)\s*$"
    r"|^\s*[A-Za-z_]\w*\s*$"
)
# a print that reports a score shows which data is being evaluated
METRIC = re.compile(r"score|accura|auc|rmse|mae|mse|loss|f1|evaluat|metric|report|confusion", re.IGNORECASE)


def shell_lines(raw: str) -> dict[int, str]:
    """`!cmd` / `%pip` lines of a notebook, keyed by their line in the concatenated code."""
    nb = json.loads(raw)
    cells = nb.get("cells")
    if cells is None:
        cells = [c for ws in nb.get("worksheets", []) for c in ws.get("cells", [])]
    out: dict[int, str] = {}
    line = 1
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", cell.get("input", ""))
        lines = ("".join(src) if isinstance(src, list) else str(src)).splitlines()
        for i, text in enumerate(lines):
            if text.lstrip().startswith(("!", "%pip", "%conda")):
                out[line + i] = text.strip()
        line += len(lines) + 1
    return out


def extract_notebook(path: Path) -> tuple[list[str], int]:
    """Numbered lines worth labeling, and how many more were cut at MAX_LINES."""
    src = load_source(path, path.name)
    if src is None or not src.is_code:
        return [], 0
    shells = (
        shell_lines(path.read_text(encoding="utf-8", errors="replace")) if path.suffix == ".ipynb" else {}
    )
    kept: list[str] = []
    for n, text in enumerate(src.lines, start=1):
        if n in shells:
            kept.append(f"{n:5d}  {shells[n]}")
            continue
        s = text.strip()
        if not s or s.startswith("#") or (DISPLAY.search(text) and not METRIC.search(text)):
            continue
        kept.append(f"{n:5d}  {text.rstrip()}")
    return kept[:MAX_LINES], max(0, len(kept) - MAX_LINES)


def iter_notebooks(
    manifest: Manifest, repos: Path, split: Split, excluded: set[str]
) -> Iterator[tuple[str, str, Path]]:
    """(competition, path relative to the entry's checkout, file) for kept Kaggle notebooks."""
    for entry in manifest.in_split(split):
        root = repos / entry.name
        if entry.source != "kaggle":
            continue
        for kernel in entry.kernels:  # not a directory walk: kernels dropped from the manifest stay out
            folder = root / kernel.dirname
            for path in sorted(folder.glob("*")) if folder.is_dir() else []:
                rel = path.relative_to(root).as_posix()
                if path.suffix in (".ipynb", ".py") and f"{entry.name}/{rel}" not in excluded:
                    yield entry.competition or entry.name, rel, path


def write_extracts(manifest: Manifest, repos: Path, split: Split, excluded: set[str], out: Path) -> int:
    """One file per competition under out/<split>/; returns the number of notebooks written."""
    by_comp: dict[str, list[str]] = {}
    count = 0
    for comp, rel, path in iter_notebooks(manifest, repos, split, excluded):
        lines, cut = extract_notebook(path)
        block = [f"=== {rel}", *lines]
        if cut:
            block.append(f"  ... {cut} more lines not shown")
        by_comp.setdefault(comp, []).extend([*block, ""])
        count += 1
    (out / split).mkdir(parents=True, exist_ok=True)
    for comp, lines in by_comp.items():
        (out / split / f"{comp}.txt").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return count
