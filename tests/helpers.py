"""Test helpers (importable as `helpers`; pytest puts tests/ on sys.path)."""

import json
import textwrap

from faultline.discovery.sources import SourceFile, SourceKind, notebook_to_code


def py(code: str, path: str = "pipeline.py") -> SourceFile:
    return SourceFile(path=path, kind=SourceKind.python, code=textwrap.dedent(code))


def notebook_json(cells: list[str]) -> str:
    return json.dumps(
        {
            "cells": [{"cell_type": "code", "source": c.splitlines(keepends=True)} for c in cells],
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {},
        }
    )


def nb(cells: list[str], path: str = "analysis.ipynb") -> SourceFile:
    code, starts = notebook_to_code(notebook_json(cells))
    return SourceFile(path=path, kind=SourceKind.notebook, code=code, cell_starts=starts)


# Built at runtime so no full credential pattern is ever committed.
FAKE_AWS_KEY = "AKIA" + "Q7XJ2M4K9P3R8T6W"
FAKE_GH_TOKEN = "ghp" + "_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
