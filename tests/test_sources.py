from pathlib import Path

from helpers import notebook_json

from faultline.discovery.sources import SourceKind, iter_sources, notebook_to_code


def test_notebook_magics_blanked_and_cells_mapped() -> None:
    code, starts = notebook_to_code(notebook_json(["import os\n", "%time x = 1\n!ls\ny = 2\n"]))
    lines = code.splitlines()
    assert starts == (1, 3)
    assert lines[0] == "import os"
    assert lines[2] == "" and lines[3] == ""  # magics blanked, line count preserved
    assert lines[4] == "y = 2"


def test_unparseable_cell_is_blanked_not_fatal() -> None:
    code, _ = notebook_to_code(notebook_json(["print 'py2'\n", "x = 1\n"]))
    assert "print 'py2'" not in code
    assert "x = 1" in code


def test_iter_sources_classifies_and_skips(tiny_repo: Path) -> None:
    (tiny_repo / ".venv" / "lib").mkdir(parents=True)
    (tiny_repo / ".venv" / "lib" / "junk.py").write_text("x = 1\n")
    (tiny_repo / "config.yaml").write_text("db: {host: localhost}\n")
    kinds = {s.path: s.kind for s in iter_sources(tiny_repo)}
    assert kinds["pipeline/train.py"] is SourceKind.python
    assert kinds["explore.ipynb"] is SourceKind.notebook
    assert kinds["requirements.txt"] is SourceKind.requirements
    assert kinds["config.yaml"] is SourceKind.config
    assert not any(p.startswith(".venv") for p in kinds)


def test_cell_of_line(tiny_repo: Path) -> None:
    nb = next(s for s in iter_sources(tiny_repo) if s.kind is SourceKind.notebook)
    assert nb.cell_of(1) == 0
    assert nb.cell_of(nb.cell_starts[2]) == 2
