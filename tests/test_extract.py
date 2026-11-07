from pathlib import Path

from helpers import nb, notebook_json

from faultline.eval.corpus import CorpusEntry, KaggleKernel, Manifest
from faultline.eval.extract import extract_notebook, shell_lines, write_extracts

CELLS = [
    "!pip install sklearn\nimport pandas as pd\nimport matplotlib.pyplot as plt",
    "df = pd.read_csv('train.csv')\ndf.head()\nplt.hist(df['Age'])\n# a comment\n\ndf.shape",
    "train = df.head(800)\nX = (df - df.mean()) / df.std()\nprint(df.columns)\nprint(accuracy_score(y, p))",
]


def write_nb(path: Path, cells: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(notebook_json(cells), encoding="utf-8")
    return path


def numbers(lines: list[str]) -> dict[int, str]:
    return {int(s[:5]): s[7:] for s in lines}


def test_extract_keeps_labelable_lines_with_detector_line_numbers(tmp_path: Path) -> None:
    lines, cut = extract_notebook(write_nb(tmp_path / "a.ipynb", CELLS))
    kept = numbers(lines)
    code = nb(CELLS).lines
    for n, text in kept.items():
        if not text.startswith("!"):
            assert code[n - 1].rstrip() == text  # same numbering as the detectors
    assert "train = df.head(800)" in kept.values()  # a positional split, not display
    assert "X = (df - df.mean()) / df.std()" in kept.values()
    assert "print(accuracy_score(y, p))" in kept.values()  # reports which data is scored
    assert cut == 0


def test_extract_drops_display_comments_and_blanks(tmp_path: Path) -> None:
    kept = set(numbers(extract_notebook(write_nb(tmp_path / "a.ipynb", CELLS))[0]).values())
    assert not kept & {"df.head()", "plt.hist(df['Age'])", "# a comment", "df.shape", "print(df.columns)"}


def test_shell_lines_are_kept_at_their_code_line(tmp_path: Path) -> None:
    path = write_nb(tmp_path / "a.ipynb", CELLS)
    assert shell_lines(path.read_text(encoding="utf-8")) == {1: "!pip install sklearn"}
    assert numbers(extract_notebook(path)[0])[1] == "!pip install sklearn"


def test_write_extracts_follows_the_manifest_and_skips_excluded(tmp_path: Path) -> None:
    repos = tmp_path / "repos"
    write_nb(repos / "kaggle-toy" / "ann__one" / "one.ipynb", CELLS)
    write_nb(repos / "kaggle-toy" / "bob__two" / "two.ipynb", CELLS)
    write_nb(repos / "kaggle-toy" / "cat__dropped" / "dropped.ipynb", CELLS)  # not in the manifest
    entry = CorpusEntry(
        name="kaggle-toy",
        source="kaggle",
        split="train",
        competition="toy",
        kernels=[KaggleKernel(ref="ann/one", author="ann"), KaggleKernel(ref="bob/two", author="bob")],
    )
    excluded = {"kaggle-toy/bob__two/two.ipynb"}
    n = write_extracts(Manifest(repos=[entry]), repos, "train", excluded, tmp_path / "out")
    text = (tmp_path / "out" / "train" / "toy.txt").read_text(encoding="utf-8")
    assert n == 1
    assert "=== ann__one/one.ipynb" in text
    assert "two.ipynb" not in text and "dropped" not in text
