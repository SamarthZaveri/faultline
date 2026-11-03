import zipfile
from pathlib import Path

import pytest
from helpers import notebook_json

from faultline.eval.yang import evaluate, load_benchmark

LEAKY = [
    "from sklearn.preprocessing import StandardScaler\nfrom sklearn.model_selection import train_test_split\n",
    "X = StandardScaler().fit_transform(X)\nX_tr, X_te, y_tr, y_te = train_test_split(X, y)\n",
]
CLEAN = [
    "from sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LogisticRegression\n",
    "X_tr, X_te, y_tr, y_te = train_test_split(X, y)\nm = LogisticRegression().fit(X_tr, y_tr)\n",
]


def make_bench(tmp_path: Path) -> Path:
    d = tmp_path / "yang2022"
    d.mkdir()
    with zipfile.ZipFile(d / "sample-notebooks.zip", "w") as z:
        z.writestr("sample_notebooks/2021-09-05-nb_1.ipynb", notebook_json(LEAKY))
        z.writestr("sample_notebooks/2021-09-05-nb_2.ipynb", notebook_json(CLEAN))
        z.writestr("sample_notebooks/2021-09-06-nb_3.ipynb", notebook_json(CLEAN))
        z.writestr("sample_notebooks/2021-09-06-nb_4.ipynb", notebook_json(LEAKY))
    (d / "ground-truth.csv").write_text(
        "nb,model,pre,overlap,multi,\n"
        "GitHub-data/notebooks/2021-09-05/nb_1.py,Y,Y (StandardScaler),N,N,,TRUE,\n"
        "GitHub-data/notebooks/2021-09-05/nb_2.py,Y,N,N,N,,FALSE,\n"
        "GitHub-data/notebooks/2021-09-06/nb_3.py,Y,Y (mean),N,Y,,TRUE,\n"  # leak we can't see
        "GitHub-data/notebooks/2021-09-06/nb_4.py,N,,,,,FALSE,\n"  # no model: unlabeled
    )
    (d / "analysis-result.csv").write_text(
        ",nb,msg,model,pre,overlap,multi\n"
        "0,GitHub-data/notebooks/2021-09-05/nb_1.py,Success!,True,True,False,False\n"
        "1,GitHub-data/notebooks/2021-09-05/nb_2.py,Success!,True,False,False,False\n"
        "2,GitHub-data/notebooks/2021-09-06/nb_3.py,Success!,True,True,False,True\n"
    )
    return d


def test_load_maps_files_labels_and_baseline(tmp_path: Path) -> None:
    items = {i.id: i for i in load_benchmark(make_bench(tmp_path))}
    assert set(items) == {"2021-09-05/nb_1", "2021-09-05/nb_2", "2021-09-06/nb_3", "2021-09-06/nb_4"}
    assert items["2021-09-05/nb_1"].labels == {"pre": True, "overlap": False, "multi": False}
    assert items["2021-09-06/nb_4"].labels == {"pre": None, "overlap": None, "multi": None}
    assert items["2021-09-06/nb_3"].baseline == {"pre": True, "overlap": False, "multi": True}
    assert {i.split for i in items.values()} <= {"dev", "test"}


def test_evaluate_counts_notebook_level_hits(tmp_path: Path) -> None:
    items = load_benchmark(make_bench(tmp_path))
    for i in items:
        object.__setattr__(i, "split", "dev")  # put everything in dev for the test
    report = evaluate(items, "dev")
    pre = report.ours["pre"]
    assert (pre.tp, pre.fp, pre.fn, pre.tn) == (1, 0, 1, 1)  # nb_1 hit, nb_3 missed, nb_2 clean
    assert report.baseline["pre"].tp == 2
    assert report.ours["multi"].fn == 1  # nb_3 is labeled multi but its code scores one model once
    assert report.labeled == 3


def test_test_half_is_held_out(tmp_path: Path) -> None:
    items = load_benchmark(make_bench(tmp_path))
    with pytest.raises(PermissionError):
        evaluate(items, "test")
