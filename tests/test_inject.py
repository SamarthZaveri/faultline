import ast
import random

import pytest
from helpers import nb, py

from faultline.detectors import run_detectors
from faultline.inject.code import MUTATORS, Mutator

TRAIN_SCRIPT = py(
    """
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    def train(path):
        df = pd.read_csv(path)
        X = df.drop(columns=["label"])
        y = df["label"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = LogisticRegression().fit(X_train, y_train)
        return model.score(X_test, y_test)
    """,
    path="train.py",
)

TRAIN_NOTEBOOK = nb(
    [
        "import pandas as pd\nfrom sklearn.model_selection import train_test_split\n",
        "df = pd.read_csv('orders.csv')\nX, y = df.drop(columns=['label']), df['label']\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y)\n",
        "print(X_train.shape)\n",
    ]
)


def test_registry_has_one_mutator_per_target_type() -> None:
    assert {m.expected_type for m in MUTATORS} == {
        "pipeline.secret.hardcoded",
        "pipeline.injection.sql",
        "pipeline.injection.shell",
        "pipeline.deserialization.unsafe",
        "data.leakage.preprocessing_before_split",
        "data.leakage.feature_selection_full",
        "data.leakage.test_in_training",
    }


@pytest.mark.parametrize("mutator", MUTATORS, ids=lambda m: m.name)
@pytest.mark.parametrize("src", [TRAIN_SCRIPT, TRAIN_NOTEBOOK], ids=["script", "notebook"])
@pytest.mark.parametrize("seed", range(8))
def test_injected_fault_parses_and_is_detected(mutator: Mutator, src: object, seed: int) -> None:
    assert mutator.applicable(src)  # type: ignore[arg-type]
    mutated, fault = mutator.apply(src, random.Random(seed))  # type: ignore[arg-type]
    ast.parse(mutated.code)
    assert fault.file == mutated.path
    assert 1 <= fault.line_start <= fault.line_end <= len(mutated.code.splitlines())
    hits = [
        f
        for f in run_detectors(mutated)
        if f.type == fault.expected_type
        and fault.line_start <= (f.location.line_start or 0) <= fault.line_end
    ]
    assert hits, f"{mutator.name} seed={seed} not detected:\n{mutated.code}"


def test_notebook_cell_starts_shift_after_insertion() -> None:
    mutator = next(m for m in MUTATORS if m.name == "preprocess_before_split")
    mutated, fault = mutator.apply(TRAIN_NOTEBOOK, random.Random(0))
    n = fault.line_end - fault.line_start + 1
    assert mutated.cell_starts[:2] == TRAIN_NOTEBOOK.cell_starts[:2]
    assert mutated.cell_starts[3] == TRAIN_NOTEBOOK.cell_starts[3] + n


def test_split_mutators_need_a_split() -> None:
    no_split = py("import pandas as pd\ndf = pd.read_csv('x.csv')\n")
    applicable = {m.name for m in MUTATORS if m.applicable(no_split)}
    assert "preprocess_before_split" not in applicable
    assert "hardcoded_secret" in applicable


def test_templates_vary_across_seeds() -> None:
    for m in MUTATORS:
        variants = {m.apply(TRAIN_SCRIPT, random.Random(s))[0].code for s in range(8)}
        assert len(variants) >= 3, m.name
