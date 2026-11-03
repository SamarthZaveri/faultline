import pytest
from helpers import nb, py

from faultline.detectors.leakage import LeakageDetector

DET = LeakageDetector()

HEAD = (
    "import pandas as pd\n"
    "from sklearn.preprocessing import StandardScaler\n"
    "from sklearn.feature_selection import SelectKBest\n"
    "from sklearn.model_selection import train_test_split, cross_val_score\n"
    "from sklearn.linear_model import LogisticRegression\n"
)


def found(code: str) -> list[tuple[str, int | None]]:
    return [(f.type, f.location.line_start) for f in DET.detect(py(HEAD + code))]


def test_scaler_fit_before_split() -> None:
    code = (
        "X = df.drop(columns=['y'])\n"
        "scaler = StandardScaler()\n"
        "X = scaler.fit_transform(X)\n"
        "X_tr, X_te, y_tr, y_te = train_test_split(X, df['y'])\n"
    )
    assert found(code) == [("data.leakage.preprocessing_before_split", 8)]


def test_inline_fit_on_ancestor_of_split_input() -> None:
    code = "X_scaled = StandardScaler().fit_transform(X)\nX_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y)\n"
    assert found(code) == [("data.leakage.preprocessing_before_split", 6)]


def test_fit_before_cross_validation() -> None:
    code = "X2 = StandardScaler().fit_transform(X)\nscores = cross_val_score(LogisticRegression(), X2, y)\n"
    assert found(code) == [("data.leakage.preprocessing_before_split", 6)]


def test_feature_selection_on_full_data() -> None:
    code = "X = SelectKBest(k=5).fit_transform(X, y)\nX_tr, X_te, y_tr, y_te = train_test_split(X, y)\n"
    assert found(code) == [("data.leakage.feature_selection_full", 6)]


def test_fit_on_test_split() -> None:
    code = (
        "X_tr, X_te, y_tr, y_te = train_test_split(X, y)\n"
        "sc = StandardScaler()\n"
        "X_tr = sc.fit_transform(X_tr)\n"
        "X_te = sc.fit_transform(X_te)\n"
    )
    assert found(code) == [("data.leakage.test_in_training", 9)]


def test_correct_pipeline_is_clean() -> None:
    code = (
        "X_tr, X_te, y_tr, y_te = train_test_split(X, y)\n"
        "sc = StandardScaler().fit(X_tr)\n"
        "X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)\n"
        "m = LogisticRegression().fit(X_tr, y_tr)\n"
        "print(m.score(X_te, y_te))\n"
    )
    assert found(code) == []


def test_fit_on_train_then_validation_split_is_clean() -> None:
    code = (
        "X_tr, X_te, y_tr, y_te = train_test_split(X, y)\n"
        "X_tr = StandardScaler().fit_transform(X_tr)\n"
        "X_a, X_b, y_a, y_b = train_test_split(X_tr, y_tr)\n"
    )
    assert found(code) == []


def test_unrelated_fit_before_split_is_clean() -> None:
    code = "Z = StandardScaler().fit_transform(other)\nX_tr, X_te, y_tr, y_te = train_test_split(X, y)\n"
    assert found(code) == []


def test_different_functions_do_not_mix() -> None:
    code = (
        "def prep(X):\n    return StandardScaler().fit_transform(X)\n"
        "def split(X, y):\n    return train_test_split(X, y)\n"
    )
    assert found(code) == []


def test_notebook_cells_reported_with_cell_index() -> None:
    src = nb(
        [
            HEAD,
            "X = StandardScaler().fit_transform(X)\n",
            "X_tr, X_te, y_tr, y_te = train_test_split(X, y)\n",
        ]
    )
    [f] = DET.detect(src)
    assert f.type == "data.leakage.preprocessing_before_split"
    assert f.location.cell == 1


@pytest.mark.parametrize("code", ["def broken(:\n", ""])
def test_unparseable_or_empty(code: str) -> None:
    assert DET.detect(py(code)) == []


def test_column_transformer_fit_before_split() -> None:
    code = (
        "from sklearn.compose import ColumnTransformer\n"
        "from sklearn.preprocessing import OneHotEncoder\n"
        "ct = ColumnTransformer([('cat', OneHotEncoder(), ['city'])])\n"
        "encoded = ct.fit_transform(df)\n"
        "X = encoded.drop(columns=['y'])\n"
        "X_tr, X_te, y_tr, y_te = train_test_split(X, encoded['y'])\n"
    )
    assert found(code) == [("data.leakage.preprocessing_before_split", 9)]


def test_pipeline_fit_before_split() -> None:
    code = (
        "from sklearn.pipeline import make_pipeline\n"
        "pipe = make_pipeline(StandardScaler(), LogisticRegression())\n"
        "pipe.fit(X, y)\n"
        "X_tr, X_te, y_tr, y_te = train_test_split(X, y)\n"
        "print(pipe.score(X_te, y_te))\n"
    )
    assert found(code) == [("data.leakage.preprocessing_before_split", 8)]


def test_label_encoding_the_target_is_not_leakage() -> None:
    code = (
        "from sklearn.preprocessing import LabelEncoder\n"
        "df['label'] = LabelEncoder().fit_transform(df['label'])\n"
        "X_tr, X_te, y_tr, y_te = train_test_split(df.drop(columns=['label']), df['label'])\n"
    )
    assert found(code) == []
