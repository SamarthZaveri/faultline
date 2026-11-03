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


# ---- A1: statistics computed on all data before the split ----
@pytest.mark.parametrize(
    "code",
    [
        "df['age'] = df['age'].fillna(df['age'].mean())\nX_tr, X_te, y_tr, y_te = train_test_split(df, y)\n",
        "X = (X - X.mean()) / X.std()\nX_tr, X_te, y_tr, y_te = train_test_split(X, y)\n",
        "m = df['age'].median()\ndf['age'].fillna(m, inplace=True)\nX_tr, X_te, y_tr, y_te = train_test_split(df, y)\n",
        (
            "df['city_enc'] = df['city'].map(df.groupby('city')['y'].mean())\n"
            "X_tr, X_te, y_tr, y_te = train_test_split(df, df['y'])\n"
        ),
    ],
)
def test_statistic_before_split(code: str) -> None:
    [(kind, _line)] = found(code)
    assert kind == "data.leakage.preprocessing_before_split"


@pytest.mark.parametrize(
    "code",
    [
        "print(df['age'].mean())\nX_tr, X_te, y_tr, y_te = train_test_split(df, y)\n",  # EDA only
        "avg = df['age'].mean()\nprint(avg)\nX_tr, X_te, y_tr, y_te = train_test_split(df, y)\n",
        "X_tr, X_te, y_tr, y_te = train_test_split(X, y)\nmu = X_tr.mean()\nX_tr = X_tr - mu\nX_te = X_te - mu\n",
    ],
)
def test_statistic_not_leaking(code: str) -> None:
    assert found(code) == []


# ---- slicing splits ----
def test_slicing_split_is_a_split() -> None:
    code = (
        "data = StandardScaler().fit_transform(data)\n"
        "n = int(len(data) * 0.8)\n"
        "train, test = data[:n], data[n:]\n"
    )
    assert found(code) == [("data.leakage.preprocessing_before_split", 6)]


def test_iloc_slicing_split_and_fit_on_test() -> None:
    code = (
        "train = df.iloc[:800]\ntest = df.iloc[800:]\nsc = StandardScaler()\ntest = sc.fit_transform(test)\n"
    )
    assert found(code) == [("data.leakage.test_in_training", 9)]


def test_unrelated_slices_are_not_a_split() -> None:
    code = "head = names[:5]\ntail = values[3:]\nX = StandardScaler().fit_transform(names)\n"
    assert found(code) == []


# ---- A3: resampling before the split ----
def test_smote_before_split() -> None:
    code = (
        "from imblearn.over_sampling import SMOTE\n"
        "X, y = SMOTE().fit_resample(X, y)\n"
        "X_tr, X_te, y_tr, y_te = train_test_split(X, y)\n"
    )
    assert found(code) == [("data.leakage.resampling_before_split", 7)]


def test_smote_on_train_split_is_clean() -> None:
    code = (
        "from imblearn.over_sampling import SMOTE\n"
        "X_tr, X_te, y_tr, y_te = train_test_split(X, y)\n"
        "X_tr, y_tr = SMOTE().fit_resample(X_tr, y_tr)\n"
    )
    assert found(code) == []


# ---- model trained on all data, then evaluated on a part of it ----
def test_model_fit_on_all_data_then_scored_on_test() -> None:
    code = (
        "model = LogisticRegression()\n"
        "model.fit(X, y)\n"
        "X_tr, X_te, y_tr, y_te = train_test_split(X, y)\n"
        "print(model.score(X_te, y_te))\n"
    )
    assert found(code) == [("data.leakage.test_in_training", 7)]


def test_final_refit_on_all_data_after_evaluation_is_clean() -> None:
    code = (
        "X_tr, X_te, y_tr, y_te = train_test_split(X, y)\n"
        "model = LogisticRegression().fit(X_tr, y_tr)\n"
        "print(model.score(X_te, y_te))\n"
        "final = LogisticRegression().fit(X, y)\n"
    )
    assert found(code) == []


# ---- A2: test set used for model selection ----
@pytest.mark.parametrize(
    "fit",
    [
        "model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], early_stopping_rounds=50)\n",
        "model.fit(X_tr, y_tr, validation_data=(X_te, y_te), epochs=10)\n",
    ],
)
def test_test_set_used_for_early_stopping(fit: str) -> None:
    code = "X_tr, X_te, y_tr, y_te = train_test_split(X, y)\n" + fit
    assert found(code) == [("data.leakage.tuning_on_test", 7)]


def test_validation_split_for_early_stopping_is_clean() -> None:
    code = (
        "X_full, X_test, y_full, y_test = train_test_split(X, y)\n"
        "X_tr, X_val, y_tr, y_val = train_test_split(X_full, y_full)\n"
        "model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])\n"
    )
    assert found(code) == []


def test_several_models_compared_on_test() -> None:
    code = (
        "X_tr, X_te, y_tr, y_te = train_test_split(X, y)\n"
        "a = LogisticRegression().fit(X_tr, y_tr)\n"
        "b = RandomForestClassifier().fit(X_tr, y_tr)\n"
        "print(a.score(X_te, y_te))\n"
        "print(b.score(X_te, y_te))\n"
    )
    assert found(code) == [("data.leakage.tuning_on_test", 10)]


def test_hyperparameter_loop_scored_on_test() -> None:
    code = (
        "X_tr, X_te, y_tr, y_te = train_test_split(X, y)\n"
        "for c in [0.1, 1, 10]:\n"
        "    m = LogisticRegression(C=c).fit(X_tr, y_tr)\n"
        "    print(c, m.score(X_te, y_te))\n"
    )
    assert found(code) == [("data.leakage.tuning_on_test", 9)]


def test_single_final_evaluation_is_clean() -> None:
    code = (
        "X_tr, X_te, y_tr, y_te = train_test_split(X, y)\n"
        "m = LogisticRegression().fit(X_tr, y_tr)\n"
        "pred = m.predict(X_te)\n"
        "print(m.score(X_te, y_te))\n"  # same model twice is still one evaluation
    )
    assert found(code) == []


def test_validation_slice_taken_first_is_still_validation() -> None:
    code = (
        "X_full, X_te, y_full, y_te = train_test_split(X, y)\n"
        "X_valid, X_train = X_full[:5000], X_full[5000:]\n"
        "y_valid, y_train = y_full[:5000], y_full[5000:]\n"
        "model.fit(X_train, y_train, validation_data=(X_valid, y_valid))\n"
    )
    assert found(code) == []


def test_parallel_assignment_keeps_names_separate() -> None:
    code = (
        "X_full, X_test, y_full, y_test = train_test_split(X, y)\n"
        "X_train, X_valid, y_train, y_valid = train_test_split(X_full, y_full)\n"
        "X_train, X_valid, X_test = X_train / 255.0, X_valid / 255.0, X_test / 255.0\n"
        "model.fit(X_train, y_train, validation_data=(X_valid, y_valid))\n"
    )
    assert found(code) == []
