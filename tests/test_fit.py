from helpers import nb, py

from faultline.contract import ColumnProfile, DatasetProfile
from faultline.detectors.fit import DataFitDetector

HEAD = (
    "import numpy as np\n"
    "import pandas as pd\n"
    "from sklearn.model_selection import train_test_split, cross_val_score, GroupShuffleSplit, TimeSeriesSplit\n"
    "from sklearn.linear_model import LogisticRegression\n"
    "from sklearn.ensemble import RandomForestClassifier\n"
    "from sklearn.metrics import accuracy_score, f1_score\n"
)  # 6 lines: code below starts at line 7


def found(code: str, *profiles: DatasetProfile) -> list[tuple[str, int | None]]:
    return [(f.type, f.location.line_start) for f in DataFitDetector(profiles).detect(py(HEAD + code))]


def col(name: str, kind: str = "numeric", **kw: object) -> ColumnProfile:
    return ColumnProfile.model_validate(
        {"name": name, "kind": kind, "missing_rate": 0.0, "distinct": 100, **kw}
    )


def profile(**kw: object) -> DatasetProfile:
    base: dict[str, object] = {
        "name": "data/orders.csv",
        "rows": 50_000,
        "columns": [col("amount"), col("x1"), col("label")],
        "target": "label",
        "task": "classification",
        "n_classes": 2,
        "minority_share": 0.5,
    }
    return DatasetProfile.model_validate(base | kw)


READ = "df = pd.read_csv('orders.csv')\n"
SPLIT = "X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)\n"


# --- temporal ---------------------------------------------------------------------------------


def test_lag_features_then_random_split() -> None:
    code = (
        READ
        + "df['lag_1'] = df['sales'].shift(1)\n"
        + "df['roll_7'] = df['sales'].rolling(7).mean()\n"
        + "X, y = df.drop(columns=['sales']), df['sales']\n"
        + SPLIT
    )
    assert found(code) == [("data.leakage.temporal_split", 11)]


def test_rolling_only_for_a_plot_is_not_time_evidence() -> None:
    code = (
        READ
        + "df['sales'].rolling(7).mean().plot()\nX, y = df.drop(columns=['sales']), df['sales']\n"
        + SPLIT
    )
    assert found(code) == []


def test_numpy_diff_is_not_a_lag_feature() -> None:
    code = READ + "df['d'] = np.diff(df['a'].values, prepend=0)\nX, y = df, df['y']\n" + SPLIT
    assert found(code) == []


def test_ordered_split_or_time_series_cv_is_fine() -> None:
    lag = READ + "df['lag_1'] = df['sales'].shift(1)\nX, y = df, df['sales']\n"
    assert found(lag + "X_tr, X_te, y_tr, y_te = train_test_split(X, y, shuffle=False)\n") == []
    assert (
        found(lag + "cv = TimeSeriesSplit(5)\nscores = cross_val_score(LogisticRegression(), X, y, cv=cv)\n")
        == []
    )


def test_time_ordered_profile_flags_random_split() -> None:
    p = profile(time_columns=["date"], time_ordered=True)
    assert found(READ + "X, y = df.drop(columns=['label']), df['label']\n" + SPLIT, p) == [
        ("data.leakage.temporal_split", 9)
    ]


def test_profile_with_unordered_time_column_is_fine() -> None:
    p = profile(time_columns=["signup_date"], time_ordered=False)
    assert found(READ + "X, y = df.drop(columns=['label']), df['label']\n" + SPLIT, p) == []


def test_cross_val_score_with_default_folds_counts_as_random() -> None:
    code = (
        READ
        + "df['lag_1'] = df['sales'].shift(1)\nscores = cross_val_score(LogisticRegression(), X, y, cv=5)\n"
    )
    assert found(code) == [("data.leakage.temporal_split", 9)]


# --- groups -----------------------------------------------------------------------------------


def test_per_entity_aggregate_then_random_split() -> None:
    code = (
        READ
        + "df['avg_spend'] = df.groupby('customer_id')['amount'].transform('mean')\n"
        + "X, y = df.drop(columns=['label']), df['label']\n"
        + SPLIT
    )
    assert found(code) == [("data.leakage.group_split", 10)]


def test_per_entity_aggregate_merged_back() -> None:
    code = (
        READ
        + "agg = df.groupby('user_id').agg(n=('amount', 'count'))\n"
        + "df = df.merge(agg, on='user_id')\n"
        + "X, y = df.drop(columns=['label']), df['label']\n"
        + SPLIT
    )
    assert found(code) == [("data.leakage.group_split", 11)]


def test_grouping_by_a_category_is_not_an_entity() -> None:
    code = (
        READ
        + "df['avg'] = df.groupby('region')['amount'].transform('mean')\nX, y = df, df['label']\n"
        + SPLIT
    )
    assert found(code) == []


def test_group_split_is_fine() -> None:
    code = (
        READ
        + "df['avg_spend'] = df.groupby('customer_id')['amount'].transform('mean')\n"
        + "gss = GroupShuffleSplit(n_splits=1, test_size=0.2)\n"
        + "tr, te = next(gss.split(X, y, groups=df['customer_id']))\n"
    )
    assert found(code) == []


def test_entity_key_in_profile_flags_random_split() -> None:
    p = profile(group_keys=["patient_id"])
    assert found(READ + "X, y = df.drop(columns=['label']), df['label']\n" + SPLIT, p) == [
        ("data.leakage.group_split", 9)
    ]


# --- stratify and metrics ---------------------------------------------------------------------


def test_class_weights_but_no_stratify() -> None:
    code = (
        READ
        + "X, y = df.drop(columns=['label']), df['label']\n"
        + SPLIT
        + "model = RandomForestClassifier(class_weight='balanced').fit(X_tr, y_tr)\n"
        + "print(f1_score(y_te, model.predict(X_te)))\n"
    )
    assert found(code) == [("data.fit.unstratified_split", 9)]


def test_stratified_split_is_fine() -> None:
    code = (
        READ
        + "X, y = df.drop(columns=['label']), df['label']\n"
        + "X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y)\n"
        + "model = RandomForestClassifier(class_weight='balanced').fit(X_tr, y_tr)\n"
        + "print(f1_score(y_te, model.predict(X_te)))\n"
    )
    assert found(code) == []


def test_no_imbalance_evidence_no_finding() -> None:
    code = (
        READ
        + "X, y = df.drop(columns=['label']), df['label']\n"
        + SPLIT
        + "model = LogisticRegression().fit(X_tr, y_tr)\n"
        + "print(accuracy_score(y_te, model.predict(X_te)))\n"
    )
    assert found(code) == []


def test_accuracy_only_on_imbalanced_profile() -> None:
    p = profile(minority_share=0.03)
    code = (
        READ
        + "X, y = df.drop(columns=['label']), df['label']\n"
        + "X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y)\n"
        + "model = LogisticRegression().fit(X_tr, y_tr)\n"
        + "print(accuracy_score(y_te, model.predict(X_te)))\n"
    )
    [finding] = DataFitDetector([p]).detect(py(HEAD + code))
    assert (finding.type, finding.location.line_start) == ("data.fit.metric_mismatch", 11)
    assert "97%" in finding.evidence.explanation  # the always-majority baseline


def test_accuracy_next_to_f1_is_fine() -> None:
    p = profile(minority_share=0.03)
    code = (
        READ
        + "X, y = df.drop(columns=['label']), df['label']\n"
        + "X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y)\n"
        + "model = LogisticRegression().fit(X_tr, y_tr)\n"
        + "print(accuracy_score(y_te, model.predict(X_te)), f1_score(y_te, model.predict(X_te)))\n"
    )
    assert found(code, p) == []


# --- profile-only rules -----------------------------------------------------------------------


def test_column_missing_from_the_data() -> None:
    code = READ + "X = df[['amount', 'discount']]\ny = df['label']\n" + SPLIT
    assert found(code, profile()) == [("data.fit.schema_mismatch", 8)]


def test_columns_the_code_creates_or_merges_in_are_fine() -> None:
    made = READ + "df['discount'] = df['amount'] * 0.1\nX = df[['amount', 'discount']]\ny = df['label']\n"
    assert found(made + SPLIT, profile()) == []
    merged = (
        READ + "df = df.merge(pd.read_csv('promos.csv'), on='promo')\nX = df[['discount']]\ny = df['label']\n"
    )
    assert found(merged + SPLIT, profile()) == []


def test_profile_for_another_file_is_not_used() -> None:
    code = "df = pd.read_csv('customers.csv')\nX = df[['discount']]\ny = df['label']\n" + SPLIT
    assert found(code, profile()) == []


def test_id_column_left_in_features() -> None:
    p = profile(columns=[col("PassengerId", identifier=True), col("Age"), col("label")])
    code = READ + "X, y = df.drop(columns=['label']), df['label']\n" + SPLIT
    assert found(code, p) == [("data.fit.id_as_feature", 8)]


def test_id_column_dropped_or_used_as_index_is_fine() -> None:
    p = profile(columns=[col("PassengerId", identifier=True), col("Age"), col("label")])
    dropped = READ + "X, y = df.drop(columns=['label', 'PassengerId']), df['label']\n" + SPLIT
    assert found(dropped, p) == []
    indexed = "df = pd.read_csv('orders.csv', index_col='PassengerId')\nX, y = df.drop(columns=['label']), df['label']\n"
    assert found(indexed + SPLIT, p) == []


def test_free_text_columns_are_not_ids() -> None:
    p = profile(columns=[col("review_text", "categorical", identifier=True), col("label")])
    assert found(READ + "X, y = df.drop(columns=['label']), df['label']\n" + SPLIT, p) == []


def test_duplicate_rows_split_without_dedupe() -> None:
    p = profile(duplicate_rate=0.04)
    code = READ + "X, y = df.drop(columns=['label']), df['label']\n" + SPLIT
    assert found(code, p) == [("data.leakage.duplicate_across_split", 9)]
    assert (
        found(READ + "df = df.drop_duplicates()\nX, y = df.drop(columns=['label']), df['label']\n" + SPLIT, p)
        == []
    )


def test_single_split_of_small_data() -> None:
    p = profile(rows=300)
    code = READ + "X, y = df.drop(columns=['label']), df['label']\n" + SPLIT
    assert found(code, p) == [("data.fit.single_split_small", 9)]
    cv = (
        READ
        + "X, y = df.drop(columns=['label']), df['label']\nscores = cross_val_score(LogisticRegression(), X, y)\n"
    )
    assert found(cv, p) == []


def test_notebook_findings_point_at_the_cell() -> None:
    src = nb(
        [
            "import pandas as pd\nfrom sklearn.model_selection import train_test_split\n",
            "df = pd.read_csv('sales.csv')\ndf['lag_1'] = df['sales'].shift(1)\n",
            "X_tr, X_te, y_tr, y_te = train_test_split(df.drop(columns=['sales']), df['sales'])\n",
        ]
    )
    [finding] = DataFitDetector().detect(src)
    assert finding.type == "data.leakage.temporal_split"
    assert finding.location.cell == 2
    assert finding.tier == "tier1"


def test_sampled_holdout_without_stratify() -> None:
    code = (
        READ
        + "val = df.sample(frac=0.2, random_state=0)\n"
        + "tr = df.drop(val.index)\n"
        + "model = RandomForestClassifier(class_weight='balanced').fit(tr[cols], tr['label'])\n"
        + "print(f1_score(val['label'], model.predict(val[cols])))\n"
    )
    assert ("data.fit.unstratified_split", 9) in found(code)


def test_index_pair_from_a_stratified_splitter_is_fine() -> None:
    code = (
        READ
        + "tr_idx, va_idx = next(StratifiedShuffleSplit(1, test_size=0.2).split(df, df['label']))\n"
        + "tr = df.iloc[tr_idx]\n"
        + "va = df.iloc[va_idx]\n"
        + "model = RandomForestClassifier(class_weight='balanced').fit(tr[cols], tr['label'])\n"
        + "print(f1_score(va['label'], model.predict(va[cols])))\n"
    )
    assert ("data.fit.unstratified_split", 10) not in found(code)
    assert ("data.fit.unstratified_split", 11) not in found(code)


def test_tflearn_validation_set_is_an_unstratified_holdout() -> None:
    code = (
        READ
        + "X, y = df.drop(columns=['label']), df['label']\n"
        + "model = RandomForestClassifier(class_weight='balanced')\n"
        + "net.fit(X, y, validation_set=0.2)\n"
    )
    assert ("data.fit.unstratified_split", 10) in found(code)
