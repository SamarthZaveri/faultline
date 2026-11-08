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


def test_scaler_refit_on_test_is_a_bug_not_leakage() -> None:
    code = (
        "X_tr, X_te, y_tr, y_te = train_test_split(X, y)\n"
        "sc = StandardScaler()\n"
        "X_tr = sc.fit_transform(X_tr)\n"
        "X_te = sc.fit_transform(X_te)\n"
    )
    assert found(code) == []


def test_model_fit_on_test_then_scored_on_test() -> None:
    code = (
        "X_tr, X_te, y_tr, y_te = train_test_split(X, y)\n"
        "m = RandomForestClassifier().fit(X_te, y_te)\n"
        "print(m.score(X_te, y_te))\n"
    )
    assert found(code) == [("data.leakage.test_in_training", 7)]


def test_transformer_fit_on_train_plus_heldout_rows() -> None:
    code = (
        "tr, dev = train_test_split(df)\n"
        "full = pd.concat([tr, dev, test])\n"
        "X = TfidfVectorizer().fit_transform(full.text)\n"
    )
    assert found(code) == [("data.leakage.preprocessing_before_split", 8)]


def test_training_set_replaced_by_all_rows() -> None:
    code = (
        "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
        "X_train, y_train = X, y\n"
        "m = RandomForestClassifier().fit(X_train, y_train)\n"
        "print(m.score(X_test, y_test))\n"
    )
    assert found(code) == [("data.leakage.test_in_training", 8)]


def test_permutation_importance_on_validation_is_clean() -> None:
    code = (
        "tr_X, val_X, tr_y, val_y = train_test_split(X, y)\n"
        "m = RandomForestClassifier().fit(tr_X, tr_y)\n"
        "perm = PermutationImportance(m).fit(val_X, val_y)\n"
    )
    assert found(code) == []


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
    # a pipeline that ends in a model trains the model: fit on all rows, then scored on the test rows
    assert found(code) == [("data.leakage.test_in_training", 8)]


def test_transformer_pipeline_fit_before_split() -> None:
    code = (
        "from sklearn.pipeline import make_pipeline\n"
        "prep = make_pipeline(SimpleImputer(), StandardScaler())\n"
        "X = prep.fit_transform(X)\n"
        "X_tr, X_te, y_tr, y_te = train_test_split(X, y)\n"
    )
    assert found(code) == [("data.leakage.preprocessing_before_split", 8)]


def test_statistic_inside_a_helper_reaches_the_data() -> None:
    code = (
        "def freq(col):\n"
        "    vc = df[col].value_counts()\n"
        "    return vc\n"
        "df['c_cnt'] = df['c'].map(freq('c'))\n"
        "cross_val_score(m, df, y)\n"
    )
    assert ("data.leakage.preprocessing_before_split", 9) in found(code)


def test_statistic_used_only_inside_a_helper_is_clean() -> None:
    code = (
        "def reduce_mem(df):\n"
        "    for c in df.columns:\n"
        "        lo = df[c].min()\n"
        "        if lo > 0:\n"
        "            df[c] = df[c].astype('float32')\n"
        "    return df\n"
        "df = reduce_mem(df)\n"
        "cross_val_score(m, df, y)\n"
    )
    assert found(code) == []


def test_row_wise_statistics_are_features_not_leakage() -> None:
    code = "df['row_mean'] = df[cols].mean(axis=1)\ncross_val_score(m, df, y)\n"
    assert found(code) == []


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


def test_iloc_slicing_split_then_scaler_before_it() -> None:
    code = "df = StandardScaler().fit_transform(df)\ntrain = df.iloc[:800]\ntest = df.iloc[800:].values\n"
    assert found(code) == [("data.leakage.preprocessing_before_split", 6)]


def test_mask_split_is_a_split() -> None:
    code = "df['a'] = df['a'].fillna(df['a'].mean())\ntr = df[df.date < '2017']\nva = df[df.date >= '2017']\n"
    assert found(code) == [("data.leakage.preprocessing_before_split", 6)]


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
        "model.fit(X_tr, y_tr, validation_data=(X_te, y_te), epochs=10, callbacks=[es])\n",
        "model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], callbacks=[lgb.early_stopping(50)])\n",
        "model = xgb.train(params, dtrain, 500, evals=[(dvalid, 'v')], early_stopping_rounds=50)\n",
    ],
)
def test_test_set_used_for_early_stopping(fit: str) -> None:
    code = (
        "X_tr, X_te, y_tr, y_te = train_test_split(X, y)\n"
        "dtrain, dvalid = xgb.DMatrix(X_tr, y_tr), xgb.DMatrix(X_te, y_te)\n" + fit
    )
    assert found(code) == [("data.leakage.tuning_on_test", 8)]


def test_single_holdout_named_valid_still_counts() -> None:
    code = (
        "X_tr, X_valid, y_tr, y_valid = train_test_split(X, y)\n"
        "model.fit(X_tr, y_tr, eval_set=[(X_valid, y_valid)], early_stopping_rounds=50)\n"
    )
    assert found(code) == [("data.leakage.tuning_on_test", 7)]


def test_monitoring_only_validation_data_is_clean() -> None:
    code = (
        "X_tr, X_te, y_tr, y_te = train_test_split(X, y)\n"
        "model.fit(X_tr, y_tr, validation_data=(X_te, y_te), epochs=10)\n"
    )
    assert found(code) == []


def test_early_stopping_inside_cv_folds_is_clean() -> None:
    code = (
        "for trn, val in KFold(5).split(X):\n"
        "    model.fit(X.iloc[trn], y.iloc[trn], eval_set=[(X.iloc[val], y.iloc[val])], early_stopping_rounds=50)\n"
    )
    assert found(code) == []


def test_validation_split_for_early_stopping_is_clean() -> None:
    code = (
        "X_full, X_test, y_full, y_test = train_test_split(X, y)\n"
        "X_tr, X_val, y_tr, y_val = train_test_split(X_full, y_full)\n"
        "model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])\n"
    )
    assert found(code) == []


def test_models_compared_on_holdout_without_selection_is_not_flagged() -> None:
    code = (
        "X_tr, X_te, y_tr, y_te = train_test_split(X, y)\n"
        "a = LogisticRegression().fit(X_tr, y_tr)\n"
        "b = RandomForestClassifier().fit(X_tr, y_tr)\n"
        "print(a.score(X_te, y_te))\n"
        "print(b.score(X_te, y_te))\n"
    )
    assert found(code) == []


def test_blend_weights_searched_on_holdout() -> None:
    code = (
        "tr, dev = train_test_split(df)\n"
        "best = 1e9\n"
        "for w in np.arange(0, 1, 0.01):\n"
        "    err = rmsle(dev.y, w * p1 + (1 - w) * p2)\n"
        "    if err < best:\n"
        "        best, best_w = err, w\n"
    )
    assert found(code) == [("data.leakage.tuning_on_test", 9)]


def test_hyperparameter_loop_keeps_best_on_test() -> None:
    code = (
        "X_tr, X_te, y_tr, y_te = train_test_split(X, y)\n"
        "best_score = 0\n"
        "for c in [0.1, 1, 10]:\n"
        "    m = LogisticRegression(C=c).fit(X_tr, y_tr)\n"
        "    s = m.score(X_te, y_te)\n"
        "    if s > best_score:\n"
        "        best_score, best_c = s, c\n"
    )
    assert found(code) == [("data.leakage.tuning_on_test", 10)]


def test_hyperparameter_sweep_scored_on_holdout() -> None:
    # the pick may be made by eye from the printed scores: the holdout still chooses C
    code = (
        "X_tr, X_te, y_tr, y_te = train_test_split(X, y)\n"
        "for c in [0.1, 1, 10]:\n"
        "    m = LogisticRegression(C=c).fit(X_tr, y_tr)\n"
        "    print(c, m.score(X_te, y_te))\n"
    )
    assert found(code) == [("data.leakage.tuning_on_test", 9)]


def test_model_classes_compared_in_a_loop_are_not_a_sweep() -> None:
    code = (
        "X_tr, X_te, y_tr, y_te = train_test_split(X, y)\n"
        "for m in [LogisticRegression(), RandomForestClassifier()]:\n"
        "    m.fit(X_tr, y_tr)\n"
        "    print(m.score(X_te, y_te))\n"
    )
    assert found(code) == []


def test_hyperparameter_sweep_inside_cv_is_clean() -> None:
    code = "for c in [0.1, 1, 10]:\n    print(c, cross_val_score(LogisticRegression(C=c), X, y).mean())\n"
    assert found(code) == []


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


# ---- broader statistic and split shapes ----
def test_cv_inside_a_helper_function_still_sees_the_leak() -> None:
    code = (
        "X = StandardScaler().fit_transform(X)\n"
        "def train_model(X, y):\n"
        "    for trn, val in KFold(5).split(X):\n"
        "        pass\n"
        "train_model(X, y)\n"
    )
    assert found(code) == [("data.leakage.preprocessing_before_split", 6)]


def test_frequency_and_group_mean_encodings_before_cv() -> None:
    code = (
        "df['c_cnt'] = df['c'].map(df['c'].value_counts())\n"
        "df['amt_m'] = df.groupby('card')['amt'].transform('mean')\n"
        "cross_val_score(m, df, y)\n"
    )
    assert found(code) == [
        ("data.leakage.preprocessing_before_split", 6),
        ("data.leakage.preprocessing_before_split", 7),
    ]


def test_per_entity_aggregate_on_group_key_is_clean() -> None:
    det = LeakageDetector(group_keys=["household"])
    code = "df['m'] = df.groupby('household')['age'].transform('mean')\ncross_val_score(m, df, y)\n"
    assert [f.type for f in det.detect(py(code))] == []


def test_loop_alias_fill_before_split() -> None:
    code = (
        "for ds in [train, test]:\n"
        "    ds['Age'] = ds['Age'].fillna(ds['Age'].median())\n"
        "X_tr, X_te = train_test_split(train)\n"
    )
    assert found(code) == [("data.leakage.preprocessing_before_split", 7)]


def test_vocabulary_encoders_are_not_leakage() -> None:
    code = "df['c'] = OrdinalEncoder().fit_transform(df[['c']])\nX_tr, X_te = train_test_split(df)\n"
    assert found(code) == []


def test_transform_used_only_for_a_plot_is_clean() -> None:
    code = "emb = PCA(2).fit_transform(X)\nplt.scatter(emb[:, 0], emb[:, 1])\ncross_val_score(m, X, y)\n"
    assert found(code) == []


def test_missing_share_column_filter_is_clean() -> None:
    code = (
        "drop = [c for c in X.columns if X[c].isna().mean() > 0.9]\n"
        "X = X.drop(columns=drop)\n"
        "cross_val_score(m, X, y)\n"
    )
    assert found(code) == []


def test_importance_selection_before_cv() -> None:
    code = (
        "rf = RandomForestClassifier().fit(X, y)\n"
        "top = X.columns[np.argsort(rf.feature_importances_)[-20:]]\n"
        "X = X[top]\n"
        "cross_val_score(m, X, y)\n"
    )
    assert found(code) == [("data.leakage.feature_selection_full", 7)]


def test_importance_selection_on_train_split_is_clean() -> None:
    code = (
        "X_tr, X_te, y_tr, y_te = train_test_split(X, y)\n"
        "rf = RandomForestClassifier().fit(X_tr, y_tr)\n"
        "top = X.columns[np.argsort(rf.feature_importances_)[-20:]]\n"
        "X_tr, X_te = X_tr[top], X_te[top]\n"
    )
    assert found(code) == []


def test_undersampling_with_pandas_before_split() -> None:
    code = "df = df.drop(df[df.y == 0].sample(frac=0.7).index)\nX_tr, X_te = train_test_split(df)\n"
    assert found(code) == [("data.leakage.resampling_before_split", 6)]


def test_toy_dataset_split_is_ignored() -> None:
    code = (
        "X, y = make_classification(n_samples=100)\n"
        "X = StandardScaler().fit_transform(X)\n"
        "X_tr, X_te = train_test_split(X)\n"
    )
    assert found(code) == []


# ---- holdouts built without train_test_split ----
def test_slices_inside_dmatrix_are_a_split() -> None:
    code = (
        "dtrain = xgb.DMatrix(train[4000:, :], label=y[4000:])\n"
        "dvalid = xgb.DMatrix(train[:4000, :], label=y[:4000])\n"
        "model = xgb.train(params, dtrain, 500, [(dvalid, 'v')], early_stopping_rounds=50)\n"
    )
    assert found(code) == [("data.leakage.tuning_on_test", 8)]


def test_slices_inside_dmatrix_for_the_test_file_are_clean() -> None:
    code = (
        "dtrain = xgb.DMatrix(train[4000:, :], label=y[4000:])\n"
        "dtest = xgb.DMatrix(test)\n"
        "model = xgb.train(params, dtrain, 500, [(dtrain, 't')], early_stopping_rounds=50)\n"
    )
    assert found(code) == []


def test_index_pair_is_a_split() -> None:
    code = (
        "data_tr = data.iloc[idx_tr]\n"
        "data_va = data.iloc[idx_va]\n"
        "model.fit(data_tr[cols], data_tr.y, eval_set=[(data_va[cols], data_va.y)], early_stopping_rounds=50)\n"
    )
    assert found(code) == [("data.leakage.tuning_on_test", 8)]


def test_index_pair_inside_a_fold_loop_is_clean() -> None:
    code = (
        "for trn_idx, val_idx in KFold(5).split(data):\n"
        "    data_tr = data.iloc[trn_idx]\n"
        "    data_va = data.iloc[val_idx]\n"
        "    model.fit(data_tr[cols], data_tr.y, eval_set=[(data_va[cols], data_va.y)], early_stopping_rounds=50)\n"
    )
    assert found(code) == []


def test_mask_split_across_frames() -> None:
    code = (
        "late = pd.to_datetime(df['date']) >= '2016-10-01'\n"
        "x_valid = df.drop(['y'], axis=1)[late]\n"
        "y_valid = df['y'].values[late]\n"
        "df = df[~late]\n"
        "model.fit(df[cols], df.y, eval_set=[(x_valid, y_valid)], early_stopping_rounds=50)\n"
    )
    assert found(code) == [("data.leakage.tuning_on_test", 10)]


def test_sample_then_drop_is_a_split() -> None:
    code = (
        "df_val = df.sample(n=10000, random_state=0)\n"
        "df_train = df.drop(df_val.index)\n"
        "model.fit(df_train[cols], df_train.y, eval_set=[(df_val[cols], df_val.y)], early_stopping_rounds=50)\n"
    )
    assert found(code) == [("data.leakage.tuning_on_test", 8)]


def test_sample_without_drop_is_not_a_split() -> None:
    code = (
        "small = df.sample(n=10000, random_state=0)\n"
        "rest = df.drop(['y'], axis=1)\n"
        "model.fit(rest, df.y, eval_set=[(small[cols], small.y)], early_stopping_rounds=50)\n"
    )
    assert found(code) == []


def test_early_stopping_inside_a_helper_on_the_caller_holdout() -> None:
    code = (
        "def fit_model(dtrain, dvalid):\n"
        "    xgtrain = lgb.Dataset(dtrain[cols], label=dtrain.y)\n"
        "    xgvalid = lgb.Dataset(dvalid[cols], label=dvalid.y)\n"
        "    return lgb.train(params, xgtrain, valid_sets=[xgvalid], early_stopping_rounds=30)\n"
        "train_df, val_df = train_test_split(df)\n"
        "bst = fit_model(train_df, val_df)\n"
    )
    assert ("data.leakage.tuning_on_test", 9) in found(code)


def test_early_stopping_inside_a_helper_on_a_nested_validation_split_is_clean() -> None:
    code = (
        "def fit_model(dtrain, dvalid):\n"
        "    xgtrain = lgb.Dataset(dtrain[cols], label=dtrain.y)\n"
        "    xgvalid = lgb.Dataset(dvalid[cols], label=dvalid.y)\n"
        "    return lgb.train(params, xgtrain, valid_sets=[xgvalid], early_stopping_rounds=30)\n"
        "full_df, test_df = train_test_split(df)\n"
        "train_df, val_df = train_test_split(full_df)\n"
        "bst = fit_model(train_df, val_df)\n"
    )
    assert found(code) == []


def test_scaler_before_a_dead_split_is_clean() -> None:
    code = (
        "X = StandardScaler().fit_transform(X)\n"
        "X_a, X_b, y_a, y_b = train_test_split(X, y)\n"
        "X_a, X_b, y_a, y_b = train_test_split(X_raw, y)\n"
        "print(model.fit(X_a, y_a).score(X_b, y_b))\n"
    )
    assert found(code) == []


def test_scaler_before_a_split_that_is_read_is_flagged() -> None:
    code = (
        "X = StandardScaler().fit_transform(X)\n"
        "X_a, X_b, y_a, y_b = train_test_split(X, y)\n"
        "print(model.fit(X_a, y_a).score(X_b, y_b))\n"
        "X_a, X_b, y_a, y_b = train_test_split(X_raw, y)\n"
    )
    assert found(code) == [("data.leakage.preprocessing_before_split", 6)]


def test_days_since_first_date_is_clean() -> None:
    code = (
        "df['n_days'] = (df['Date'].dt.date - df['Date'].dt.date.min()).apply(lambda x: x.days)\n"
        "X_tr, X_te, y_tr, y_te = train_test_split(df, y)\n"
    )
    assert found(code) == []


def test_minimum_of_a_numeric_column_is_still_a_statistic() -> None:
    code = "df['x0'] = df['x'] - df['x'].min()\nX_tr, X_te, y_tr, y_te = train_test_split(df, y)\n"
    assert found(code) == [("data.leakage.preprocessing_before_split", 6)]


def test_single_shuffle_split_loop_is_a_holdout() -> None:
    code = (
        "sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)\n"
        "for tr_i, te_i in sss.split(X, y):\n"
        "    X_train, X_val = X[tr_i], X[te_i]\n"
        "    y_train, y_val = y[tr_i], y[te_i]\n"
        "model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50)\n"
    )
    assert found(code) == [("data.leakage.tuning_on_test", 10)]


def test_five_fold_shuffle_split_loop_is_not_a_holdout() -> None:
    code = (
        "sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2)\n"
        "for tr_i, te_i in sss.split(X, y):\n"
        "    X_train, X_val = X[tr_i], X[te_i]\n"
        "    y_train, y_val = y[tr_i], y[te_i]\n"
        "    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50)\n"
    )
    assert found(code) == []


def test_rows_kept_from_a_frame_are_not_a_holdout() -> None:
    code = (
        "train1 = train[:4000000]\n"
        "train = train[4000000:8000000]\n"
        "X_tr, X_va, y_tr, y_va = train_test_split(train, y)\n"
        "model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])\n"
    )
    assert found(code) == []


def test_frame_overwritten_by_its_training_piece_is_still_a_split() -> None:
    code = (
        "valid = train[:4000]\n"
        "train = train[4000:]\n"
        "model.fit(train[cols], train.y, eval_set=[(valid[cols], valid.y)], early_stopping_rounds=50)\n"
    )
    assert found(code) == [("data.leakage.tuning_on_test", 8)]


def test_column_assignment_keeps_the_frame_history() -> None:
    code = (
        "X = StandardScaler().fit_transform(X)\n"
        "df = pd.DataFrame(X)\n"
        "df['fold'] = -1\n"
        "for tr, va in KFold(5).split(X=df, y=y):\n"
        "    pass\n"
    )
    assert found(code) == [("data.leakage.preprocessing_before_split", 6)]


def test_selected_columns_saved_and_reloaded() -> None:
    code = (
        "sel = SelectKBest(k=50)\n"
        "sel.fit(X, y)\n"
        "X[X.columns[sel.get_support()]].to_csv('feats.csv')\n"
        "cols = pd.read_csv('../input/mine/feats.csv').columns\n"
        "scores = cross_val_score(LogisticRegression(), df[cols], y)\n"
    )
    assert ("data.leakage.feature_selection_full", 7) in found(code)


def test_a_different_file_read_back_is_not_linked() -> None:
    code = (
        "sel = SelectKBest(k=50)\n"
        "sel.fit(X, y)\n"
        "X[X.columns[sel.get_support()]].to_csv('feats.csv')\n"
        "cols = pd.read_csv('../input/other/train.csv').columns\n"
        "scores = cross_val_score(LogisticRegression(), df[cols], y)\n"
    )
    assert found(code) == []


def test_importances_on_all_rows_choose_columns_for_the_holdout() -> None:
    code = (
        "X_tr, X_va, y_tr, y_va = train_test_split(train, y)\n"
        "rf = RandomForestRegressor().fit(train, y)\n"
        "imp = pd.DataFrame({'f': train.columns, 'i': rf.feature_importances_})\n"
        "top = imp.sort_values('i')[:50]['f'].values\n"
        "va = X_va[top]\n"
    )
    assert ("data.leakage.feature_selection_full", 8) in found(code)


def test_importances_on_the_training_piece_are_clean() -> None:
    code = (
        "X_tr, X_va, y_tr, y_va = train_test_split(train, y)\n"
        "rf = RandomForestRegressor().fit(X_tr, y_tr)\n"
        "imp = pd.DataFrame({'f': train.columns, 'i': rf.feature_importances_})\n"
        "top = imp.sort_values('i')[:50]['f'].values\n"
        "va = X_va[top]\n"
    )
    assert found(code) == []


def test_importances_from_the_training_piece_then_cv_on_it_are_clean() -> None:
    code = (
        "X_tr, X_te, y_tr, y_te = train_test_split(X, y)\n"
        "rf = RandomForestRegressor().fit(X_tr, y_tr)\n"
        "keep = X_tr.columns[rf.feature_importances_ > 0.01]\n"
        "X_tr2 = X_tr[keep]\n"
        "scores = cross_val_score(LogisticRegression(), X_tr2, y_tr)\n"
    )
    assert found(code) == []


def test_importances_from_a_same_seed_split_are_clean() -> None:
    code = (
        "a_tr, a_te, b_tr, b_te = train_test_split(X, y, test_size=0.3, random_state=0)\n"
        "rf = RandomForestRegressor().fit(a_tr, b_tr)\n"
        "keep = X.columns[rf.feature_importances_ > 0.01]\n"
        "X_tr, X_te, y_tr, y_te = train_test_split(X[keep], y, test_size=0.3, random_state=0)\n"
    )
    assert found(code) == []


def test_importances_from_the_training_piece_then_cv_on_all_rows() -> None:
    code = (
        "a_tr, a_te, b_tr, b_te = train_test_split(X, y, test_size=0.2)\n"
        "rf = RandomForestRegressor().fit(a_tr, b_tr)\n"
        "keep = X.columns[rf.feature_importances_ > 0.01]\n"
        "scores = cross_val_score(LogisticRegression(), X[keep], y)\n"
    )
    assert ("data.leakage.feature_selection_full", 8) in found(code)


def test_loader_that_scales_its_own_data_before_the_split() -> None:
    code = (
        "def load():\n"
        "    data = pd.read_csv('train.csv')\n"
        "    X = StandardScaler().fit(data).transform(data)\n"
        "    return X\n"
        "X = load()\n"
        "X_tr, X_te, y_tr, y_te = train_test_split(X, y)\n"
    )
    assert ("data.leakage.preprocessing_before_split", 10) in found(code)


def test_loader_that_scales_only_for_a_plot_is_clean() -> None:
    code = (
        "def load():\n"
        "    data = pd.read_csv('train.csv')\n"
        "    Z = StandardScaler().fit(data).transform(data)\n"
        "    plt.plot(Z)\n"
        "    return data\n"
        "X = load()\n"
        "X_tr, X_te, y_tr, y_te = train_test_split(X, y)\n"
    )
    assert found(code) == []


def test_inline_slices_in_fit_and_predict_are_a_split() -> None:
    code = (
        "X = StandardScaler().fit_transform(X)\n"
        "net.fit(X[:200000], y[:200000])\n"
        "pred = net.predict_proba(X[200000:])\n"
    )
    assert found(code) == [("data.leakage.preprocessing_before_split", 6)]


def test_inline_slices_of_training_rows_only_are_clean() -> None:
    code = (
        "net.fit(X[:200000], y[:200000])\n"
        "pred = net.predict_proba(X[200000:])\n"
        "X2 = StandardScaler().fit_transform(X)\n"
    )
    assert found(code) == []


def test_concatenated_train_test_put_back_apart_is_not_a_holdout() -> None:
    code = (
        "df = pd.concat([train, test])\n"
        "df = StandardScaler().fit_transform(df)\n"
        "model.fit(df[: len(train)], y)\n"
        "pred = model.predict(df[len(train) :])\n"
    )
    assert found(code) == []
