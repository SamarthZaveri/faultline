from faultline.eval.dedupe import find_duplicates, jaccard_estimate, minhash, shingles

BASE = """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('../input/titanic/train.csv')
train['Age'] = train['Age'].fillna(train['Age'].median())
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
X = pd.get_dummies(train[features])
y = train['Survived']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X_train, y_train)
print(model.score(X_val, y_val))
"""


def test_exact_duplicate_ignores_comments_and_whitespace() -> None:
    copy = "# forked from someone\n" + BASE.replace("\n", "\n\n").replace("    ", "  ")
    dups = find_duplicates([("a/nb.ipynb", BASE), ("b/nb.ipynb", copy)])
    assert dups == {"b/nb.ipynb": ("a/nb.ipynb", "exact")}


def test_near_duplicate_small_edit() -> None:
    edited = BASE.replace("n_estimators=100, max_depth=5", "n_estimators=300, max_depth=7")
    dups = find_duplicates([("a", BASE), ("b", edited)])
    assert dups == {"b": ("a", "near")}


def test_different_code_is_kept() -> None:
    other = """
import numpy as np
import torch
def train_epoch(model, loader, opt):
    for xb, yb in loader:
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(xb), yb)
        loss.backward()
        opt.step()
    return float(loss)
"""
    assert find_duplicates([("a", BASE), ("b", other)]) == {}


def test_first_item_wins() -> None:
    dups = find_duplicates([("train/x", BASE), ("test/y", BASE), ("dev/z", BASE)])
    assert dups == {"test/y": ("train/x", "exact"), "dev/z": ("train/x", "exact")}


def test_tiny_files_only_dedupe_exactly() -> None:
    assert find_duplicates([("a", "import os\n"), ("b", "import sys\n")]) == {}
    assert find_duplicates([("a", ""), ("b", "")]) == {}  # empty files are not "duplicates"


def test_minhash_estimate_tracks_jaccard() -> None:
    a, b = shingles(BASE), shingles(BASE.replace("median", "mean"))
    true = len(a & b) / len(a | b)
    assert abs(jaccard_estimate(minhash(a), minhash(b)) - true) < 0.2
