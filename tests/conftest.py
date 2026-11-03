import textwrap
from pathlib import Path

import pytest
from helpers import notebook_json


@pytest.fixture
def tiny_repo(tmp_path: Path) -> Path:
    """A small realistic pipeline: ingest -> clean -> train, with one notebook."""
    root = tmp_path / "tiny"
    (root / "pipeline").mkdir(parents=True)
    (root / "pipeline" / "ingest.py").write_text(
        textwrap.dedent(
            """
            import pandas as pd

            def ingest():
                df = pd.read_csv("https://example.com/orders.csv")
                df.to_parquet("data/raw_orders.parquet")
            """
        )
    )
    (root / "pipeline" / "clean.py").write_text(
        textwrap.dedent(
            """
            import pandas as pd

            def clean():
                df = pd.read_parquet("data/raw_orders.parquet")
                df = df.dropna()
                df.to_parquet("data/clean_orders.parquet")
            """
        )
    )
    (root / "pipeline" / "train.py").write_text(
        textwrap.dedent(
            """
            import pandas as pd
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split

            def train():
                df = pd.read_parquet("data/clean_orders.parquet")
                X = df.drop(columns=["label"])
                y = df["label"]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                model = LogisticRegression().fit(X_train, y_train)
                return model.score(X_test, y_test)
            """
        )
    )
    (root / "explore.ipynb").write_text(
        notebook_json(
            [
                "import pandas as pd\nfrom sklearn.model_selection import train_test_split\n",
                "%matplotlib inline\ndf = pd.read_parquet('data/clean_orders.parquet')\n",
                "X, y = df.drop(columns=['label']), df['label']\nX_tr, X_te, y_tr, y_te = train_test_split(X, y)\n",
            ]
        )
    )
    (root / "requirements.txt").write_text("pandas==2.2.2\nscikit-learn\n")
    return root
