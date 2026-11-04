"""Rows that appear in both training and test data (data.leakage.duplicate_across_split).

Static rules can't see this: the code splits correctly, but the raw data already holds copies
(re-scraped pages, resubmitted records, augmented images saved as new rows), so the split puts
copies on both sides and the test score rewards memorization. Rows are compared by content
hashes of the shared columns, leaving out the target and id-like columns (an id would make every
row look unique). Near-duplicates are not covered yet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from faultline.data.frame import as_str_columns, data_finding, is_identifier, row_hashes, rows_preview
from faultline.schema import Finding, Severity, Tier


def shared_rows(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Positions of test rows whose content also appears in train."""
    return np.flatnonzero(np.isin(row_hashes(test, columns), row_hashes(train, columns)))


class SplitOverlapDetector:
    name = "split_overlap"
    version = "0.1"

    def __init__(self, min_rate: float = 0.01) -> None:
        self.min_rate = min_rate

    @property
    def id(self) -> str:
        return f"{self.name}@{self.version}"

    def check(
        self, train: pd.DataFrame, test: pd.DataFrame, *, dataset: str, target: str | None = None
    ) -> list[Finding]:
        """Row numbers in findings are 0-based positions in `test`."""
        tr, te = as_str_columns(train), as_str_columns(test)
        columns = [
            c
            for c in te.columns
            if c in tr.columns and c != target and not is_identifier(te[c]) and not is_identifier(tr[c])
        ]
        if not columns or len(te) == 0:
            return []
        rows = shared_rows(tr, te, columns)
        rate = len(rows) / len(te)
        if rate < self.min_rate:
            return []
        return [
            data_finding(
                detector=self.id,
                type="data.leakage.duplicate_across_split",
                title=f"{rate:.1%} of test rows also appear in the training data",
                severity=Severity.high if rate >= 0.05 else Severity.medium,
                confidence=0.95,
                dataset=dataset,
                columns=columns if len(columns) <= 10 else [],
                rule_id="overlap.exact",
                explanation=f"{len(rows)} of {len(te)} test rows are exact copies of training rows "
                f"(compared on {len(columns)} columns, ignoring ids and the target). The model is scored "
                "partly on rows it memorized, so the test score overstates performance on new data.",
                fix="Drop duplicates before splitting (drop_duplicates on the feature columns), or split by "
                "a group key (GroupShuffleSplit) so copies of one record stay on one side.",
                metrics={
                    "shared_rows": len(rows),
                    "test_rows": len(te),
                    "rate": round(rate, 4),
                    "rows": rows_preview(rows),
                },
                tier=Tier.tier1,
            )
        ]
