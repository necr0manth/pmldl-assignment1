"""Stage 1: load -> clean all rows -> split -> save, as required by the assignment."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pmldl.common import CLASSES, FEATURES, ROOT, TARGET, parameters, sha256, write_json

# Explicit, conservative demo acceptance limits, in centimetres. These are NOT
# fitted to the complete dataset (or the future test partition), nor claimed to
# be universal botanical limits. Edit data.outlier_bounds in params.yaml.
DEFAULT_BOUNDS = {
    "sepal_length": [3.0, 9.0],
    "sepal_width": [1.0, 5.0],
    "petal_length": [0.5, 8.0],
    "petal_width": [0.05, 3.0],
}


def validate_bounds(bounds: Mapping[str, Sequence[float]]) -> dict[str, list[float]]:
    if set(bounds) != set(FEATURES):
        raise ValueError("outlier_bounds must specify exactly the four input features")
    result = {}
    for feature in FEATURES:
        try:
            low, high = map(float, bounds[feature])
        except (ValueError, TypeError) as error:
            raise ValueError(f"Invalid outlier bounds for {feature}") from error
        if not np.isfinite([low, high]).all() or not 0 < low < high:
            raise ValueError(f"Outlier bounds for {feature} must satisfy 0 < low < high")
        result[feature] = [low, high]
    return result


def clean_data(raw: pd.DataFrame, outlier_bounds=None) -> tuple[pd.DataFrame, dict]:
    """Remove invalid/incomplete rows, duplicates and threshold-defined outliers.

    Dropping missing rows is explicitly allowed by the assignment. Fixed bounds
    make every cleaning decision independent of other rows and avoid estimating
    imputation/outlier statistics on data that will later be used for testing.
    """
    bounds = validate_bounds(DEFAULT_BOUNDS if outlier_bounds is None else outlier_bounds)
    required = ["row_id", *FEATURES, TARGET]
    missing = set(required) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    if raw.empty or raw["row_id"].isna().any() or raw["row_id"].duplicated().any():
        raise ValueError("Raw data must have nonempty, unique row_id values")
    data = raw[required].copy()
    data[FEATURES] = data[FEATURES].apply(pd.to_numeric, errors="coerce")
    data[FEATURES] = data[FEATURES].replace([np.inf, -np.inf], np.nan)
    data[FEATURES] = data[FEATURES].mask(data[FEATURES] <= 0)
    valid_label = data[TARGET].isin(CLASSES)
    incomplete = data[FEATURES].isna().any(axis=1)
    invalid = ~valid_label | incomplete
    report = {
        "raw_rows": len(raw), "invalid_rows_removed": int(invalid.sum()),
        "missing_rows_removed": int(incomplete.sum()),
        "invalid_labels_removed": int((~valid_label).sum()),
        "removed_invalid_row_ids": data.loc[invalid, "row_id"].tolist(),
    }
    data = data.loc[~invalid].copy()
    duplicate = data.duplicated(subset=[*FEATURES, TARGET])
    report["duplicates_removed"] = int(duplicate.sum())
    data = data.loc[~duplicate].copy()
    outlier = pd.Series(False, index=data.index)
    for feature, (low, high) in bounds.items():
        outlier |= ~data[feature].between(low, high, inclusive="both")
    report.update({
        "outliers_removed": int(outlier.sum()),
        "removed_outlier_row_ids": data.loc[outlier, "row_id"].tolist(),
        "outlier_bounds": bounds,
        "policy": "Drop missing/invalid rows; remove duplicates and fixed-bound outliers BEFORE splitting.",
    })
    data = data.loc[~outlier].copy()
    if set(data[TARGET]) != set(CLASSES):
        raise ValueError("Cleaned input must contain all three Iris classes")
    report["cleaned_rows"] = len(data)
    return data, report


def prepare(raw_path: Path, output: Path, *, test_size=0.2,
            random_state=42, outlier_bounds=None) -> dict:
    # No split or learned preprocessing occurs before this entire cleaning step.
    cleaned, report = clean_data(pd.read_csv(raw_path), outlier_bounds)
    train, test = train_test_split(cleaned, test_size=test_size,
                                   random_state=random_state, stratify=cleaned[TARGET])
    output.mkdir(parents=True, exist_ok=True)
    train.sort_values("row_id").to_csv(output / "train.csv", index=False)
    test.sort_values("row_id").to_csv(output / "test.csv", index=False)
    report.update(train_rows=len(train), test_rows=len(test), random_state=random_state,
                  raw_sha256=sha256(raw_path))
    write_json(output / "cleaning_report.json", report)
    return report


def main() -> None:
    print(json.dumps(prepare(ROOT / "data/raw/iris.csv", ROOT / "data/processed",
                             **parameters()["data"]), indent=2))


if __name__ == "__main__":
    main()
