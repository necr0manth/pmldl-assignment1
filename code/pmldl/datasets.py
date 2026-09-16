"""Stage 1: read a raw CSV, clean it, and persist stratified data partitions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pmldl.common import CLASSES, FEATURES, ROOT, TARGET, parameters, sha256, write_json


def clean_partitions(train: pd.DataFrame, test: pd.DataFrame, multiplier: float):
    """Learn outlier bounds and imputation from training data only.

    Remove training outliers, but retain valid test outliers for honest evaluation.
    Missing test measurements are imputed with the training medians.
    """
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("iqr_multiplier must be positive and finite")
    q1, q3 = train[FEATURES].quantile(0.25), train[FEATURES].quantile(0.75)
    low, high = q1 - multiplier * (q3 - q1), q3 + multiplier * (q3 - q1)
    if low.isna().any() or high.isna().any():
        raise ValueError("A training feature has no observed values")
    outliers = ((train[FEATURES] < low) | (train[FEATURES] > high)).any(axis=1)
    test_outliers = ((test[FEATURES] < low) | (test[FEATURES] > high)).any(axis=1)
    retained = train.loc[~outliers].copy()
    medians = retained[FEATURES].median()
    if retained.empty or medians.isna().any():
        raise ValueError("Insufficient training data after outlier removal")
    report = {
        "train_outliers_removed": int(outliers.sum()),
        "removed_train_row_ids": train.loc[outliers, "row_id"].tolist(),
        "test_outliers_retained": int(test_outliers.sum()),
        "train_values_imputed": int(retained[FEATURES].isna().sum().sum()),
        "test_values_imputed": int(test[FEATURES].isna().sum().sum()),
        "training_medians": medians.to_dict(),
        "training_iqr_lower": low.to_dict(), "training_iqr_upper": high.to_dict(),
    }
    retained[FEATURES] = retained[FEATURES].fillna(medians)
    test = test.copy()
    test[FEATURES] = test[FEATURES].fillna(medians)
    return retained, test, report


def prepare(raw_path: Path, output: Path, *, test_size=0.2,
            random_state=42, iqr_multiplier=1.5) -> dict:
    raw = pd.read_csv(raw_path)
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
    valid = data[TARGET].isin(CLASSES) & data[FEATURES].notna().any(axis=1)
    data = data.loc[valid]
    before_dedup = len(data)
    data = data.drop_duplicates(subset=[*FEATURES, TARGET])
    if set(data[TARGET]) != set(CLASSES):
        raise ValueError("The input must contain all three Iris classes")
    train, test = train_test_split(
        data, test_size=test_size, random_state=random_state, stratify=data[TARGET]
    )
    train, test, report = clean_partitions(train, test, iqr_multiplier)
    if set(train[TARGET]) != set(CLASSES):
        raise ValueError("Cleaning removed all training examples of a class")
    output.mkdir(parents=True, exist_ok=True)
    train.sort_values("row_id").to_csv(output / "train.csv", index=False)
    test.sort_values("row_id").to_csv(output / "test.csv", index=False)
    report.update({
        "raw_rows": len(raw), "invalid_rows_removed": int((~valid).sum()),
        "duplicates_removed": before_dedup - len(data),
        "train_rows": len(train), "test_rows": len(test),
        "random_state": random_state, "raw_sha256": sha256(raw_path),
        "policy": "Fit cleaning on train only; keep test outliers for honest evaluation.",
    })
    write_json(output / "cleaning_report.json", report)
    return report


def main() -> None:
    print(json.dumps(prepare(ROOT / "data/raw/iris.csv", ROOT / "data/processed",
                             **parameters()["data"]), indent=2))


if __name__ == "__main__":
    main()
