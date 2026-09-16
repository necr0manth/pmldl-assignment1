import numpy as np
import pandas as pd
import pytest

from pmldl.common import CLASSES, FEATURES
from pmldl.datasets import clean_partitions, prepare


def test_preparation_is_deterministic_and_disjoint(project):
    raw = project / "data/raw/iris.csv"
    output = project / "data/processed"
    report = prepare(raw, output)
    first = {path.name: path.read_bytes() for path in output.iterdir()}
    prepare(raw, output)
    assert first == {path.name: path.read_bytes() for path in output.iterdir()}
    train, test = pd.read_csv(output / "train.csv"), pd.read_csv(output / "test.csv")
    assert set(train.row_id).isdisjoint(test.row_id)
    assert set(train.species) == set(test.species) == set(CLASSES)
    assert not train.isna().any().any() and not test.isna().any().any()
    assert report["duplicates_removed"] == 1
    assert report["train_outliers_removed"] == 2
    assert len(train) == 117 and len(test) == 30
    assert not pd.concat([train, test]).duplicated(subset=[*FEATURES, "species"]).any()


def test_imputation_and_outliers_use_training_statistics_only():
    train = pd.DataFrame({feature: [1., 2., 3., 4., 5., 6., 7., 8., 9., 1000.] for feature in FEATURES})
    train["row_id"] = range(10)
    train.loc[0, "petal_width"] = np.nan
    test = pd.DataFrame({feature: [2000., np.nan] for feature in FEATURES})
    test["row_id"] = [10, 11]
    cleaned, held_out, report = clean_partitions(train, test, 1.5)
    assert report["removed_train_row_ids"] == [9]
    assert report["train_values_imputed"] == 1
    assert report["test_values_imputed"] == 4
    assert report["test_outliers_retained"] == 1
    assert held_out.iloc[0].sepal_length == 2000  # Do not cherry-pick test examples.
    assert cleaned.loc[0, "petal_width"] == 5.5
    assert held_out.loc[1, "sepal_length"] == 5
    changed_test = test.copy()
    changed_test[FEATURES] = 1000000.
    again, _, another_report = clean_partitions(train, changed_test, 1.5)
    pd.testing.assert_frame_equal(cleaned, again)
    assert report["training_medians"] == another_report["training_medians"]


@pytest.mark.parametrize("kind", ["missing_column", "duplicate_id", "no_classes"])
def test_invalid_raw_data_fails_clearly(project, kind):
    path = project / "data/raw/iris.csv"
    raw = pd.read_csv(path)
    if kind == "missing_column":
        raw = raw.drop(columns="sepal_width")
    elif kind == "duplicate_id":
        raw.loc[1, "row_id"] = raw.loc[0, "row_id"]
    else:
        raw["species"] = "unknown"
    raw.to_csv(path, index=False)
    with pytest.raises(ValueError):
        prepare(path, project / "data/processed")


def test_missing_values_in_input_are_imputed(project):
    path = project / "data/raw/iris.csv"
    raw = pd.read_csv(path)
    raw.loc[0, "sepal_length"] = np.nan
    raw.to_csv(path, index=False)
    report = prepare(path, project / "data/processed")
    assert report["train_values_imputed"] + report["test_values_imputed"] == 1


@pytest.mark.parametrize("multiplier", [0, -1, float("nan"), float("inf")])
def test_invalid_outlier_parameters(partitions, multiplier):
    with pytest.raises(ValueError):
        clean_partitions(*partitions, multiplier)
