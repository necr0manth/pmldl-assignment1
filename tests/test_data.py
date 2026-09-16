import numpy as np
import pandas as pd
import pytest

from pmldl import datasets
from pmldl.common import CLASSES, FEATURES
from pmldl.datasets import DEFAULT_BOUNDS, clean_data, prepare, validate_bounds


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
    # The original Iris file has no incomplete rows or fixed-bound outliers.
    # Tests below insert both; never fabricate removals to make a report look busy.
    assert report["outliers_removed"] == report["missing_rows_removed"] == 0
    assert len(train) == 119 and len(test) == 30
    assert not pd.concat([train, test]).duplicated(subset=[*FEATURES, "species"]).any()


def test_all_cleaning_happens_before_split(project, monkeypatch):
    path = project / "data/raw/iris.csv"
    raw = pd.read_csv(path)
    raw.loc[0, "petal_width"] = np.nan
    raw.loc[1, "sepal_length"] = 1000
    bad_ids = set(raw.loc[[0, 1], "row_id"])
    raw.to_csv(path, index=False)
    original_split = datasets.train_test_split
    def split(cleaned, **kwargs):
        assert not cleaned[FEATURES].isna().any().any()
        assert not bad_ids.intersection(cleaned.row_id)
        for feature, (low, high) in DEFAULT_BOUNDS.items():
            assert cleaned[feature].between(low, high).all()
        return original_split(cleaned, **kwargs)
    monkeypatch.setattr(datasets, "train_test_split", split)
    report = prepare(path, project / "data/processed")
    assert report["missing_rows_removed"] == report["outliers_removed"] == 1


def test_cleaning_thresholds_do_not_depend_on_other_rows(project):
    raw = pd.read_csv(project / "data/raw/iris.csv")
    first, _ = clean_data(raw)
    extra = raw.iloc[:3].copy()
    extra["row_id"] = [1001, 1002, 1003]
    extra["sepal_length"] = 1000000
    second, report = clean_data(pd.concat([raw, extra], ignore_index=True))
    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))
    assert report["outliers_removed"] == 3


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 0, -1, "not a number"])
def test_incomplete_invalid_measurements_are_removed(project, bad):
    raw = pd.read_csv(project / "data/raw/iris.csv")
    raw["sepal_length"] = raw["sepal_length"].astype(object)
    raw.loc[0, "sepal_length"] = bad
    cleaned, report = clean_data(raw)
    assert raw.loc[0, "row_id"] not in set(cleaned.row_id)
    assert report["missing_rows_removed"] == 1


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


@pytest.mark.parametrize("bad", [[0, 3], [5, 3], [float("nan"), 3], [1, float("inf")], [1]])
def test_invalid_outlier_bounds(bad):
    with pytest.raises(ValueError):
        validate_bounds(dict(DEFAULT_BOUNDS, sepal_length=bad))


def test_outlier_bounds_require_all_features():
    with pytest.raises(ValueError):
        validate_bounds({"sepal_length": [3, 9]})
