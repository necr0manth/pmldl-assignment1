"""Regenerate the committed raw CSV from scikit-learn's offline Iris snapshot."""
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris

root = Path(__file__).resolve().parents[1]
data = load_iris()
frame = pd.DataFrame(data.data, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
frame.insert(0, "row_id", range(len(frame)))
frame["species"] = data.target_names[data.target]
path = root / "data/raw/iris.csv"
path.parent.mkdir(parents=True, exist_ok=True)
frame.to_csv(path, index=False, float_format="%.1f")
print(f"Wrote {len(frame)} rows to {path}")
