# Iris data provenance

`iris.csv` contains the 150 rows of scikit-learn's bundled `load_iris()` dataset
exported using scikit-learn 1.8.0 by `scripts/export_iris.py`. It has four numeric
measurement columns (centimetres), a species label, and an added zero-based
`row_id`. There are three classes: setosa, versicolor and virginica.

Source attribution: Fisher, R. (1936). *Iris* [Dataset]. UCI Machine Learning
Repository. https://doi.org/10.24432/C56C76

UCI dataset page: https://archive.ics.uci.edu/dataset/53/iris

UCI license: Creative Commons Attribution 4.0 International (CC BY 4.0):
https://creativecommons.org/licenses/by/4.0/

scikit-learn's loader documents corrections to two observations compared with
some copies of the UCI file. This project uses that bundled snapshot unchanged,
except for CSV formatting, column names, row IDs, and string target labels:
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html

Recreate the file after installing the requirements:

```bash
python scripts/export_iris.py
```

The pipeline always reads the committed CSV, not the loader. It does not silently
redownload data. Missing values are absent in the original snapshot; tests inject
missing entries and outliers to exercise the cleaning routines.
