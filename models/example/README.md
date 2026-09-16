# Included trained example

The 3.7 KB `model.joblib` was produced by the real MLflow training pipeline at
commit `63dd374501386f6a43e7e6810c7c5abb939856e4`, Actions run `35130202220`.
Its two scheduled Docker deployments and Chromium/Firefox browser checks passed;
that workflow subsequently exposed an unrelated systemd unit-quoting bug.
The unit is fixed in the current source, and current CI separately verifies and
deploys this exact committed model before testing new training runs.

`metadata.json` binds the binary to the raw CSV, parameters, training source,
partitions and library versions. `provenance.json` records export verification.
Run `python scripts/example_model.py verify` to recompute the held-out metrics.
The archived MLflow model URI identifies the original CI tracking store; it is
not a public model server. The self-contained joblib bundle does not require
that old tracking store. Start the normal full pipeline to create local MLflow
history and a new runtime model rather than using the archived URI.
