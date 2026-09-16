# Implementation validation

Local validation on 2026-09-16, Python 3.13.5:

- `python -m pytest -q --junitxml=reports/local-junit.xml`: **42 passed**,
  **2 integration modules skipped** because MLflow and Streamlit are unavailable
  in the authoring sandbox. These skips represent three integration tests.
- Data preparation, feature engineering, actual classifier training/evaluation,
  joblib round-trip, and FastAPI in-process requests were executed successfully.
- Scheduler/deployment unit tests verify process exit codes, locking, scheduling,
  image-build ordering, stale-model detection, and error paths. Docker operations
  in these unit tests are mocked, not real container deployments.
- MLflow logging, the real DVC CLI, Streamlit execution and real Docker deployment
  were **not verified locally**: those packages/Docker are unavailable here.
  The GitHub Actions workflow is provided to perform the complete verification
  in an environment that can install dependencies and run Docker.

## Observed model result

117 training rows, 30 test rows, 14 engineered features.

```json
{
  "accuracy": 0.9666666666666667,
  "f1_macro": 0.9665831244778612,
  "precision_macro": 0.9696969696969697,
  "recall_macro": 0.9666666666666667,
  "log_loss": 0.11402936671610965
}
```

These are observations from the actual local model run, not fabricated CI results.
The holdout is very small and fixed; this is an MLOps demonstration, not an
independent benchmark or a claim of production accuracy.

CI status can change; consult the run for the exact commit in the Actions tab.
A workflow definition alone is not evidence of a successful deployment.
