# Real browser and five-minute scheduler verification

`python scripts/verify_browser_schedule.py` starts the actual scheduler with
`--interval 300 --max-runs 2`. There are no mock timers or substituted API calls.
The scheduler performs both complete DVC graphs, including rebuilding and
recreating the two Docker services, before exiting.

Chromium, controlled with Playwright, opens the real Streamlit page, fills all
four fields, presses **Predict**, and checks the rendered species, probability
chart and model run ID. It does this for setosa, versicolor and virginica both
before and after the second automatic deployment. Direct API calls provide a
reference; they do not replace the form submissions. The start-to-start interval
and a changed model ID are checked separately. The test then stops the real API
container, checks that the form displays an error without a stale prediction,
and starts the API again to check recovery.

## Run locally

Use a fresh checkout and the same Python 3.12 / Docker prerequisites as the README.
Do not run another scheduler or use the project's ports while this test runs.
The script uses the default localhost ports and rejects port/project overrides.

```bash
python -m pip install -r requirements-dev.txt -r requirements-browser.txt
python -m playwright install --with-deps chromium
python scripts/verify_browser_schedule.py
```

This is a real-time test; it does **not** accelerate the five-minute interval.
The local script leaves the containers running. Stop them with `python pipeline.py down`.
The GitHub Actions workflow stops its test containers in its final cleanup step.

## Evidence

The `pipeline-evidence` workflow artifact contains `reports/browser/`:

- `summary.json`: run IDs, real start times, assertions and prediction inputs.
- `first-*.png` and `second-*.png`: screenshots after each form submission.
- `first-trace.zip` and `second-trace.zip`: Playwright traces.
- `api-stopped-error.png` and `api-restored.png`: actual API outage and recovery.
- The scheduler log and copies of metadata/deployment receipts from both cycles.

Open a trace using `python -m playwright show-trace reports/browser/first-trace.zip`.
A failed test also captures the failing page when possible. Check the workflow's
actual conclusion and `summary.json`; this document describes the test, and is
not itself proof that it passed. The test does not change the data-cleaning
policy or certify the TA's interpretation of that policy.

Reference: https://playwright.dev/python/docs/ci
