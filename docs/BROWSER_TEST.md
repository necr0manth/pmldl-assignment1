# Real browser and service verification

The workflow runs `scripts/verify_browser_schedule.py` against real Docker
services. It starts the actual scheduler for two 300-second slots. It does not
accelerate time or substitute the application, model or HTTP calls.

Chromium and Firefox open the real Streamlit form, change four input fields,
click Predict, and check the displayed species, all three probability labels,
model run ID and expanded metrics panel. The SAME pages stay open through the
second training and recreation of both containers. There is deliberately no
`goto`, reload or new page for that second check: it tests reconnection of an
existing tab. The new run ID must be observed after a real form submission.

After both cycles the test stops the API, checks that both browser sessions
remove old results and show an error, starts the API, and confirms recovery.
It then starts an actual MLflow UI and opens a logged run, checking the run name
and visible metric keys. Screenshots and traces are retained even on failure.

## Commands

```bash
python -m pip install -r requirements-dev.txt -r requirements-browser.txt
python -m playwright install --with-deps chromium firefox
python scripts/verify_browser_schedule.py
```

Do not run another scheduler during the test. The local test leaves deployed
containers running; `python pipeline.py down` stops them. CI cleans up its own
containers. Use an existing deployment, including exported port variables, with:

```bash
python scripts/verify_browser_schedule.py --existing --evidence-dir reports/browser-existing
```

`scripts/verify_service.py` is a separate real systemd user-manager test intended
for a disposable Linux runner; it installs and removes a dedicated test unit.
It checks startup, automatic restart following SIGKILL and graceful stop, but
not host reboot or actual logout. Do not confuse it with the normal installer
`python scripts/install_service.py` used to run the project persistently.

## Reports

The Actions artifact `pipeline-evidence` contains JSON summaries, screenshots,
traces, a systemd journal, scheduler logs and matching trained-model files.
See [validation notes](VALIDATION.md) for exact paths and limitations. Open a
trace with `python -m playwright show-trace reports/browser/chromium-trace.zip`.

Earlier browser testing found Vega chart-update errors and stale bars; native
Streamlit progress bars and explicitly checked percentages replaced that plot.
A check that merely sees a chart container is not sufficient.

Reference: https://playwright.dev/python/docs/ci
