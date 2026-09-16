# PMLDL Assignment 1 — automated deployment

An Iris classifier with **DVC → MLflow → Docker Compose**, a **FastAPI** model
API and a separate **Streamlit** application. The scheduler runs the complete
pipeline immediately and then every **300 seconds**. No cloud accounts or paid
services are required.

## Quick start: the complete assignment

Use Git, **Python 3.12**, Docker Engine/Desktop and Docker Compose **v2.20+**.
Docker must already be running and accessible to your user. The controller now
supports **native Windows Python (PowerShell)** as well as Linux/macOS/WSL2;
you do not need a WSL terminal. The API and app still use Linux Docker images.
On Windows, start Docker Desktop in **Linux containers** mode. Its internal
WSL2/Hyper-V backend is separate from where you run the Python controller.

### Windows / PowerShell

```powershell
git clone https://github.com/necr0manth/pmldl-assignment1.git
cd pmldl-assignment1
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip check
docker info
docker compose version
.\.venv\Scripts\python.exe pipeline.py schedule --interval 300
```

Calling the venv executable directly avoids PowerShell activation-policy issues.
Use a **Windows-created** venv, not one copied from WSL/Linux. For other commands
below, replace `python` with `.\.venv\Scripts\python.exe` unless your venv is
activated. To inspect MLflow from a second PowerShell terminal:

```powershell
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

The optional systemd installer and `make` shortcuts are not Windows requirements.
The foreground scheduler must stay running; Windows Task Scheduler installation
is not supplied. See [Windows verification boundaries](docs/WINDOWS.md).

### Linux / macOS / WSL2

Docker must be accessible without `sudo`.

```bash
git clone https://github.com/necr0manth/pmldl-assignment1.git
cd pmldl-assignment1
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip check
docker info
docker compose version
python pipeline.py schedule --interval 300
```

Leave the scheduler running. The first cycle cleans/splits the CSV, trains and
logs a model, builds both Docker images, starts both services, and checks a real
prediction. Every scheduled cycle repeats **all three stages** with a new model
run ID; it does not just restart containers or reuse DVC's training cache.

| Service | Address |
| --- | --- |
| Web app | http://localhost:8501 |
| Interactive API docs | http://localhost:8000/docs |
| API health and current model ID | http://localhost:8000/health |
| Model metadata and metrics | http://localhost:8000/model |

Enter four measurements and press **Predict**. The app obtains its prediction
from the separate API, displays the species, all three probabilities and the
model run ID. Expand **Current model and test metrics** to inspect the model.
The default input should predict **setosa**. Example inputs for versicolor are
`6.0, 2.9, 4.5, 1.5`; for virginica, `6.5, 3.0, 5.8, 2.2` (sepal length/width,
then petal length/width, all in cm).

## Commands

Run from the repository root with the virtual environment activated:

```bash
python pipeline.py run                      # Full pipeline once
python pipeline.py schedule                 # Full pipeline every 5 minutes
python pipeline.py schedule --max-runs 2     # Two actual scheduled attempts
python pipeline.py schedule --interval 600  # Longer interval for a slower host
python pipeline.py status                   # Latest attempt and log path
python -m pytest -q                         # Unit and integration tests
python pipeline.py run --train-only         # Development only; omits deployment
```

Ctrl+C stops the scheduler but leaves the last deployed containers running.
Stop the scheduler before removing the containers with `python pipeline.py down`.
The commands `make run`, `make schedule`, `make test`, `make status`, `make down`
and `make mlflow` are conveniences using `.venv/bin/python`.

## What each stage does

### 1. Data engineering — `code/pmldl/datasets.py`

The order follows the assignment literally: **load → clean → split → save**.
Read `data/raw/iris.csv`; validate columns and unique row IDs; convert invalid
measurements to missing values; remove rows with missing measurements or invalid
labels; remove duplicate observations; remove outliers; only then create a
seeded, stratified 80/20 train/test split and save the two CSVs plus a cleaning
report in `data/processed/`.

Missing rows are **removed**, not imputed; both are permitted by the assignment.
Outliers are defined by the explicit per-feature acceptance bounds in
`params.yaml`: sepal length 3–9, sepal width 1–5, petal length 0.5–8 and petal
width 0.05–3 cm. These conservative **demo policy limits** are not universal
botanical claims and are not estimated from the eventual test data. Thus the
pre-split cleanup does not fit medians, quantiles or a scaler on the whole file.
Change the bounds for a different data policy, not to optimize holdout accuracy.

The bundled Iris file has one duplicate, no missing measurements and no outliers
under these fixed limits: default output is **119 train rows and 30 test rows**.
The tests inject missing values and extreme outliers and check that neither
reaches the splitting function. The report records actual removals, including
zero counts; it does not invent outliers in an already clean dataset.
`row_id` is provenance, never a model feature.

### 2. Model engineering — `code/pmldl/training.py`

Read the saved partitions. Fit a scikit-learn pipeline of polynomial features
(degree 2, 14 features), standardization and logistic regression **on train only**.
Use the same fitted transformations at evaluation and inference.

Log parameters, accuracy, macro F1/precision/recall, log loss, confusion matrix,
cleaning report and a reloadable model to **MLflow**. SQLite tracking is in
`mlflow.db`; model artifacts are in `mlartifacts/`. Package the model with its
preprocessing in `models/model.joblib`; write `models/metadata.json` and
`reports/metrics.json`/`evaluation.json`. Metadata includes data/parameter/source
hashes and model dependency versions. The tiny fixed holdout is a demonstration,
not an independent real-world benchmark.

In another activated terminal, inspect experiment history with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open http://localhost:5000 and select `iris-classification`. The UI is optional:
training writes to SQLite without a running tracking server.

### 3. Deployment — `code/pmldl/deployment.py`

Verify the model SHA-256, build both images, then recreate services with Compose
and wait for health checks. The newly trained model is **copied into the API
image**, not read from a stale host mount. The app talks to `http://api:8000` on
the internal Docker network. Both containers are non-root with read-only root
filesystems and writable `/tmp` mounts.

Check API readiness, make a real prediction, check its run ID against the new
model and check Streamlit readiness. Save the receipt in
`reports/deployment.json`. A failed build leaves the prior containers running;
a failed recreation is **not automatically rolled back**. Brief downtime and
loss of an old Streamlit session's unsent inputs are possible. The browser test
checks that an already open tab can submit a new prediction after redeployment.
This is a local educational deployment, not a production zero-downtime service.

## Included trained model

`models/example/` contains a small trained **model.joblib**, matching metadata
and export provenance. It is a real model snapshot, not a placeholder. Keeping
it apart from `models/model.joblib` avoids Git-tracked files conflicting with
DVC's generated outputs. Scheduled training never overwrites the checked-in
example and never substitutes it for a new training run.

Verify it or deploy it without retraining (a convenience, **not** the complete
three-stage assignment demonstration):

```bash
python scripts/example_model.py verify
python scripts/example_model.py install
python -m pmldl.deployment
```

Verification checks hashes, schema, dependency versions and reproduced holdout
metrics before installation. If data, parameters or training sources change,
regenerate the snapshot instead of silently serving an incompatible artifact.
After a successful full run, export to a new directory for review:

```bash
python scripts/example_model.py export --directory reports/new-example
```

Only promote those matching files together to `models/example/`. Never load
untrusted pickle/joblib files. CI verifies and deploys the checked-in snapshot,
then separately runs the normal retraining pipeline.

## Automation, configuration and persistent operation

The scheduler executes `python -m dvc repro --force --no-run-cache deploy`.
DVC orders `prepare → train → deploy`. Starts use 300-second slots. When a cycle
exceeds that interval, missed slots are skipped rather than overlapped. Separate
native locks through `filelock` prevent multiple schedulers and concurrent pipeline attempts. Do not
remove lock files to bypass running processes, or run DVC manually alongside the
scheduler. The mere presence of a lock file is not evidence of a running process.
On Windows, cancellation force-stops the active DVC process tree with
`taskkill /T /F`; on POSIX it first uses SIGTERM. Output is read on a separate
thread so a silent child cannot indefinitely block Ctrl+C handling on Windows. Failures are recorded in `runs/<id>.log`, `runs/<id>.json` and
`runs/latest.json`; the next scheduled attempt retries the complete pipeline.

The generated data, current model, logs and tracking database are Git-ignored.
`dvc.lock` is intentionally **not** Git-ignored: DVC needs its stage signatures
Git-visible. No DVC remote is necessary; raw data, code, parameters and the
separate example model are committed. DVC is used here for orchestration.

### Linux systemd user service

With the project virtual environment activated, the installer fills the actual
checkout and interpreter paths automatically. It writes the unit; the explicit
`systemctl` command starts/enables it:

```bash
python scripts/install_service.py
systemctl --user daemon-reload
systemctl --user enable --now pmldl-pipeline.service
journalctl --user -u pmldl-pipeline.service -f
```

Use this **instead of**, not alongside, the foreground scheduler. To keep the
user manager running after logout and allow startup without an interactive
login, enable lingering (`sudo loginctl enable-linger "$USER"`) and ensure
Docker starts on the host. CI exercises a real user service, crash/restart and
stop, but does not reboot the host or certify your own login/boot setup.
Stop it with `systemctl --user stop pmldl-pipeline.service` before `pipeline.py down`.

### Ports and security

```bash
API_PORT=18000 APP_PORT=18501 COMPOSE_PROJECT_NAME=pmldl-iris-alt python pipeline.py run
```

Export the **same** variables when stopping the services. `BIND_ADDRESS` defaults
to `127.0.0.1`. Use environment variables, not a Compose-only `.env`, so the host
smoke-test client and Compose see the same settings. Do not run simultaneous
builds from multiple checkouts sharing image tags. The API requires all four
finite positive measurements up to 30 cm and rejects extra fields with HTTP 422.

No authentication or TLS is provided. Keep the app on localhost; for a remote
server prefer `ssh -L 8501:localhost:8501 user@server`. Direct Python dependencies
are pinned, but transitive dependencies and the Docker base tag are not a fully
immutable environment lockfile.

## Verification and submission

See [validation status and limitations](docs/VALIDATION.md) and
[browser test details](docs/BROWSER_TEST.md). CI installs a fresh Python 3.12 venv,
checks dependencies and the DVC DAG, runs all Linux tests without allowing skips,
verifies the included model, exercises actual scheduled Docker deployments in
Chromium and Firefox without replacing tabs, opens model metrics and MLflow UI,
stops/restores the real API, tests a real systemd user service and tests alternate
ports. Logs, screenshots, traces and a verified model candidate are in the
`pipeline-evidence` Actions artifact. Exact source inputs are archived separately.
A native Windows job runs the platform-independent suite (only Linux systemd
unit tests are excluded), real lock/process-cancellation tests and two actual
DVC/MLflow **training-only** runs. This does not certify a full Windows Docker
Desktop deployment; the full container/browser checks run on Linux.

A workflow definition is not proof of a passed run: inspect the Actions result
for your commit. **CI is neither permanent hosting nor your own demonstration.**
For the TA meeting, run the scheduler on the demonstration machine, make a web
prediction, show two Docker containers and MLflow experiments, and observe the
next automatic run and changed model ID. Submit the public repository link.

## Repository layout

```text
code/pmldl/              # Data, model, API, app, scheduler and service helpers
code/deployment/         # Separate Dockerfiles and Compose definition
data/raw/iris.csv       # Committed input, no runtime download
data/processed/          # Generated cleaned train/test files
models/example/          # Checked-in trained example and matching provenance
models/model.joblib      # Generated current model (DVC output, Git-ignored)
reports/                 # Generated metrics, receipts, browser and service evidence
scripts/                 # Browser/service checks and snapshot/install CLIs
services/systemd/        # User-service template, rendered automatically
tests/                   # Unit and integration tests
params.yaml              # Cleaning, split and model settings
dvc.yaml                 # Three-stage dependency graph
pipeline.py              # Entry point
```

Alternative logical layouts are allowed by the assignment. Airflow folders are
not needed because this project uses DVC.

## Data attribution and references

The CSV is exported from scikit-learn's bundled Iris dataset; see
[data provenance](data/raw/README.md). Source: [Fisher (1936), Iris, UCI](https://doi.org/10.24432/C56C76),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Row IDs and simplified
column/class names are added. Neither forbidden lab dataset is used.

References: [DVC repro](https://dvc.org/doc/command-reference/repro),
[MLflow](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html),
[Docker Compose](https://docs.docker.com/reference/cli/docker/compose/up/),
[Streamlit forms](https://docs.streamlit.io/develop/api-reference/execution-flow/st.form),
[Playwright](https://playwright.dev/python/docs/ci).
