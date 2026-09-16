# PMLDL Assignment 1 — automated deployment

An end-to-end Iris classifier with **DVC → MLflow → Docker Compose**, a **FastAPI**
model API and a separate **Streamlit** web application. A local scheduler runs
**all three stages immediately and every 300 seconds** while it is running.

## Quick start

Use Linux, macOS, or **WSL2** on Windows. Prerequisites: Git, Python **3.12**,
Docker Engine/Desktop, and Docker Compose **v2.20+**. Docker must be running and
accessible to your user without `sudo`. Internet access is needed for dependency
installation and initial container builds; the dataset is already in the repository.
Native Windows is not supported by the scheduler's POSIX file locks.

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

The first run prepares the data, trains and evaluates a new model, builds **two
images**, starts **two containers**, waits for both health checks and performs a
real prediction smoke test. Leave the scheduler process running for subsequent
runs; cloning the repository alone does not start any services.

| Service | Address |
| --- | --- |
| Web application | http://localhost:8501 |
| Interactive API documentation | http://localhost:8000/docs |
| API health and loaded model ID | http://localhost:8000/health |
| Model metadata and metrics | http://localhost:8000/model |

In the application, edit the four measurements and press **Predict**. The app
calls the API over HTTP, then displays the predicted species and probabilities.
The default measurements should predict **setosa**. The app does not load or run
the model itself.

## Commands

Run these from the repository root, with the virtual environment activated:

```bash
python pipeline.py run                           # All three stages, once
python pipeline.py schedule                      # All stages every 5 minutes
python pipeline.py schedule --interval 600        # Longer interval, if necessary
python pipeline.py schedule --max-runs 2          # Two complete scheduled attempts
python pipeline.py status                        # Latest result and log path
python -m pytest -q                              # Unit and integration tests
python pipeline.py run --train-only              # Development only: no deployment
```

Press **Ctrl+C** to stop the foreground scheduler. The already deployed containers
remain running. Stop the scheduler first, then remove the containers with:

```bash
python pipeline.py down
```

`make run`, `make schedule`, `make test`, `make status`, `make down`, and
`make mlflow` are equivalent conveniences using `.venv/bin/python`.
**Training-only mode is not the complete assignment demonstration.**

## What each stage does

### 1. Data engineering (`code/pmldl/datasets.py`)

Read `data/raw/iris.csv` on every run. Validate the schema and row IDs, convert
invalid/nonpositive/nonfinite measurements to missing values, remove invalid
labels and entirely empty observations, and remove duplicate samples.
Produce a seeded, stratified 80/20 split. Learn 1.5×IQR bounds on the training
partition, remove its outliers, and impute missing measurements with retained
training medians. Save `data/processed/train.csv`, `test.csv`, and
`cleaning_report.json`.

**Leakage prevention:** statistical cleaning is fitted after a provisional split,
not on the whole dataset. Valid test outliers are retained rather than discarded
to make the held-out evaluation easier. This is a deliberate implementation
choice. The pristine Iris data has no missing values; tests inject them to verify
imputation. With the checked-in CSV and default parameters, one duplicate and two
training outliers are removed, leaving **117 training rows and 30 test rows**.
`row_id` is provenance only and is never a model feature.

### 2. Model engineering (`code/pmldl/training.py`)

Read the two saved partitions. Build a scikit-learn `Pipeline` with
`PolynomialFeatures(degree=2, include_bias=False)` (14 features), `StandardScaler`,
and `LogisticRegression`. Fit transformations and the classifier only on training
data; apply the same fitted pipeline to test data and inference inputs.

Log parameters, **accuracy, macro F1, macro precision, macro recall and log loss**,
the confusion matrix, cleaning report, and a reloadable model to **MLflow**.
Tracking metadata is stored in `mlflow.db` (SQLite); artifacts are in `mlartifacts/`.
Package preprocessing and the classifier together in `models/model.joblib` and
save run/version/hash information in `models/metadata.json`. Evaluation files are
in `reports/metrics.json` and `reports/evaluation.json`.

A local run of the default model produced **accuracy 0.9667**, **macro F1 0.9666**,
and **log loss 0.1140** on the 30 held-out observations. These are a small demo
holdout, not a claim about real-world botanical accuracy; repeating the fixed
split is not independent validation. See [validation notes](docs/VALIDATION.md).

Open the experiment history in a second terminal:

```bash
source .venv/bin/activate
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Then visit http://localhost:5000 and select `iris-classification`. The MLflow UI
is optional; training logs directly to SQLite and does not need a running server.

### 3. Deployment (`code/pmldl/deployment.py`)

Check the packaged model's SHA-256 against its metadata. Build the API and app
images using the Dockerfiles in `code/deployment/api` and `code/deployment/app`.
Run `docker compose up --detach --force-recreate --wait` against
`code/deployment/docker-compose.yml`. The freshly trained model is **copied into
the API image**, not read from an old host-mounted artifact.

The containers communicate over Compose's internal network (`http://api:8000`).
Both run as non-root, with read-only root filesystems and writable `/tmp` mounts.
Health checks gate readiness. The deployment stage checks `/health`, sends a
real `/predict` request, verifies that the returned model run ID matches the new
training run, and checks Streamlit health. Its receipt is `reports/deployment.json`.

## Automation and failure behavior

The scheduler calls:

```bash
python -m dvc repro --force --no-run-cache deploy
```

The dependency graph in `dvc.yaml` orders `prepare → train → deploy`. Both force
flags are intentional: a scheduled run must redo all stages even when the raw
CSV is unchanged. Each training attempt creates a distinct MLflow run; identical
predictions on fixed data are expected, but the served **run ID changes**.

The initial run starts immediately. Later starts use 300-second slots. If a run
exceeds the interval, missed slots are skipped; runs never overlap and do not
accumulate a catch-up queue. Separate OS locks prevent two scheduler instances
and simultaneous pipeline runs. Lock files remain on disk, but the OS releases
the locks when the process exits; do not delete them to bypass an active lock.
Use `pipeline.py` rather than running DVC manually alongside the scheduler.

Failures stop downstream stages and are recorded in `runs/<id>.log`,
`runs/<id>.json`, and `runs/latest.json`; the scheduler tries again at the next
slot. A failed image build leaves existing containers running. **A failed
container recreation is not automatically rolled back**, and deployment can
cause brief downtime. This is a local educational deployment, not a production
blue/green rollout.

The model artifacts, SQLite database, run logs, and `dvc.lock` are generated
locally and ignored by Git. No DVC remote is needed because raw data, parameters
and source code are committed. Scheduled run digests are kept locally; this
project uses DVC for orchestration, not as a remote model registry.

### Persistent scheduling on Linux

Edit the two `/ABSOLUTE/PATH/pmldl-assignment1` placeholders in
`services/systemd/pmldl-pipeline.service` to your checkout's absolute path. Then:

```bash
mkdir -p ~/.config/systemd/user
cp services/systemd/pmldl-pipeline.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now pmldl-pipeline.service
journalctl --user -u pmldl-pipeline.service -f
```

This is an alternative to the foreground scheduler, not an additional scheduler.
To keep a user service running after logout, an administrator may need to enable
user lingering (`loginctl enable-linger USER`). The machine and Docker daemon
must remain running. Stop with `systemctl --user stop pmldl-pipeline.service`
before `python pipeline.py down`.

## API example and configuration

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

The response contains `species`, `probabilities` (one probability per class),
and `model_run_id`. All four input fields are required finite numbers in
centimetres, greater than 0 and at most 30; extra fields are rejected with HTTP
422. These API bounds are input validation, not the model's training support.

Change data/model settings in `params.yaml`. To change ports, export `API_PORT`
and `APP_PORT` **before starting the scheduler**, for example:

```bash
API_PORT=8001 APP_PORT=8502 python pipeline.py schedule
```

`BIND_ADDRESS` defaults to `127.0.0.1`, and `COMPOSE_PROJECT_NAME` defaults to
`pmldl-iris`. Use the same environment when stopping services. Avoid overriding
these through a Compose-only `.env` file: the host's smoke-test client also needs
the values. Do not run several checkouts using the same ports/image tags.

The demo has no authentication or TLS. Keep it on localhost; for a remote server,
prefer SSH forwarding, e.g. `ssh -L 8501:localhost:8501 user@server`. Treat
`model.joblib` as trusted executable data: never replace it with an untrusted
pickle/joblib file. Use the pinned model requirements in both training and API
images when changing dependencies. Direct dependencies are pinned, but the
Docker base tag and transitive dependencies are not a full immutable lockfile.

## Repository layout

```text
code/
  pmldl/                 # Data, model, API, UI, deployment and scheduler modules
  deployment/
    api/Dockerfile
    app/Dockerfile
    docker-compose.yml
data/
  raw/iris.csv           # Committed input; no runtime data download
  processed/             # Generated train/test CSVs and cleaning report
models/                  # Generated model bundle and metadata
reports/                 # Generated evaluation and deployment evidence
services/systemd/        # Optional persistent five-minute scheduler
scripts/export_iris.py   # Recreate the raw CSV from scikit-learn
requirements/            # Shared model and separate API/UI requirements
tests/                   # Unit tests, MLflow and Streamlit integration tests
.github/workflows/ci.yml # Two full Docker deployments on push/PR
params.yaml
dvc.yaml
pipeline.py
```

A single importable `pmldl` package avoids path/import workarounds across stages.
Airflow and a `services/airflow` directory are unnecessary because DVC is used.

## Tests and demonstration

The GitHub Actions workflow installs the complete requirements, runs pytest,
executes the full pipeline twice on a Docker-capable runner, checks that the API
serves the second model, and uploads logs/reports as `pipeline-evidence`.
**CI is verification, not the five-minute scheduler or persistent hosting.**
Check its actual status in the repository's Actions tab; having a workflow file
alone is not proof that the Docker build passed.

For the TA demonstration, start the scheduler, open the app, make a prediction,
show the current run ID and metrics, and observe another automatic run after five
minutes (or a longer configured interval). Show that the UI uses the new model ID,
that two containers are running, and that MLflow contains distinct runs.

Troubleshooting: `docker info` must work; a Compose `--wait` error means Compose
needs updating; port conflicts can be resolved with `API_PORT`/`APP_PORT`; a
missing model means training has not succeeded yet. Inspect `runs/latest.json`
and the referenced log first. For container logs:

```bash
docker compose -p pmldl-iris -f code/deployment/docker-compose.yml logs --tail 100
```

## Data attribution and references

The CSV is exported from scikit-learn's bundled Iris dataset. Source:
[Fisher (1936), Iris, UCI Machine Learning Repository](https://doi.org/10.24432/C56C76),
licensed by UCI under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The export adds row IDs, simplifies feature names, and renders class names as
strings; see [data provenance](data/raw/README.md). No CelebFaces or smoking-status
data is used.

Implementation references: [DVC repro](https://dvc.org/doc/command-reference/repro),
[MLflow scikit-learn integration](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html),
[Docker Compose up](https://docs.docker.com/reference/cli/docker/compose/up/),
and [Streamlit forms](https://docs.streamlit.io/develop/api-reference/execution-flow/st.form).
