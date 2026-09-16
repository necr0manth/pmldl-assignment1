# Verification status

## Current portability revision

The controller no longer imports `fcntl` unconditionally. It uses native
cross-platform `filelock` locks, Windows process-tree cancellation, interruptible
output streaming and UTF-8 logs. Text snapshot verification accepts LF/CRLF-only
differences without weakening binary model integrity checks. The data/model
training code, parameters and committed model have not changed in this revision.

See [Windows support and exact verification scope](WINDOWS.md). The workflow now
runs a native Windows job in addition to the full Linux Docker/browser job.
Check the Actions result for the **exact current commit** and download the
artifacts. A workflow definition is not evidence of a successful run.

- Linux: all tests, committed-model deployment, two real five-minute DVC/Docker
  cycles, same Chromium/Firefox tabs across redeployment, numeric probabilities,
  model IDs, expanded metrics, real API outage/recovery, MLflow UI, real systemd
  lifecycle/crash restart and nondefault-port deployment.
- Windows: all applicable Python tests (Linux systemd unit tests excluded),
  real locks/process-tree cancellation, exact snapshot checks and two real
  DVC/MLflow training runs with model reload and FastAPI predictions.
- `pipeline-evidence`: Linux reports, screenshots, traces, model and journal.
- `windows-evidence`: native test report and `reports/native/summary.json`.

The Windows job does not run Docker Desktop. It must not be cited as proof of
an end-to-end Windows Docker deployment. The browser/deployment suite runs with
real Linux containers in the Linux job.

## Historical verified implementation — 2026-09-16

Commit `101e72d33fd18be57100ae4edbea9f62de6a7712` passed
[Actions run 35131761245](https://github.com/necr0manth/pmldl-assignment1/actions/runs/35131761245)
at 18:11:21 UTC: **70 tests, zero failures/errors/skips**, exact included model
verification/deployment, same-tab Chromium/Firefox checks, MLflow UI, real
systemd crash recovery and nondefault-port deployment. Its measured interval
was **300.000295 seconds**. See [historical machine-readable evidence](verification.json).
The artifact was `pipeline-evidence`, ID `10461349743`, SHA-256
`904f1db1d34dc8be3ab8aaf425469f028d335f010148227aef3d752aa33e53fe`.
Those historical results do not certify the later scheduler changes or Windows.

An earlier version (`0b031f5`) passed 45 tests and a real Chromium/Docker test,
but used a different cleaning order. The earliest local-sandbox-only report
(42 tests plus unavailable optional modules) is not current CI evidence.

## Assignment alignment and model

Data processing is **load -> clean -> split -> save**. Missing rows, duplicates
and fixed-bound outliers are removed before splitting, without fitting statistics
on future test rows. Default data yields 119 train and 30 test rows. Tests inject
missing values/outliers and check they do not reach the splitting function.
Feature engineering and model fitting still use train only. The tiny fixed
holdout is a demonstration, not an independent botanical benchmark.

`models/example/` contains the actual 3714-byte trained snapshot with matching
metadata/provenance, separate from DVC's runtime outputs. Verification recomputes
its metrics and checks model bytes, source/input hashes and dependency versions.
Normal scheduling always trains new models rather than serving the example.

## Explicit limitations

CI is not the user's computer. There is no Windows Docker Desktop, host reboot,
actual logout, multi-day soak, load, public-internet/TLS or exhaustive disk/network
failure test. macOS, WSL2 as a host and Safari/WebKit are not exercised. IPv6 URL
formatting has unit coverage but no live IPv6 Docker deployment. No automatic
rollback, zero downtime or Windows Task Scheduler installation is supplied.
