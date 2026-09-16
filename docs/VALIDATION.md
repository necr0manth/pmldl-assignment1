# Verification status

This document separates actual evidence from the checks implemented in CI. A
workflow definition alone is not a passed run. Check the Actions conclusion for
the exact commit and its `pipeline-evidence` artifact.

## Current verification coverage

The current workflow uses Linux and Python 3.12 in a fresh virtual environment.
It runs the complete Python suite and explicitly rejects skipped tests. It then:

1. Verifies the checked-in example's hashes and recomputes its holdout metrics;
   installs and deploys it, and checks predictions in Chromium and Firefox.
2. Starts the real scheduler for two full DVC/Docker cycles spaced 300 seconds
   apart. It checks three predictions and every displayed probability in each
   browser before and after the second cycle, retaining the SAME tabs throughout.
3. Opens the model-metrics panel, opens a logged run in the real MLflow UI,
   stops the API and checks stale-result removal and recovery.
4. Installs a real systemd user service, observes a successful pipeline, kills
   its scheduler, verifies automatic service restart and another full pipeline,
   then stops the service and checks that both process locks are released.
5. Runs another full pipeline with alternate ports and Compose project name,
   checks both browsers, and exports a verified model candidate.

The browser, deployment and systemd integration checks use real processes and
network connections, not mocked clocks, API requests or Docker operations.
Unit tests still intentionally mock isolated error paths.

## Where the evidence lives

- `reports/junit.xml`: exact test count, failures and skips.
- `reports/browser/summary.json`: run spacing, browser versions, same-tab checks,
  rendered probabilities, API recovery and MLflow UI results.
- `reports/browser-example/summary.json`: deployment of the committed snapshot.
- `reports/browser-alt/summary.json`: nondefault-port checks.
- `reports/systemd/summary.json` and `journal.log`: actual service lifecycle.
- `reports/example-verification.json`: validation of the committed snapshot.
- `reports/example-candidate/`: newly trained model, metadata and provenance.
- Browser folders also contain actual screenshots and Playwright traces.

Models and metrics are observations from executed code, not hard-coded claimed
results. The held-out dataset is small and fixed; this is an MLOps demonstration,
not an independent botanical benchmark.

## Explicit limitations

The CI environment is not the user's computer. Native Windows is unsupported;
WSL2 and macOS are not exercised by Linux CI. There is no host-reboot or actual
logout test, multi-day soak test, load test, public-internet/TLS test, or test of
every possible network/disk/Docker failure. Firefox and Chromium are covered;
Safari/WebKit is not. IPv6 URL formatting has unit coverage, not a real IPv6
container deployment. No automatic deployment rollback or zero downtime is
promised or required for this educational assignment.

## Historical results (not current-code certification)

The previous implementation at `0b031f547309dc7b7142766330cfba761fc985d2`
passed 45 Python tests with no skips and a real Chromium/Docker scheduler test
on 2026-09-16 (Actions run `35126242861`, start gap 300.00056 s). That version used
a different data-cleaning order and did not test same-tab reconnection or the
systemd service. Its results must not be used as proof for the revised cleaning
policy. The earliest authoring-sandbox result (42 tests plus missing optional
modules) has been superseded by real CI; it is not the current validation status.
