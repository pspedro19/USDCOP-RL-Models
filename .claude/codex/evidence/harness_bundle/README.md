---
kind: audit
status: HISTORICAL
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors: []
---
# Harness evidence bundle

Generated: 2026-07-21T00:04:48.649577+00:00

Evidence is local/reproducible; no provider credentials or secrets are captured.

## Runs

- `market_statistics`: exit `0` — `C:\Users\pedro\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\python.exe scripts/analysis/audit_market_data_statistics.py`
- `acquisition_assets`: exit `0` — `C:\Users\pedro\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\python.exe scripts/analysis/audit_acquisition_assets.py`
- `reconciliation`: exit `0` — `C:\Users\pedro\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\python.exe scripts/analysis/reconcile_seed_backups.py`
- `manifest_validation`: exit `0` — `C:\Users\pedro\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\python.exe scripts/validation/validate_acquisition_manifests.py`
- `harness`: exit `1` — `C:\Users\pedro\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\python.exe .claude/codex/harness/harness_engine.py --no-tests`

## Visual evidence

- Screenshot/video capture is **PENDING** until a browser session and authenticated sandbox are available.
- The bundle contains command logs, JSON evidence and SHA-256 hashes; it does not claim a visual or live-provider run.
