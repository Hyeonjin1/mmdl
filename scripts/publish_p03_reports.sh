#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="$PWD/.venv-p03/bin/python"
# Export the successful P03 stages first, then publish the complete batch once.
for run in P03_20261008T075815Z_gpu_cleanup P03_20261008T075816Z_selftest P03_20261008T075816Z_validate P03_20261008T075816Z_train_bf16 P03_20261008T075816Z_evaluate P03_20261008T081920Z_completion_audit; do
    "$PY" code/reporting/reports.py --run-id "$run"
done
"$PY" code/reporting/reports.py --run-id P03_20261008T081920Z_completion_audit --publish
