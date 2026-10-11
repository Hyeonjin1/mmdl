#!/usr/bin/env bash
set -euo pipefail
cd /home/user/MMDL
# Export only existing terminal runs; never launches a model or dataset download.
.venv-p03/bin/python code/reporting/reports.py --run-id P04_20261011T034011Z_validate_E1_E2_cpu
.venv-p03/bin/python code/reporting/reports.py --run-id P04_20261011T033804Z_latest_vllm_B0_v2 --publish
