#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0
# Explicitly authorized by the user: only current GPU0 Python compute PIDs.
# The launcher records owners, commands, parents, signals and CUDA probe results.
.venv-p03/bin/python code/p03/launcher.py --config configs/P03_gpu_cleanup_authorized.json --stage gpu_cleanup
bash scripts/run_p03.sh
