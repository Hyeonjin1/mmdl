#!/usr/bin/env bash
set -euo pipefail
cd /home/user/MMDL
.venv-p03/bin/python code/p04/launch.py --config configs/P04/E12_latest_vllm.json --stage setup_eval
.venv-p04-vllm/bin/python code/p04/launch.py --config configs/P04/E12_latest_vllm.json --stage envcheck
