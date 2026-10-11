#!/usr/bin/env bash
set -euo pipefail
cd /home/user/MMDL
CUDA_VISIBLE_DEVICES=0 .venv-p03/bin/python code/p04/host_steps.py host_gpu_probe
