#!/usr/bin/env bash
set -euo pipefail
cd /home/user/MMDL
export CUDA_VISIBLE_DEVICES=0
exec .venv-p03/bin/python code/p04/suite.py "$@"
