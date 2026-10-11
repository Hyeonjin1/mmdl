#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
exec "$task_root/.venv-restore-train/bin/python" -u "$task_root/code/restore/wait_and_run.py" >> "$task_root/.restore/host_suite.log" 2>&1
