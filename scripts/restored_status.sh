#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
"$task_root/.venv-restore-train/bin/python" "$task_root/code/restore/run.py" --status
if [[ -f "$task_root/.restore/model_progress.json" ]]; then
  cat "$task_root/.restore/model_progress.json"
fi
if [[ -f "$task_root/.restore/host_suite.log" ]]; then
  tail -n 12 "$task_root/.restore/host_suite.log"
fi
