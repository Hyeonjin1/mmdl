#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
exec "$task_root/.venv-restore-train/bin/python" "$task_root/code/restore/run.py" --suite
