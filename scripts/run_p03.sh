#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# The launcher chooses an eligible free RTX4090 and records its UUID.
# An existing CUDA_VISIBLE_DEVICES restriction is respected.
PY="$PWD/.venv-p03/bin/python"
CONFIG="$PWD/configs/P03_chartqa64.json"
PREFIX="P03_$(date -u +%Y%m%dT%H%M%SZ)"
# Every stage creates and checks mandatory records through the common launcher.
"$PY" code/p03/launcher.py --config "$CONFIG" --stage selftest --run-id "${PREFIX}_selftest"
"$PY" code/p03/launcher.py --config "$CONFIG" --stage validate --run-id "${PREFIX}_validate"
"$PY" code/p03/launcher.py --config "$CONFIG" --stage train --run-id "${PREFIX}_train_bf16"
"$PY" - "$CONFIG" "$PWD/runs/${PREFIX}_train_bf16" <<'PY'
import json,sys
from pathlib import Path
cfg=json.loads(Path(sys.argv[1]).read_text());run=Path(sys.argv[2])
metrics=json.loads((run/"metrics.json").read_text())
assert metrics["status"]=="completed" and metrics["optimizer_steps"]==20
cfg["adapter_path"]=metrics["adapter_path"]
cfg["purpose"]="P03: separate-process base/adapter generation checks on16 disjoint charts; no improvement claim"
(run/"evaluation_config.json").write_text(json.dumps(cfg,indent=2))
PY
# This launches a NEW Python process; training weights are already released.
"$PY" code/p03/launcher.py --config "$PWD/runs/${PREFIX}_train_bf16/evaluation_config.json" --stage evaluate --run-id "${PREFIX}_evaluate"
