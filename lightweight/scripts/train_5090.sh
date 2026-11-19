#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
mkdir -p "$ROOT/work_dirs"
# Hold one lock for the entire training process to prevent duplicate launches.
exec 9>"$ROOT/work_dirs/lightweight_5090.lock"
flock -n 9 || { echo 'This checkout already has a lightweight training job.' >&2; exit 1; }
ulimit -n 65536
export OMP_NUM_THREADS=4
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONPATH="$ROOT/mmpose${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT/mmpose"
RUN="$ROOT/work_dirs/lightweight_amp100_$(date +%Y%m%d_%H%M%S)"
exec python -u tools/train.py "$ROOT/lightweight/configs/small30_head256_amp100.py" --amp --work-dir "$RUN" "$@"
