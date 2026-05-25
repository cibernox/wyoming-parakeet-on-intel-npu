#!/bin/bash
# Container entrypoint:
#   1. Bootstrap model files into /data on first run.
#   2. Hand off to the wyoming server.
set -euo pipefail

DATA_DIR="${DATA_DIR:-/data}"

echo "[entrypoint] Starting wyoming-parakeet-on-intel-npu"
echo "[entrypoint] DATA_DIR=$DATA_DIR"

# 1. Make sure model files + static decoder IR exist
python3 -m scripts.bootstrap --data-dir "$DATA_DIR"

# 2. Run the wyoming server (foreground, replace shell so signals propagate)
exec python3 -m wyoming_parakeet_npu "$@"
