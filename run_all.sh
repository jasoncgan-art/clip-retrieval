#!/bin/bash
# End-to-end reproduction: download -> embed -> retrieve -> evaluate.
# Exits on the first failure so a partial run never looks like a success.
# Usage: ./run_all.sh [--demo]   (--demo = labeled CIFAR-10 demo set)
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
# CPU torch first (small wheel, no CUDA deps); on a GPU box, replace
# this line with a CUDA-index torch install and leave the rest as-is.
pip install -q torch --index-url https://download.pytorch.org/whl/cpu
pip install -q -r requirements.txt

python3 run.py "$@"
echo "Done. See artifacts/metrics.json for measured Recall@K."
