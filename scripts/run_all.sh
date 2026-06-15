#!/usr/bin/env bash
# Chain every fit and then execute every notebook, in dependency order.
# The processed panel and split labels are committed, so this runs without
# any downloads. Artifacts land in artifacts/, figures in reports/figures/.
#
# Usage:
#   bash scripts/run_all.sh            # fits then notebooks
#   bash scripts/run_all.sh fits       # fits only
#   bash scripts/run_all.sh notebooks  # notebooks only
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-$HOME/miniforge3/envs/nem-demand-forecast/bin/python}"
JT="${JT:-$HOME/miniforge3/envs/nem-demand-forecast/bin/jupytext}"
STAGE="${1:-all}"

run_fits() {
  $PY scripts/fit_arima.py
  $PY scripts/fit_gbdt.py
  $PY scripts/fit_bart.py
  $PY scripts/fit_bsts_innovations.py
}

run_notebooks() {
  for nb in notebooks/01_*.py notebooks/02_*.py notebooks/03_*.py notebooks/04_*.py \
            notebooks/05_*.py notebooks/06_*.py notebooks/07_*.py; do
    echo "=== executing ${nb} ==="
    $JT --to notebook --execute "${nb}"
  done
}

case "$STAGE" in
  fits) run_fits ;;
  notebooks) run_notebooks ;;
  all) run_fits; run_notebooks ;;
  *) echo "unknown stage: $STAGE (expected fits|notebooks|all)"; exit 1 ;;
esac
echo "run_all: ${STAGE} complete"
