#!/bin/sh
set -eu
PVC_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$PVC_ROOT"
PVC_PYTHON=${PVC_PYTHON:-python3.14}
if [ ! -x .venv/bin/python ]; then
  "$PVC_PYTHON" -m venv .venv
fi
.venv/bin/python -m pip install -r submission/requirements-final.lock
PYTHONPATH=src .venv/bin/python -m pvc_intelligence.final_pipeline run --raw "${1:-data/pvc_raw.xlsb}"
PYTHONPATH=src .venv/bin/python submission/tools/validate_submission.py
