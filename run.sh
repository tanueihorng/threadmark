#!/bin/zsh
set -e

cd "${0:A:h}"
export HF_HOME="$PWD/.cache/huggingface"
export PYANNOTE_METRICS_ENABLED=0
export DYLD_LIBRARY_PATH="/opt/homebrew/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"

exec "$PWD/.venv/bin/python" -m uvicorn app:app --host 127.0.0.1 --port 8765
