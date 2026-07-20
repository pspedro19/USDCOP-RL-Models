#!/bin/bash
# Convenience wrapper to run algotrader with virtual environment

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python"

# Check if venv exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo "⚠️  Virtual environment not found. Creating..."
    python3 -m venv "$SCRIPT_DIR/venv"
    "$SCRIPT_DIR/venv/bin/pip" install -r "$SCRIPT_DIR/requirements.txt"
fi

# Run algotrader with all arguments
"$VENV_PYTHON" "$SCRIPT_DIR/algotrader.py" "$@"
