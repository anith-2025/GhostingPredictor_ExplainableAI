#!/usr/bin/env bash
# Quick installer for macOS / Linux (run from project root)
set -euo pipefail
python3 -m venv ghost_env
source ghost_env/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo
echo "Requirements installed. Activate with: source ghost_env/bin/activate"
