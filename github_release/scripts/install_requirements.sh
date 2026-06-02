#!/usr/bin/env bash
set -euo pipefail
cd ../backend
python3 -m venv ghost_env
source ghost_env/bin/activate
python -m pip install --upgrade pip
pip install -r ../../requirements.txt
echo "Requirements installed."
