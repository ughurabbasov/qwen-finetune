#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r "${ROOT_DIR}/requirements.txt"

echo "Environment ready. Activate with: source .venv/bin/activate"
