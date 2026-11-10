#!/usr/bin/env bash
# Rerun every random forest result from the two CSVs in the repository root.
set -euo pipefail
cd "$(dirname "$0")"
python3 src/hedonic_ols.py
python3 src/random_forest.py
python3 src/validation.py
