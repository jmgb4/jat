#!/usr/bin/env bash
# Cross-platform setup wrapper for Linux / macOS.
# Usage: ./setup.sh [--cpu-only] [--skip-gpu-stack] [--skip-playwright]
#
# This delegates to scripts/jat.py so setup logic stays in one place.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if ! command -v python3 &>/dev/null; then
    echo "Python 3 is required but not found on PATH." >&2
    exit 1
fi

python3 scripts/jat.py setup "$@"
