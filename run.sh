#!/usr/bin/env bash
# Run Job Application Tailor on Linux / macOS.
# Usage: ./run.sh [--host HOST] [--port PORT] [--no-reload] [--skip-gpu-check]
#
# Set JAT_NO_RELOAD=1 in the environment to disable hot-reload, e.g.:
#   JAT_NO_RELOAD=1 ./run.sh
#
# This delegates to scripts/jat.py so runtime flow is shared across platforms.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if ! command -v python3 &>/dev/null; then
    echo "Python 3 is required but not found on PATH." >&2
    exit 1
fi

python3 scripts/jat.py run "$@"
