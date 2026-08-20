#!/bin/bash
# Wrapper for scheduled runs.
#
# launchd starts jobs with almost no environment — no PATH to your venv, and
# none of the variables your shell profile sets. So this script rebuilds the
# environment explicitly before calling Python. Anything that works here works
# under launchd; anything that relies on your interactive shell will not.
#
# Usage:  run_update.sh [extra args passed to 05_daily_update.py]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

if [[ ! -f .env ]]; then
    echo "ERROR: .env not found in $HERE" >&2
    exit 1
fi

# Load credentials. .env is gitignored and should be chmod 600.
set -a
# shellcheck disable=SC1091
source .env
set +a

if [[ ! -x .venv/bin/python ]]; then
    echo "ERROR: .venv not found in $HERE — create it with python3 -m venv .venv" >&2
    exit 1
fi

exec .venv/bin/python 05_daily_update.py "$@"
