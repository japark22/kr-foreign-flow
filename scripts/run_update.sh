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

.venv/bin/python 05_daily_update.py --with-market "$@"

# Refresh the monitor page from the updated store, then publish it if it
# changed. The commit carries only the page -- data never leaves the machine.
.venv/bin/python 14_monitor.py

# The research page is rebuilt on the weekly run only: its inputs are the
# validation results, which change when a study is re-run, not every day.
if [[ "$*" == *--report* ]]; then
    .venv/bin/python 13_build_report.py || echo "  (research page build failed)"
fi

PAGES=(docs/monitor.html docs/index.html)
if [[ -n "$(git status --porcelain -- "${PAGES[@]}")" ]]; then
    git add "${PAGES[@]}"
    git commit -m "report: scheduled refresh $(date +%Y-%m-%d)"
    git push
fi
