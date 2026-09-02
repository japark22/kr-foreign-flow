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

# Rebuilt on the weekly run, and whenever the page is missing: a build that
# failed once must not leave the site without its research page.
if [[ "$*" == *--report* || ! -f docs/index.html ]]; then
    .venv/bin/python 13_build_report.py || echo "  (research page build failed)"
fi

# The event-study page is rebuilt only after the published results file has
# been reassembled, and 41_publish refuses to write that file when a result
# is older than the panel it came from. A stale figure therefore cannot reach
# the page: the build fails first and says so.
if [[ "$*" == *--report* || ! -f docs/event.html ]]; then
    if .venv/bin/python 41_publish.py; then
        .venv/bin/python 42_build_event_page.py \
            || echo "  (event page build failed)"
    else
        echo "  (results are stale -- event page left as it was)"
    fi
fi

# Publish whichever pages are actually on disk. Naming a missing file here
# once staged its deletion and removed the page from the site, then aborted
# every later run before the commit -- so absent means skip, never stage.
PAGES=()
for page in docs/monitor.html docs/index.html docs/event.html; do
    [[ -f "$page" ]] && PAGES+=("$page")
done

if [[ ${#PAGES[@]} -eq 0 ]]; then
    echo "  no pages on disk to publish"
elif [[ -n "$(git status --porcelain -- "${PAGES[@]}")" ]]; then
    git add -- "${PAGES[@]}"
    if git commit -m "report: scheduled refresh $(date +%Y-%m-%d)"; then
        git push || echo "  push failed -- page committed, not published"
    fi
else
    echo "  pages unchanged"
fi
