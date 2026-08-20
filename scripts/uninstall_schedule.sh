#!/bin/bash
# Remove the scheduled jobs. Data and reports are left alone.
set -euo pipefail

AGENTS="$HOME/Library/LaunchAgents"

for label in com.krxflow.collect com.krxflow.report; do
    plist="$AGENTS/$label.plist"
    if [[ -f "$plist" ]]; then
        launchctl unload "$plist" 2>/dev/null || true
        rm -f "$plist"
        echo "  removed $label"
    else
        echo "  $label was not installed"
    fi
done
