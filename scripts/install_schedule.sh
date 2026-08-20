#!/bin/bash
# Install the two scheduled jobs on macOS via launchd.
#
#   collect  — every day at 19:00 local. Fetches whatever trading days are
#              missing. On a day with nothing new it makes one request and
#              exits, so running it daily costs nothing and means a single
#              missed run never turns into a gap.
#
#   report   — Mondays at 08:00 local. Regenerates reports/validation.md and
#              pushes it. Weekly, because the statistics move on the scale of
#              weeks; a daily commit would be churn in the history for no
#              added information.
#
# Both are safe to run by hand at any time.
#
#   bash scripts/install_schedule.sh            # collection only
#   bash scripts/install_schedule.sh --push     # collection + weekly push
#
# Undo with scripts/uninstall_schedule.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AGENTS="$HOME/Library/LaunchAgents"
RUNNER="$HERE/scripts/run_update.sh"

WANT_PUSH=0
[[ "${1:-}" == "--push" ]] && WANT_PUSH=1

chmod +x "$RUNNER"
mkdir -p "$AGENTS" "$HERE/data/logs"

write_plist () {
    local label="$1" ; shift
    local plist="$AGENTS/$label.plist"
    local schedule="$1" ; shift

    cat > "$plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$label</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>$RUNNER</string>
$(for a in "$@"; do echo "    <string>$a</string>"; done)
  </array>
  <key>WorkingDirectory</key><string>$HERE</string>
  <key>StandardOutPath</key><string>$HERE/data/logs/$label.out.log</string>
  <key>StandardErrorPath</key><string>$HERE/data/logs/$label.err.log</string>
  <key>RunAtLoad</key><false/>
  <key>StartCalendarInterval</key>
$schedule
</dict>
</plist>
PLIST

    launchctl unload "$plist" 2>/dev/null || true
    launchctl load "$plist"
    echo "  installed $label"
}

echo "Installing scheduled jobs for $HERE"

write_plist "com.krxflow.collect" \
'  <dict>
    <key>Hour</key><integer>19</integer>
    <key>Minute</key><integer>0</integer>
  </dict>'

if [[ $WANT_PUSH -eq 1 ]]; then
    write_plist "com.krxflow.report" \
'  <dict>
    <key>Weekday</key><integer>1</integer>
    <key>Hour</key><integer>8</integer>
    <key>Minute</key><integer>0</integer>
  </dict>' \
        "--report" "--push"
else
    write_plist "com.krxflow.report" \
'  <dict>
    <key>Weekday</key><integer>1</integer>
    <key>Hour</key><integer>8</integer>
    <key>Minute</key><integer>0</integer>
  </dict>' \
        "--report"
fi

cat <<'NOTES'

Installed. Notes:

  - Times are your Mac's local time. Korean data for day D is published after
    the 15:30 KST close, and the pipeline only ever asks for days strictly
    before today, so the exact hour is not critical.

  - If the Mac is asleep at 19:00 the run is skipped, not queued. That is
    fine: the next run looks 30 days back and fills any gap.

  - Check it is loaded:      launchctl list | grep krxflow
  - Run one now by hand:     bash scripts/run_update.sh
  - Watch what it did:       tail -f data/logs/updates.log
  - Remove:                  bash scripts/uninstall_schedule.sh

NOTES
