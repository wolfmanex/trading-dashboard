#!/usr/bin/env bash
# Run the intraday checks every 15 minutes (New York time) until END (HH:MM), saving the sent alerts to
# the signal-log branch after each one. Used by .github/workflows/market-session.yml; expects the
# signal-log branch checked out in ./signal-log-data. Writes the number of in-session checks to
# $GITHUB_OUTPUT as `checks`.
set -u
end="$1"
checks=0

save_alerts() {
    (
        cd signal-log-data || exit 0
        for file in intraday_alerts.csv position_alerts.csv; do
            if [ -f "$file" ]; then git add "$file"; fi
        done
        git diff --cached --quiet && exit 0
        git commit -q -m "Intraday alerts $(TZ=America/New_York date '+%F %H:%M')"
        # The scan and the app (journal) also write to this branch; rebase onto them if they got there first.
        for attempt in 1 2 3; do
            git push -q origin signal-log && exit 0
            git pull -q --rebase origin signal-log
        done
        echo "Could not push the sent alerts."
    )
}

while :; do
    now=$(TZ=America/New_York date +%H:%M)
    [[ "$now" < "$end" ]] || break
    # Pick up journal changes made in the app since the last check.
    # --autostash: the afternoon job may start with merged, uncommitted alert files (see market-session.yml).
    git -C signal-log-data pull -q --rebase --autostash origin signal-log || true
    python intraday_alerts.py --log signal-log-data/signal_log.csv --sent signal-log-data/intraday_alerts.csv \
        --journal signal-log-data/journal.csv --position-sent signal-log-data/position_alerts.csv \
        || echo "Intraday check failed; trying again next round."
    if [[ ! "$now" < "09:45" ]]; then checks=$((checks + 1)); fi
    save_alerts
    # Sleep to a few seconds past the next quarter hour.
    sleep $(( 900 - $(date +%s) % 900 + 5 ))
done

echo "checks=$checks" >> "${GITHUB_OUTPUT:-/dev/null}"
