# deploy/

macOS launchd plists that keep the data pipeline running without relying on
`python predict.py serve` being up 24/7.

## Why this exists

The in-process APScheduler (in `predict.py::_start_scheduler`) only fires
while the FastAPI server is running. For a personal project that doesn't run
on a server, that means:

- **Audit history starves**: `/upcoming` snapshots only get written when
  someone opens the web UI (or the server happens to be up at 09/15/21/03
  Asia/Shanghai). One week of typical usage produced **66 distinct snapshots**
  vs. the theoretical max of ~28/week.
- **Backfill lags**: matches table falls behind reality by 4-5 days when the
  daily 06:00 cron never fires.

These plists move both jobs out-of-process. They run whether the web server
is up or down.

## Install

```bash
# Snapshot (every 6 hours)
cp deploy/com.aaronsun.football-predictor-claude.snapshot.plist \
   ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.aaronsun.football-predictor-claude.snapshot.plist

# Backfill (daily at 06:00)
cp deploy/com.aaronsun.football-predictor-claude.backfill.plist \
   ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.aaronsun.football-predictor-claude.backfill.plist
```

`RunAtLoad=true` on the snapshot plist means it fires immediately on load.
The backfill plist waits for its next 06:00 slot (heavier job, don't want
both firing at once during install).

## Verify

```bash
launchctl list | grep football-predictor-claude
# Expect two entries with PID -, ExitCode 0

tail -f /tmp/football-predictor-claude.snapshot.log
tail -f /tmp/football-predictor-claude.backfill.log
```

The snapshot log should show a fresh JSON payload every 6 hours:

```json
{
  "ok": true,
  "fixture_count": 10,
  "days_ahead": 14,
  "leagues_queried": 42,
  "history_dir": "/Users/seek/Documents/football-predictor-claude/data/cache/upcoming/history"
}
```

## Uninstall

```bash
launchctl unload ~/Library/LaunchAgents/com.aaronsun.football-predictor-claude.snapshot.plist
launchctl unload ~/Library/LaunchAgents/com.aaronsun.football-predictor-claude.backfill.plist
rm ~/Library/LaunchAgents/com.aaronsun.football-predictor-claude.snapshot.plist
rm ~/Library/LaunchAgents/com.aaronsun.football-predictor-claude.backfill.plist
```

## What this does NOT do

These plists don't run the web server itself. If you want the FastAPI app up
24/7 (for the UI / browser access), set up a separate tmux-or-supervisord
arrangement — these plists only handle the headless data pipeline.

## Relationship to the in-process scheduler

You can leave the in-process scheduler enabled. When both run, you get:

- **Server up**: snapshots from APScheduler every 6h *and* from launchd every
  6h. The on-disk caches (TSDB 6h, fd.org 24h) absorb the duplication, so
  there's no real cost — just redundancy.
- **Server down**: only launchd snapshots fire. This is the failure mode we
  built these plists to handle.

To disable the in-process one if you prefer launchd-only:
```bash
export FOOTBALL_PREDICTOR_UPCOMING_SNAPSHOT=false
```
