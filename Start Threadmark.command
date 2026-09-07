#!/bin/zsh
set -e

APP_DIR="${0:A:h}"
APP_URL="http://127.0.0.1:8765"
LOG_FILE="$APP_DIR/server.log"
PID_FILE="$APP_DIR/.server.pid"

cd "$APP_DIR"

if curl --silent --fail --max-time 1 "$APP_URL" >/dev/null 2>&1; then
  open "$APP_URL"
  exit 0
fi

nohup "$APP_DIR/run.sh" >"$LOG_FILE" 2>&1 &
SERVER_PID=$!
print -r -- "$SERVER_PID" >"$PID_FILE"

for attempt in {1..40}; do
  if curl --silent --fail --max-time 1 "$APP_URL" >/dev/null 2>&1; then
    open "$APP_URL"
    exit 0
  fi

  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    open -a TextEdit "$LOG_FILE"
    echo "Threadmark could not start. The server log has been opened."
    read -k 1 "?Press any key to close."
    exit 1
  fi

  sleep 0.25
done

open -a TextEdit "$LOG_FILE"
echo "Threadmark took too long to start. The server log has been opened."
read -k 1 "?Press any key to close."
exit 1
