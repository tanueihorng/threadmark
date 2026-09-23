#!/bin/sh
# Start Threadmark and open the recorder, from any working directory.
set -eu

APP_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec /bin/zsh "$APP_DIR/Start Threadmark.command" "$@"
