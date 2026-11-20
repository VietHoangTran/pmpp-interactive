#!/bin/sh
# Start the interactive textbook. Installs dependencies on first run.
set -e
cd "$(dirname "$0")/app"
command -v node >/dev/null 2>&1 || { echo "Node.js 20 or newer is required: https://nodejs.org"; exit 1; }
[ -d node_modules ] || npm install
exec npm start
