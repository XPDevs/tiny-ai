#!/usr/bin/env bash
# chat.sh — chat with a trained model in the terminal.
#
#   ./chat.sh                 # uses ./model.GenAI
#   ./chat.sh my_model.GenAI  # uses another model file
#   TEMP=0.6 ./chat.sh        # change the sampling temperature (default 0.3)
set -euo pipefail
cd "$(dirname "$0")"
exec node chat.js "$@"
