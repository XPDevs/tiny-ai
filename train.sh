#!/usr/bin/env bash
# Train the from-scratch model, test it, and emit a .GenAI file.
# Usage: ./train.sh [output.GenAI]
set -euo pipefail
cd "$(dirname "$0")"
OUT="${1:-model.GenAI}"
echo "training tiny-ai -> $OUT"
node train.js "$OUT"
