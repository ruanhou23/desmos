#!/usr/bin/env bash
# Desmos Bézier Renderer Launcher
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Run backend directly using system Python
python3 backend.py "$@"
