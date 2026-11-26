#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
python3 -m uvicorn backend.app:app --host 0.0.0.0 --port 8088
