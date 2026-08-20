#!/bin/bash
# ==========================================================
# NervNet MAC Controller — One-Click Project Launcher
# ==========================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "=========================================================="
echo "◈ Starting NervNet MAC Controller Mission-Control Dashboard"
echo "=========================================================="

# Check for Python
if command -v python3 &>/dev/null; then
    PY="python3"
else
    PY="python"
fi

# Set up virtual environment if needed
if [ ! -d "dashboard-v2/.venv" ]; then
    echo "[+] Creating virtual environment in dashboard-v2/.venv..."
    $PY -m venv dashboard-v2/.venv
    dashboard-v2/.venv/bin/pip install --upgrade pip
    dashboard-v2/.venv/bin/pip install -r requirements.txt
fi

echo "[+] Starting server at http://127.0.0.1:8600"
echo "[+] Press CTRL+C to stop the dashboard"
echo "=========================================================="

cd dashboard-v2
exec .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8600 "$@"
