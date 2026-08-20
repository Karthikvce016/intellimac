#!/bin/zsh
# NervNet dashboard launcher
cd "$(dirname "$0")"
exec .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8600 "$@"
