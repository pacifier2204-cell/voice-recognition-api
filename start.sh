#!/usr/bin/env bash

echo "Starting Voice Recognition API..."

uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-10000}
