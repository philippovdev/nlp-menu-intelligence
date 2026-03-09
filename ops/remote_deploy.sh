#!/usr/bin/env bash
set -euo pipefail

: "${DEPLOY_PATH:?DEPLOY_PATH is required}"

cd "${DEPLOY_PATH}"

docker compose up -d --build api

cd frontend
npm ci
npm run build

nginx -t
service nginx reload

echo "Application deploy complete."
