#!/usr/bin/env bash
set -euo pipefail

: "${DEPLOY_PATH:?DEPLOY_PATH is required}"
: "${DEPLOY_REF:?DEPLOY_REF is required}"
: "${REPO_URL:?REPO_URL is required}"

parent_dir="$(dirname "${DEPLOY_PATH}")"
mkdir -p "${parent_dir}"

if [ ! -d "${DEPLOY_PATH}/.git" ]; then
  git clone --branch "${DEPLOY_REF}" "${REPO_URL}" "${DEPLOY_PATH}"
else
  git -C "${DEPLOY_PATH}" fetch origin "${DEPLOY_REF}"
  git -C "${DEPLOY_PATH}" checkout "${DEPLOY_REF}"
  git -C "${DEPLOY_PATH}" pull --ff-only origin "${DEPLOY_REF}"
fi

git -C "${DEPLOY_PATH}" rev-parse HEAD
echo "Repository sync complete at ${DEPLOY_PATH}"

