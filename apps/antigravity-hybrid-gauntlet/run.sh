#!/usr/bin/env bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Runs the Antigravity Hybrid Gauntlet multi-agent tool & demo.
#
#   ./run.sh                    runs the full interactive Rich HUD demo
#   ./run.sh --minimal          runs the linear SDK console quickstart
#   ./run.sh --files <path.py> --test-cmd "<cmd>"
#                               audits & patches your own Python file(s)
#   MODEL=e2b ./run.sh          runs with the compact E2B Gemma 4 checkpoint
#
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:-26b}"

# The code uses PEP 604 unions and str.removesuffix, so 3.10 is the floor.
if ! python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "error: Python 3.10+ is required (found $(python3 -V 2>&1))." >&2
  exit 1
fi

# litert-lm ships GPU wheels for Apple Silicon. Elsewhere the install may fail
# or fall back to CPU, so say so up front rather than failing mysteriously.
if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "note: this demo is developed and tested on Apple Silicon macOS."
  echo "      On $(uname -s)/$(uname -m), litert-lm GPU acceleration may be"
  echo "      unavailable and the local model will be slow or unsupported."
  echo
fi

if [[ ! -x ./.venv/bin/python ]]; then
  echo "==> Creating virtual environment (.venv) and installing requirements..."
  rm -rf ./.venv
  python3 -m venv .venv
  ./.venv/bin/pip install --upgrade pip -q
  ./.venv/bin/pip install -r requirements.txt
fi

# Only set a CA bundle if the environment has not already chosen one, so we do
# not override a corporate proxy's custom trust store.
if [[ -z "${SSL_CERT_FILE:-}" ]]; then
  export SSL_CERT_FILE="$(./.venv/bin/python -c 'import certifi; print(certifi.where())')"
fi
export LITERT_MODEL_PATH="${LITERT_MODEL_PATH:-$HOME/.litert-lm/models/gemma4-${MODEL}/model.litertlm}"

if [[ -f secrets.local.sh ]]; then
  # shellcheck disable=SC1091
  source secrets.local.sh
fi

if [[ -z "${GEMINI_API_KEY:-}" && "${GOOGLE_GENAI_USE_VERTEXAI:-}" != "true" ]]; then
  echo "note: no cloud credential found. Step 1 (Cloud Architect) will be skipped"
  echo "      and the built-in offline plan used instead. Export GEMINI_API_KEY"
  echo "      to run the live hybrid cloud/on-device workflow."
  echo
fi

# Reap a harness left behind by a previous *run of this checkout* only. A bare
# `pkill -f localharness` would also kill harnesses belonging to other projects
# or to the user's editor, which is not ours to do.
pkill -f "^${PWD}/\.venv/.*localharness" 2>/dev/null || true

if [[ $# -eq 0 || "$1" == -* ]]; then
  exec ./.venv/bin/python -u hybrid_gauntlet.py "$@"
else
  exec ./.venv/bin/python -u "$@"
fi
