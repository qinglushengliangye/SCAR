#!/usr/bin/env bash
# Download third-party JS assets into static/vendor/ so the Gradio front-end
# can render the knowledge graph without any outbound network calls at
# runtime (useful for air-gapped demos and reproducible deployments).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="${HERE}/static/vendor"
mkdir -p "${VENDOR_DIR}"

VIS_URL="${VIS_NETWORK_URL:-https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js}"
VIS_TARGET="${VENDOR_DIR}/vis-network.min.js"

if [[ -s "${VIS_TARGET}" ]]; then
  echo "[fetch_vendor] ${VIS_TARGET} already present ($(du -h "${VIS_TARGET}" | cut -f1))"
else
  echo "[fetch_vendor] downloading ${VIS_URL} -> ${VIS_TARGET}"
  if command -v curl >/dev/null 2>&1; then
    curl -fSL --retry 3 -o "${VIS_TARGET}" "${VIS_URL}"
  elif command -v wget >/dev/null 2>&1; then
    wget -q -O "${VIS_TARGET}" "${VIS_URL}"
  else
    echo "[fetch_vendor] neither curl nor wget available" >&2
    exit 1
  fi
  echo "[fetch_vendor] done ($(du -h "${VIS_TARGET}" | cut -f1))"
fi

# Sanity check: the vendored file must at least declare the vis namespace.
if ! head -c 4096 "${VIS_TARGET}" | grep -q "vis"; then
  echo "[fetch_vendor] downloaded file does not look like vis-network.min.js" >&2
  exit 2
fi
