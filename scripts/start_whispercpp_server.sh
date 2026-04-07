#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHISPERCPP_REPO="${WHISPERCPP_REPO:-${ROOT_DIR}/tmp/whisper.cpp}"
WHISPERCPP_BUILD_DIR="${WHISPERCPP_BUILD_DIR:-${WHISPERCPP_REPO}/build}"
if [[ -x "${ROOT_DIR}/.venv/bin/cmake" ]]; then
  CMAKE_BIN="${ROOT_DIR}/.venv/bin/cmake"
else
  CMAKE_BIN="cmake"
fi
WHISPERCPP_CUDA_ARCHITECTURES="${WHISPERCPP_CUDA_ARCHITECTURES:-86}"
WHISPERCPP_BUILD_JOBS="${WHISPERCPP_BUILD_JOBS:-1}"
WHISPERCPP_HOST="${WHISPERCPP_HOST:-127.0.0.1}"
WHISPERCPP_PORT="${WHISPERCPP_PORT:-8178}"
WHISPERCPP_MODEL_DIR="${WHISPERCPP_MODEL_DIR:-/media/yaroslav/DATA/ai_models/whisper.cpp/models}"
WHISPERCPP_MODEL_NAME="${WHISPERCPP_MODEL_NAME:-medium.en}"
WHISPERCPP_MODEL_PATH="${WHISPERCPP_MODEL_PATH:-${WHISPERCPP_MODEL_DIR}/ggml-${WHISPERCPP_MODEL_NAME}.bin}"
WHISPERCPP_PROMPT="${WHISPERCPP_PROMPT:-Yaroslav. Jaroslav. Ukraine. 26 years old. Hello everyone, my name is Yaroslav.}"

if [[ ! -d "${WHISPERCPP_REPO}" ]]; then
  echo "Missing whisper.cpp repo at ${WHISPERCPP_REPO}." >&2
  echo "Clone it first or set WHISPERCPP_REPO." >&2
  exit 1
fi

mkdir -p "${WHISPERCPP_MODEL_DIR}"

if [[ ! -x "${WHISPERCPP_BUILD_DIR}/bin/whisper-server" ]]; then
  "${CMAKE_BIN}" -S "${WHISPERCPP_REPO}" -B "${WHISPERCPP_BUILD_DIR}" \
    -DGGML_CUDA=1 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${WHISPERCPP_CUDA_ARCHITECTURES}"
  "${CMAKE_BIN}" --build "${WHISPERCPP_BUILD_DIR}" --parallel "${WHISPERCPP_BUILD_JOBS}" --config Release --target whisper-server
fi

if [[ ! -f "${WHISPERCPP_MODEL_PATH}" ]]; then
  (
    cd "${WHISPERCPP_REPO}"
    ./models/download-ggml-model.sh "${WHISPERCPP_MODEL_NAME}" "${WHISPERCPP_MODEL_DIR}"
  )
fi

exec "${WHISPERCPP_BUILD_DIR}/bin/whisper-server" \
  --host "${WHISPERCPP_HOST}" \
  --port "${WHISPERCPP_PORT}" \
  --language en \
  --model "${WHISPERCPP_MODEL_PATH}" \
  --prompt "${WHISPERCPP_PROMPT}" \
  --no-timestamps \
  --suppress-nst
