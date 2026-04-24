#!/usr/bin/env bash
# Generate predictor_pb2.py and predictor_pb2_grpc.py from proto/predictor.proto.
#
# Can be invoked locally (from services/grpc_predictor/) or from the Docker
# build context where the working directory is /app.
#
# Usage:
#   ./generate_pb2.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTO_DIR="${HERE}/proto"
OUT_DIR="${HERE}"

echo "[generate_pb2] proto dir : ${PROTO_DIR}"
echo "[generate_pb2] output dir: ${OUT_DIR}"

python -m grpc_tools.protoc \
    --proto_path="${PROTO_DIR}" \
    --python_out="${OUT_DIR}" \
    --grpc_python_out="${OUT_DIR}" \
    "${PROTO_DIR}/predictor.proto"

echo "[generate_pb2] generated:"
ls -1 "${OUT_DIR}"/predictor_pb2*.py
