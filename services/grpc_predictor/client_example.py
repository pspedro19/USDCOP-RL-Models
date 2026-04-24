"""
Demo client for the USDCOP H5 gRPC prediction service.

Sends a sample feature vector to ``PredictorService.Predict`` and calls
``HealthCheck``. Prints both responses in a human-readable format.

Usage:
    # Locally (after ./generate_pb2.sh has produced the pb2 files):
    python client_example.py

    # Against a custom host/port:
    GRPC_TARGET=localhost:50051 python client_example.py
"""

from __future__ import annotations

import os
import sys
from typing import Dict

import grpc

try:
    import predictor_pb2  # type: ignore[import-not-found]
    import predictor_pb2_grpc  # type: ignore[import-not-found]
except ImportError as exc:
    sys.stderr.write(
        "[client] ERROR: predictor_pb2 / predictor_pb2_grpc not found.\n"
        "[client] Run ./generate_pb2.sh first.\n"
        f"[client] Underlying import error: {exc}\n"
    )
    raise


# Sample H5 feature vector (roughly representative; values are z-scored
# for the macro features and raw for RSI).
SAMPLE_FEATURES: Dict[str, float] = {
    "dxy_z": 0.5,
    "wti_z": -0.3,
    "rsi_14": 45.0,
    "vol_regime_ratio": 1.15,
    "trend_slope_60d": -0.002,
    "ust10y_z": 0.2,
    "vix_z": 0.8,
    "embi_col_z": -0.1,
}


def main() -> int:
    target = os.environ.get("GRPC_TARGET", "localhost:50051")
    print(f"[client] connecting to {target}")

    # Insecure channel is fine for local demos; production would use TLS.
    with grpc.insecure_channel(target) as channel:
        stub = predictor_pb2_grpc.PredictorServiceStub(channel)

        # -- HealthCheck --
        print("[client] -> HealthCheck")
        try:
            health = stub.HealthCheck(predictor_pb2.HealthRequest(), timeout=5.0)
            print(
                f"[client] <- HealthCheck: ready={health.ready} "
                f"model_loaded={health.model_loaded!r}"
            )
        except grpc.RpcError as exc:
            print(f"[client] HealthCheck failed: {exc.code().name}: {exc.details()}")
            return 1

        # -- Predict --
        print("[client] -> Predict with sample features:")
        for k, v in sorted(SAMPLE_FEATURES.items()):
            print(f"           {k:>20s} = {v:+.4f}")

        request = predictor_pb2.PredictRequest(features=SAMPLE_FEATURES)
        try:
            response = stub.Predict(request, timeout=5.0)
        except grpc.RpcError as exc:
            print(f"[client] Predict failed: {exc.code().name}: {exc.details()}")
            return 1

        print("[client] <- Predict response:")
        print(f"           direction        = {response.direction}")
        print(f"           confidence       = {response.confidence:.4f}")
        print(f"           ensemble_return  = {response.ensemble_return:+.6f}")
        print(f"           model_version    = {response.model_version}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
