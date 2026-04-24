"""
USDCOP H5 Smart Simple gRPC Prediction Server.

Loads a Ridge + BayesianRidge ensemble from
``outputs/forecasting/h5_weekly_models/latest/*.pkl`` if present, otherwise
falls back to a deterministic in-memory stub so the server always starts
(demo / course robustness).

Binds to ``0.0.0.0:50051`` by default. Override via ``GRPC_PORT`` env var.

Contract (see ``proto/predictor.proto``):
    Service : predictor.PredictorService
    RPCs    : Predict, HealthCheck

Logging goes to stdout in a simple timestamped format.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import pathlib
import signal
import sys
import threading
import time
from concurrent import futures
from typing import Any, Dict, List, Optional, Tuple

import grpc

# ---------------------------------------------------------------------------
# pb2 imports — generated via generate_pb2.sh (run during Docker build or
# locally before invoking this script).
# ---------------------------------------------------------------------------
try:
    import predictor_pb2  # type: ignore[import-not-found]
    import predictor_pb2_grpc  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover - clear error for the operator
    sys.stderr.write(
        "[server] ERROR: predictor_pb2 / predictor_pb2_grpc not found.\n"
        "[server] Run ./generate_pb2.sh first, or let Docker build it.\n"
        f"[server] Underlying import error: {exc}\n"
    )
    raise


# ---------------------------------------------------------------------------
# Logging setup — stdout, timestamped.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("grpc_predictor")


# ---------------------------------------------------------------------------
# Paths / configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL_DIR = pathlib.Path(
    os.environ.get(
        "H5_MODEL_DIR",
        "/app/outputs/forecasting/h5_weekly_models/latest",
    )
)
FALLBACK_MODEL_DIRS: List[pathlib.Path] = [
    DEFAULT_MODEL_DIR,
    pathlib.Path("outputs/forecasting/h5_weekly_models/latest"),
    pathlib.Path(__file__).resolve().parent.parent.parent
    / "outputs"
    / "forecasting"
    / "h5_weekly_models"
    / "latest",
]
GRPC_PORT = int(os.environ.get("GRPC_PORT", "50051"))


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
class EnsembleModel:
    """
    Wraps the Ridge + BayesianRidge H5 ensemble or a deterministic stub.

    The ``predict`` method always accepts a ``dict[str, float]`` of features
    and returns ``(direction, confidence, ensemble_return, model_version)``.
    """

    # Thresholds tuned for the stub so all three directions appear in demos.
    LONG_THRESHOLD = 0.0005  # > +5 bps log-return => LONG
    SHORT_THRESHOLD = -0.0005  # < -5 bps log-return => SHORT

    def __init__(
        self,
        ridge: Optional[Any] = None,
        br: Optional[Any] = None,
        feature_order: Optional[List[str]] = None,
        version: str = "stub-deterministic",
    ) -> None:
        self._ridge = ridge
        self._br = br
        self._feature_order = feature_order or []
        self._version = version
        self._is_stub = ridge is None or br is None

    # ---- factory ---------------------------------------------------------
    @classmethod
    def load(cls, candidate_dirs: List[pathlib.Path]) -> "EnsembleModel":
        """Try each candidate directory; fall back to stub if none match."""
        for model_dir in candidate_dirs:
            try:
                if not model_dir.exists():
                    continue
                ridge_path = cls._first_match(model_dir, ["ridge", "Ridge"])
                br_path = cls._first_match(
                    model_dir, ["bayesian", "bayesianridge", "br"]
                )
                if ridge_path is None or br_path is None:
                    logger.info(
                        "model dir %s exists but does not contain both "
                        "ridge+br pkls; skipping",
                        model_dir,
                    )
                    continue
                logger.info("loading ridge from %s", ridge_path)
                logger.info("loading br    from %s", br_path)
                import joblib  # local import so stub-only use doesn't need it

                ridge = joblib.load(ridge_path)
                br = joblib.load(br_path)

                feature_order = cls._try_load_feature_order(model_dir)
                version = f"h5_smart_simple::{model_dir.name}"
                logger.info(
                    "loaded ensemble: ridge=%s, br=%s, feature_order=%s, "
                    "version=%s",
                    type(ridge).__name__,
                    type(br).__name__,
                    feature_order[:5] + (["..."] if len(feature_order) > 5 else []),
                    version,
                )
                return cls(
                    ridge=ridge,
                    br=br,
                    feature_order=feature_order,
                    version=version,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "failed to load model from %s: %s", model_dir, exc
                )

        logger.warning(
            "no Ridge+BR pkls found in any candidate dir; falling back to "
            "deterministic stub. Searched: %s",
            [str(p) for p in candidate_dirs],
        )
        return cls()  # stub

    # ---- helpers ---------------------------------------------------------
    @staticmethod
    def _first_match(
        model_dir: pathlib.Path, name_tokens: List[str]
    ) -> Optional[pathlib.Path]:
        """Return the first *.pkl whose filename contains any of the tokens."""
        for pkl in sorted(model_dir.glob("*.pkl")):
            name = pkl.name.lower()
            if any(tok.lower() in name for tok in name_tokens):
                return pkl
        return None

    @staticmethod
    def _try_load_feature_order(model_dir: pathlib.Path) -> List[str]:
        """Best-effort load of a feature_order.json; empty list on miss."""
        for candidate in ["feature_order.json", "features.json"]:
            path = model_dir / candidate
            if path.exists():
                try:
                    import json

                    data = json.loads(path.read_text())
                    if isinstance(data, list):
                        return [str(x) for x in data]
                    if isinstance(data, dict) and "features" in data:
                        return [str(x) for x in data["features"]]
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning(
                        "could not parse %s: %s", path, exc
                    )
        return []

    # ---- prediction ------------------------------------------------------
    @property
    def version(self) -> str:
        return self._version

    @property
    def description(self) -> str:
        if self._is_stub:
            return "stub-deterministic"
        return f"ridge+br ({self._version})"

    def predict(
        self, features: Dict[str, float]
    ) -> Tuple[str, float, float, str]:
        """Return (direction, confidence, ensemble_return, model_version)."""
        if self._is_stub:
            return self._predict_stub(features)
        return self._predict_real(features)

    # ---- real ensemble ---------------------------------------------------
    def _predict_real(
        self, features: Dict[str, float]
    ) -> Tuple[str, float, float, str]:
        import numpy as np  # local import keeps stub-only paths lighter

        feature_names = self._feature_order or sorted(features.keys())
        x = np.array(
            [[float(features.get(name, 0.0)) for name in feature_names]],
            dtype=np.float64,
        )
        ridge_pred = float(self._ridge.predict(x)[0])
        br_pred = float(self._br.predict(x)[0])
        ensemble_return = 0.5 * (ridge_pred + br_pred)

        direction = self._direction_from_return(ensemble_return)
        # Confidence = bounded magnitude scaled by model agreement.
        agreement = 1.0 - min(1.0, abs(ridge_pred - br_pred) / 0.01)
        magnitude = min(1.0, abs(ensemble_return) / 0.01)
        confidence = max(0.0, min(1.0, 0.5 * agreement + 0.5 * magnitude))

        return direction, confidence, ensemble_return, self._version

    # ---- deterministic stub ---------------------------------------------
    def _predict_stub(
        self, features: Dict[str, float]
    ) -> Tuple[str, float, float, str]:
        """
        Deterministic stub: hash the feature dict to a stable pseudo-return.

        Produces values in roughly [-0.01, +0.01] (±100 bps log-return)
        so that LONG/SHORT/FLAT are all reachable with standard feature
        magnitudes. Stable across runs for the same input.
        """
        if not features:
            return "FLAT", 0.0, 0.0, "stub-empty-features"

        # Build a stable representation and hash it to 8 bytes.
        canonical = ",".join(
            f"{k}={features[k]:.6f}" for k in sorted(features.keys())
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).digest()
        # Map 4 bytes -> int -> signed float in [-1, 1]
        raw = int.from_bytes(digest[:4], "big", signed=False)
        u = (raw / 0xFFFFFFFF) * 2.0 - 1.0  # [-1, 1]

        # Scale to ±0.01 log-return and add a gentle feature-driven nudge so
        # clients see features matter.
        feature_sum = sum(v for v in features.values() if math.isfinite(v))
        nudge = math.tanh(feature_sum / 10.0) * 0.003
        ensemble_return = u * 0.01 + nudge

        direction = self._direction_from_return(ensemble_return)
        confidence = min(1.0, abs(ensemble_return) / 0.01)
        version = f"stub-{digest.hex()[:8]}"
        return direction, confidence, ensemble_return, version

    @classmethod
    def _direction_from_return(cls, r: float) -> str:
        if r > cls.LONG_THRESHOLD:
            return "LONG"
        if r < cls.SHORT_THRESHOLD:
            return "SHORT"
        return "FLAT"


# ---------------------------------------------------------------------------
# gRPC servicer
# ---------------------------------------------------------------------------
class PredictorServicer(predictor_pb2_grpc.PredictorServiceServicer):
    """gRPC servicer bridging the ``PredictorService`` contract to the model."""

    def __init__(self, model: EnsembleModel) -> None:
        self._model = model
        self._ready = True

    # -- Predict ----------------------------------------------------------
    def Predict(self, request, context):  # noqa: N802 (gRPC naming)
        """Handle a single Predict RPC. Never raises — always returns a response."""
        features = dict(request.features)
        logger.info(
            "Predict: %d feature(s) received; sample=%s",
            len(features),
            list(features.items())[:3],
        )
        try:
            direction, confidence, ensemble_return, version = self._model.predict(
                features
            )
            logger.info(
                "Predict result: dir=%s conf=%.3f ret=%+.6f version=%s",
                direction,
                confidence,
                ensemble_return,
                version,
            )
            return predictor_pb2.PredictResponse(
                direction=direction,
                confidence=float(confidence),
                ensemble_return=float(ensemble_return),
                model_version=version,
            )
        except Exception as exc:  # defensive — NEVER crash the server
            logger.exception("Predict failed: %s", exc)
            return predictor_pb2.PredictResponse(
                direction="UNKNOWN",
                confidence=0.0,
                ensemble_return=0.0,
                model_version=f"error:{type(exc).__name__}",
            )

    # -- HealthCheck ------------------------------------------------------
    def HealthCheck(self, request, context):  # noqa: N802
        """Return readiness + which model is currently loaded."""
        return predictor_pb2.HealthResponse(
            ready=self._ready,
            model_loaded=self._model.description,
        )


# ---------------------------------------------------------------------------
# Server bootstrap
# ---------------------------------------------------------------------------
def serve() -> None:
    """Start the gRPC server and block until SIGINT/SIGTERM."""
    logger.info("starting gRPC predictor service")
    logger.info("search paths for H5 models:")
    for p in FALLBACK_MODEL_DIRS:
        logger.info("  - %s (exists=%s)", p, p.exists())

    model = EnsembleModel.load(FALLBACK_MODEL_DIRS)
    logger.info("model description: %s", model.description)

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", 4 * 1024 * 1024),
            ("grpc.max_receive_message_length", 4 * 1024 * 1024),
        ],
    )
    predictor_pb2_grpc.add_PredictorServiceServicer_to_server(
        PredictorServicer(model), server
    )
    bind_addr = f"0.0.0.0:{GRPC_PORT}"
    server.add_insecure_port(bind_addr)
    server.start()
    logger.info("gRPC predictor listening on %s", bind_addr)

    # Graceful shutdown on SIGTERM/SIGINT.
    stop_event = threading.Event()

    def _handle_signal(signum, _frame):
        logger.info("received signal %s; shutting down", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        while not stop_event.is_set():
            time.sleep(1.0)
    finally:
        logger.info("stopping server (grace=5s)")
        server.stop(grace=5.0).wait()
        logger.info("server stopped")


if __name__ == "__main__":
    serve()
