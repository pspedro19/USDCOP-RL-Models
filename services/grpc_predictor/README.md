# USDCOP H5 gRPC Prediction Service

A self-contained gRPC microservice that serves 5-day directional predictions
for USDCOP, backed by the H5 Smart Simple Ridge + BayesianRidge ensemble
(with a deterministic in-memory stub as fallback so the server always starts).

This is one of the two non-REST technologies for the MLOps course project.

---

## Contract

- **Port**: `50051`
- **Service**: `predictor.PredictorService`
- **Proto**: [`proto/predictor.proto`](proto/predictor.proto)

### RPCs

| RPC | Request | Response |
|-----|---------|----------|
| `Predict` | `map<string,double> features` | `direction (LONG/SHORT/FLAT)`, `confidence [0,1]`, `ensemble_return` (signed log-return), `model_version` |
| `HealthCheck` | (empty) | `ready (bool)`, `model_loaded (string)` |

On any internal error `Predict` returns `direction="UNKNOWN", confidence=0.0`
and encodes the error class in `model_version`. The server never crashes on
bad input.

---

## Build & Run (Docker — recommended)

From the repo root:

```bash
# 1) Build the image
docker build -t usdcop-grpc-predictor:latest services/grpc_predictor

# 2) Run (read-only bind of the outputs dir — harmless if models are missing)
docker run --rm -p 50051:50051 \
    -v "$PWD/outputs:/app/outputs:ro" \
    --name usdcop-grpc-predictor \
    usdcop-grpc-predictor:latest
```

You should see:

```
... gRPC predictor listening on 0.0.0.0:50051
... model description: stub-deterministic   # or ridge+br (h5_smart_simple::...)
```

---

## Run locally (no Docker)

```bash
cd services/grpc_predictor
pip install -r requirements.txt

# Generate the pb2 files once
./generate_pb2.sh

# Start the server
python server.py
```

In another shell:

```bash
cd services/grpc_predictor
python client_example.py
```

Expected client output:

```
[client] connecting to localhost:50051
[client] -> HealthCheck
[client] <- HealthCheck: ready=True model_loaded='stub-deterministic'
[client] -> Predict with sample features:
                      dxy_z = +0.5000
                     rsi_14 = +45.0000
                   ...
[client] <- Predict response:
           direction        = SHORT
           confidence       = 0.73...
           ensemble_return  = -0.007...
           model_version    = stub-<8hex>
```

Run the client against a different target with `GRPC_TARGET`:

```bash
GRPC_TARGET=localhost:50051 python client_example.py
```

---

## Model loading behavior

At startup the server searches, in order:

1. `$H5_MODEL_DIR` (default `/app/outputs/forecasting/h5_weekly_models/latest`)
2. `outputs/forecasting/h5_weekly_models/latest`
3. `<repo>/outputs/forecasting/h5_weekly_models/latest`

In the first directory that exists AND contains **both** a Ridge `.pkl` and a
BayesianRidge `.pkl` (filenames matched by substring: `ridge`, `bayesian`/`br`),
the files are loaded with `joblib` and used for inference. If a
`feature_order.json` / `features.json` is present next to the pkls, it is
honored; otherwise features are sorted alphabetically at predict time.

If no match is found, the server falls back to a **deterministic stub** that
hashes the feature map to a stable pseudo log-return in roughly ±100 bps.
This is intentional for demo / offline robustness; in production you would
gate the server on real pkls being present.

---

## Files

```
services/grpc_predictor/
├── Dockerfile              # python:3.11-slim image, generates pb2 at build time
├── README.md               # this file
├── client_example.py       # demo client (Predict + HealthCheck)
├── generate_pb2.sh         # invokes grpc_tools.protoc to produce pb2 files
├── proto/
│   └── predictor.proto     # shared contract (proto3, package=predictor)
├── requirements.txt        # grpcio, grpcio-tools, sklearn, pandas, numpy, joblib
└── server.py               # gRPC server (binds 0.0.0.0:50051)
```

---

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `GRPC_PORT` | `50051` | Listening port |
| `H5_MODEL_DIR` | `/app/outputs/forecasting/h5_weekly_models/latest` | First directory to try for Ridge+BR pkls |
| `GRPC_TARGET` | `localhost:50051` | Used by `client_example.py` only |

---

## Troubleshooting

- **`ImportError: predictor_pb2`** — you skipped pb2 generation. Run
  `./generate_pb2.sh` (or rebuild the Docker image).
- **Server says `stub-deterministic` but you expected real models** — check
  the startup log for the "search paths" list; the first matching directory
  must contain at least one pkl matching `*ridge*` and one matching
  `*bayesian*`/`*br*`.
- **`UNKNOWN` direction in responses** — the server caught an exception
  during `predict()`. Check server logs; input is returned verbatim.
