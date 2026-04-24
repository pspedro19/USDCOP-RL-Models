# Course Project Integration Tests

Integration tests that validate the MLOps course project deliverable: gRPC predictor
service, Kafka (Redpanda) bridge, plus supporting infrastructure (Airflow, MLflow,
PostgreSQL, Redis, SignalBridge).

Tests automatically skip when a required service is unreachable, so CI without the
full Docker stack still passes.

## Running

```bash
# Full stack (requires docker compose -f docker-compose.compact.yml up -d)
pytest tests/integration/test_course_project.py -v

# Only static compliance checks (no services needed, always runs)
pytest tests/integration/test_course_project.py::TestCourseCompliance -v

# Skip all integration tests
pytest -m 'not integration'

# Only gRPC tests
pytest tests/integration/test_course_project.py::TestGrpcPredictor -v

# Only Kafka tests
pytest tests/integration/test_course_project.py::TestKafka -v
```

## Environment variables (override defaults)

| Variable | Default | Purpose |
|----------|---------|---------|
| `GRPC_HOST` | `localhost` | gRPC predictor host |
| `GRPC_PORT` | `50051` | gRPC predictor port |
| `KAFKA_BROKER_EXTERNAL` | `localhost:19092` | Kafka/Redpanda broker (external listener) |
| `MLFLOW_URL` | `http://localhost:5001` | MLflow tracking server |
| `SIGNALBRIDGE_URL` | `http://localhost:8085` | SignalBridge API base URL |

## Dependencies

```bash
pip install grpcio grpcio-tools kafka-python pyyaml
```

The gRPC stubs (`predictor_pb2.py`, `predictor_pb2_grpc.py`) are auto-generated
when the `grpc-predictor` Docker image builds. Tests skip gracefully if they are
absent.
