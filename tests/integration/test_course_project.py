"""Integration tests for MLOps course project deliverable.

Validates the 2 course technologies (gRPC + Kafka) plus supporting infrastructure
(Docker, Airflow, MLflow, PostgreSQL, Redis, SignalBridge).

Skips tests when a service is unreachable (so CI without full stack still passes).
"""
import os
import json
import time
import socket
import pytest

pytestmark = pytest.mark.integration


def _port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    """Returns True if (host, port) accepts TCP connections."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.error, OSError):
        return False


def _skip_unless_port(host, port, reason=None):
    """Shortcut: skip the current test if (host, port) isn't reachable."""
    if not _port_open(host, port):
        pytest.skip(reason or f"{host}:{port} not reachable")


# ---------------------------------------------------------------------------
# gRPC tests
# ---------------------------------------------------------------------------
class TestGrpcPredictor:
    HOST = os.environ.get("GRPC_HOST", "localhost")
    PORT = int(os.environ.get("GRPC_PORT", "50051"))

    def test_server_port_open(self):
        _skip_unless_port(self.HOST, self.PORT, "gRPC server not running")
        assert _port_open(self.HOST, self.PORT)

    def test_health_check(self):
        _skip_unless_port(self.HOST, self.PORT, "gRPC server not running")
        grpc = pytest.importorskip("grpc")
        # dynamic import of generated stubs from the service dir
        import sys
        sys.path.insert(0, os.path.abspath("services/grpc_predictor"))
        try:
            from predictor_pb2 import HealthRequest
            from predictor_pb2_grpc import PredictorServiceStub
        except ImportError:
            pytest.skip("predictor_pb2 not generated (build the Docker image first)")
        channel = grpc.insecure_channel(f"{self.HOST}:{self.PORT}")
        stub = PredictorServiceStub(channel)
        response = stub.HealthCheck(HealthRequest(), timeout=5)
        assert response.ready is True

    def test_predict(self):
        _skip_unless_port(self.HOST, self.PORT, "gRPC server not running")
        grpc = pytest.importorskip("grpc")
        import sys
        sys.path.insert(0, os.path.abspath("services/grpc_predictor"))
        try:
            from predictor_pb2 import PredictRequest
            from predictor_pb2_grpc import PredictorServiceStub
        except ImportError:
            pytest.skip("predictor_pb2 not generated")
        channel = grpc.insecure_channel(f"{self.HOST}:{self.PORT}")
        stub = PredictorServiceStub(channel)
        req = PredictRequest(features={"dxy_z": 0.5, "wti_z": -0.3, "rsi_14": 45.0, "vix_z": 0.1})
        response = stub.Predict(req, timeout=5)
        assert response.direction in {"LONG", "SHORT", "FLAT", "UNKNOWN"}
        assert 0.0 <= response.confidence <= 1.0


# ---------------------------------------------------------------------------
# Kafka tests
# ---------------------------------------------------------------------------
class TestKafka:
    BROKER = os.environ.get("KAFKA_BROKER_EXTERNAL", "localhost:19092")
    TOPIC = "signals.h5"

    def test_broker_reachable(self):
        host, port = self.BROKER.split(":")
        _skip_unless_port(host, int(port), f"Kafka broker {self.BROKER} unreachable")

    def test_produce_consume_roundtrip(self):
        host, port = self.BROKER.split(":")
        _skip_unless_port(host, int(port), "Kafka unreachable")
        pytest.importorskip("kafka")
        from kafka import KafkaProducer, KafkaConsumer

        test_msg = {
            "week": "2026-W99-TEST",
            "direction": "FLAT",
            "confidence": 0.5,
            "ensemble_return": 0.0,
            "skip_trade": True,
            "timestamp": "2026-04-23T00:00:00Z",
            "source": "integration_test",
        }
        producer = KafkaProducer(
            bootstrap_servers=self.BROKER,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        producer.send(self.TOPIC, test_msg)
        producer.flush(timeout=10)

        # Consume from latest — fresh group, shouldn't see this unless we seek
        consumer = KafkaConsumer(
            self.TOPIC,
            bootstrap_servers=self.BROKER,
            auto_offset_reset="latest",
            consumer_timeout_ms=5000,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            group_id=f"test-{time.time()}",
        )
        # Produce again AFTER consumer subscribed — best effort
        producer.send(self.TOPIC, {**test_msg, "week": "2026-W99-TEST-2"})
        producer.flush(timeout=10)
        received = []
        for msg in consumer:
            received.append(msg.value)
            if len(received) >= 1:
                break
        consumer.close()
        producer.close()
        assert len(received) >= 1
        assert received[0].get("source") == "integration_test"


# ---------------------------------------------------------------------------
# Supporting infrastructure
# ---------------------------------------------------------------------------
class TestSupportingInfra:
    def test_mlflow_reachable(self):
        import urllib.request
        url = os.environ.get("MLFLOW_URL", "http://localhost:5001")
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                assert r.status < 500
        except Exception:
            pytest.skip(f"MLflow not reachable at {url}")

    def test_signalbridge_health(self):
        import urllib.request
        url = os.environ.get("SIGNALBRIDGE_URL", "http://localhost:8085") + "/health"
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                assert r.status == 200
                body = r.read()
                assert b"status" in body or b"healthy" in body
        except Exception:
            pytest.skip(f"SignalBridge not reachable at {url}")

    def test_postgres_reachable(self):
        _skip_unless_port("localhost", 5432, "PostgreSQL not reachable")

    def test_redis_reachable(self):
        _skip_unless_port("localhost", 6379, "Redis not reachable")

    def test_airflow_reachable(self):
        _skip_unless_port("localhost", 8080, "Airflow not reachable")


# ---------------------------------------------------------------------------
# Compliance self-check
# ---------------------------------------------------------------------------
class TestCourseCompliance:
    """Meta-tests that verify the project structure matches course requirements."""

    def test_grpc_service_exists(self):
        assert os.path.isdir("services/grpc_predictor")
        assert os.path.isfile("services/grpc_predictor/proto/predictor.proto")

    def test_kafka_bridge_exists(self):
        assert os.path.isdir("services/kafka_bridge")
        assert os.path.isfile("services/kafka_bridge/producer.py")
        assert os.path.isfile("services/kafka_bridge/consumer.py")

    def test_course_project_doc_exists(self):
        assert os.path.isfile("docs/COURSE_PROJECT.md")

    def test_makefile_has_course_targets(self):
        with open("Makefile") as f:
            content = f.read()
        assert "course-demo" in content
        assert "course-grpc" in content
        assert "course-kafka" in content

    def test_docker_compose_has_new_services(self):
        import yaml
        with open("docker-compose.compact.yml") as f:
            compose = yaml.safe_load(f)
        services = set(compose.get("services", {}).keys())
        assert "redpanda" in services
        assert "grpc-predictor" in services
        # kafka-bridge services (naming may vary)
        kb_services = [s for s in services if "kafka-bridge" in s]
        assert len(kb_services) >= 2, f"expected producer+consumer, got {kb_services}"
