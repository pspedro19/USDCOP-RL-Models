"""Regresion del kafka_bridge producer (bug: SELECT pedia columna `week` inexistente
en forecast_h5_signals -> error Postgres cada 60s y bridge sin emitir jamas)."""
from __future__ import annotations

import importlib.util
import re
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PRODUCER_PATH = REPO / "services" / "kafka_bridge" / "producer.py"
DDL_PATH = REPO / "database" / "migrations" / "050_consolidated_h5_ddl.sql"


def _load_producer():
    # kafka-python no esta en el entorno de test; stub minimo para importar el modulo
    if "kafka" not in sys.modules:
        kafka_stub = types.ModuleType("kafka")
        kafka_stub.KafkaProducer = object
        errors_stub = types.ModuleType("kafka.errors")
        errors_stub.KafkaError = Exception
        errors_stub.NoBrokersAvailable = Exception
        sys.modules["kafka"] = kafka_stub
        sys.modules["kafka.errors"] = errors_stub
    spec = importlib.util.spec_from_file_location("kafka_bridge_producer", PRODUCER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ddl_columns() -> set[str]:
    ddl = DDL_PATH.read_text(encoding="utf-8")
    m = re.search(
        r"CREATE TABLE IF NOT EXISTS forecast_h5_signals\s*\((.*?)\n\);",
        ddl,
        re.DOTALL,
    )
    assert m, "forecast_h5_signals no encontrada en 050_consolidated_h5_ddl.sql"
    cols = set()
    for line in m.group(1).splitlines():
        token = line.strip().split(" ")[0].strip('",')
        if token and not token.upper() in {"CONSTRAINT", "UNIQUE", "PRIMARY", "FOREIGN", "CHECK"}:
            cols.add(token)
    return cols


def test_select_columns_exist_in_schema():
    mod = _load_producer()
    sql = mod.SELECT_LATEST_SQL
    body = sql.split("SELECT", 1)[1].split("FROM", 1)[0]
    selected = [c.strip().rstrip(",") for c in body.splitlines() if c.strip().rstrip(",")]
    ddl_cols = _ddl_columns()
    missing = [c for c in selected if c not in ddl_cols]
    assert not missing, f"columnas del SELECT ausentes en el esquema: {missing}"


def test_select_does_not_reference_bare_week():
    mod = _load_producer()
    assert not re.search(r"^\s+week,?\s*$", mod.SELECT_LATEST_SQL, re.MULTILINE)
    assert "inference_week" in mod.SELECT_LATEST_SQL
    assert "inference_year" in mod.SELECT_LATEST_SQL


def test_row_to_message_contract():
    mod = _load_producer()
    row = {
        "inference_year": 2026,
        "inference_week": 17,
        "direction": -1,
        "confidence_tier": "HIGH",
        "ensemble_return": -0.012,
        "skip_trade": False,
        "hard_stop_pct": 2.8,
        "take_profit_pct": 1.4,
        "adjusted_leverage": 1.5,
        "created_at": None,
    }
    msg = mod.row_to_message(row)
    assert msg["week"] == "2026-W17"          # contrato README: "2026-W17"
    assert msg["direction"] == "SHORT"         # contrato README: SHORT/LONG (DB: -1/1)
    assert msg["skip_trade"] is False

    # HONESTIDAD (red-team #5): confidence_tier es VARCHAR ('HIGH'/'MEDIUM'/'LOW'),
    # asi que float(tier) no existe y `confidence` ES None en TODO mensaje real.
    # No se inventa un numero: la fuente numerica esta pendiente de decision de
    # contrato (candidatos: confidence_agreement / confidence_magnitude — C-006).
    assert msg["confidence"] is None
    # Aditivo: el tier textual SI se emite, para no ocultar la informacion disponible.
    assert msg["confidence_tier"] == "HIGH"

    row["direction"] = 1
    assert mod.row_to_message(row)["direction"] == "LONG"

    row["direction"] = 0
    assert mod.row_to_message(row)["direction"] == "FLAT"


def test_row_to_message_missing_fields_safe():
    mod = _load_producer()
    msg = mod.row_to_message({})
    assert msg["week"] == ""
    assert msg["confidence"] is None  # confidence_tier es VARCHAR: fuente numerica pendiente de decision (C-006)
    assert msg["confidence_tier"] is None


class _FakeProducer:
    def flush(self, timeout=None):
        pass


def test_run_loop_cursor_does_not_skip_failed_row(monkeypatch):
    """At-least-once (hallazgo colateral red-team): si publish falla en la fila 2
    de 3, el cursor NO puede avanzar mas alla de la fila 1 — avanzar con la fila 3
    perderia la fila 2 para siempre (el SELECT es WHERE id > last_id)."""
    mod = _load_producer()

    rows = [
        {"id": 1, "inference_year": 2026, "inference_week": 17, "direction": 1},
        {"id": 2, "inference_year": 2026, "inference_week": 18, "direction": -1},
        {"id": 3, "inference_year": 2026, "inference_week": 19, "direction": 1},
    ]

    published: list[str] = []
    saved: list[int] = []

    def fake_publish(producer, payload):
        published.append(payload["week"])
        return payload["week"] != "2026-W18"  # la fila 2 falla

    def fake_sleep(seconds):
        mod._shutdown = True  # una sola pasada del while

    monkeypatch.setattr(mod, "fetch_new_signals", lambda last_id: rows)
    monkeypatch.setattr(mod, "publish", fake_publish)
    monkeypatch.setattr(mod, "load_last_published_id", lambda: 0)
    monkeypatch.setattr(mod, "save_last_published_id", lambda i: saved.append(i))
    monkeypatch.setattr(mod, "_sleep_interruptible", fake_sleep)
    mod._shutdown = False
    try:
        mod.run_loop(_FakeProducer())
    finally:
        mod._shutdown = False

    # El cursor queda en la ultima fila publicada CON EXITO (la 1), nunca en la 3.
    assert saved == [1], f"cursor avanzo mal (fila 2 perdida): saved={saved}"
    # Y tras el fallo se corta el batch: la fila 3 no se intenta este tick.
    assert published == ["2026-W17", "2026-W18"], f"batch no se corto: {published}"
