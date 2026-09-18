"""Prospective LLM ablations with immutable inputs and a shared, durable cost cap.

This is a NEW experiment; no historical prompt, model, or ledger is rewritten.
SQLite is the authoritative append-only decision/budget ledger. A started request
without a committed outcome is deliberately NOT retried on resume: its billing
and response are unknown. Network clients have no hidden retries.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlencode, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from collections.abc import Callable

VARIANTS = ("L0", "L1", "L2")
PROVIDERS = ("deepseek", "azure_openai")
AUTHORITY_MICRO_USD = 100_000_000
COHORT_SESSIONS = 20
BARS = 59
SNAPSHOT_FIELDS = ("session_date", "bar", "cutoff_utc", "context_created_at_utc", "market", "daily_features",
                   "daily_available_at_utc", "close", "cost_context")
SYSTEM_PROMPT = """Eres un analista cuantitativo de USD/COP. Responde exclusivamente JSON con las claves:
direccion (short, flat o long), tamano (0, 0.5 o 1) y confianza (0 a 1).
Usa únicamente el contexto observado hasta la barra indicada; no inventes noticias ni datos futuros.
Si no hay una ventaja clara, responde flat con tamano 0."""


class PilotBlocked(ValueError):
    """A scientific, temporal, or spending precondition was not satisfied."""


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    safe_path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_path(path: Path) -> Path:
    """Deny sensitive names before reading, hashing, or creating any artifact."""
    resolved = Path(path).resolve()
    for part in resolved.parts:
        lower = part.lower()
        if (lower == "secrets" or lower.startswith(".env") or lower.endswith((".pem", ".key"))
                or (lower.startswith(("credentials", "service-account")) and lower.endswith(".json"))):
            raise PilotBlocked("sensitive path is forbidden")
    return resolved


def allowed_url(value: str, *, provider: str | None = None, pricing: bool = False) -> str:
    if provider not in PROVIDERS:
        raise PilotBlocked("recognized billing provider required")
    if not isinstance(value, str) or value != value.strip() or any(c.isspace() for c in value):
        raise PilotBlocked("provider URL must be explicit text without whitespace")
    parsed = urlparse(value)
    if (parsed.scheme != "https" or parsed.username or parsed.password or parsed.query or parsed.fragment
            or not parsed.hostname or parsed.port not in (None, 443)):
        raise PilotBlocked("URL must use TLS without credentials, query, fragment, or nonstandard port")
    host = parsed.hostname.lower()
    if pricing:
        hosts = ({"api-docs.deepseek.com", "api.deepseek.com"} if provider == "deepseek"
                 else {"azure.microsoft.com", "learn.microsoft.com"})
        if host not in hosts:
            raise PilotBlocked("pricing evidence domain must match the billing provider")
    elif (provider == "deepseek" and host != "api.deepseek.com") or (
            provider == "azure_openai" and not host.endswith(".openai.azure.com")):
        raise PilotBlocked("provider endpoint is not an approved provider host")
    elif parsed.path not in ("", "/"):
        raise PilotBlocked("endpoint must be a provider origin, not an arbitrary URL path")
    return value


def instant(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise PilotBlocked("timezone required")
    return parsed.astimezone(UTC)


def utc_now() -> datetime:
    return datetime.now(UTC)


def positive(value: object, label: str, *, zero: bool = False) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0 or (not zero and number == 0):
        raise PilotBlocked(f"invalid {label}")
    return number


def _hash(value: object, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise PilotBlocked(f"missing/invalid {label} SHA256")
    return value


def _count(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PilotBlocked(f"{label} must be an integer >= {minimum}")
    return value


def _model_identifier(value: object, label: str) -> str:
    if (not isinstance(value, str) or not value or not value.isprintable()
            or any(c.isspace() for c in value)):
        raise PilotBlocked(f"{label} must be explicit nonblank model metadata")
    return value


def _tariff(value: object) -> Fraction:
    """Exact declared decimal, not a float-rounded or authenticated vendor quote."""
    if type(value) not in (str, int, float, Decimal):
        raise PilotBlocked("tariff must be a finite positive decimal")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise PilotBlocked("tariff must be a finite positive decimal") from None
    if not number.is_finite() or number <= 0:
        raise PilotBlocked("tariff must be a finite positive decimal")
    return Fraction(number)


def validate_dictionary(dictionary: dict) -> None:
    """Every named input must declare semantics, native units, and representation."""
    features = dictionary.get("features", [])
    if not features or len({f["name"] for f in features}) != len(features):
        raise PilotBlocked("dictionary missing or duplicate feature")
    for field in features:
        if not field.get("meaning") or not field.get("native_unit"):
            raise PilotBlocked("feature semantics and native units required")
        if field.get("representation") == "zscore_clipped":
            positive(field.get("std"), "scaler std")
            if not math.isfinite(float(field.get("mean"))):
                raise PilotBlocked("nonfinite scaler mean")
            if field.get("clip") != [-5, 5]:
                raise PilotBlocked("frozen clipping must be declared as [-5, 5]")
        elif field.get("representation") not in {"native", "probability"}:
            raise PilotBlocked("unknown feature representation")


def bind_dictionary_to_scaler(dictionary: dict, artifacts: dict[str, Path]) -> tuple[list, list]:
    schema = json.loads(safe_path(artifacts["schema"]).read_text(encoding="utf-8"))
    scaler = json.loads(safe_path(artifacts["scaler"]).read_text(encoding="utf-8"))
    market_order = scaler["features"]
    macro_order, regimes = schema["groups"]["macro"], schema["groups"]["regimen"]
    fields = {f["name"]: f for f in dictionary["features"]}
    if set(fields) != set(market_order + macro_order + regimes):
        raise PilotBlocked("dictionary does not match frozen scaler/schema feature names")
    means = scaler["mean"] + scaler.get("macro_mean", [])
    scales = scaler["scale"] + scaler.get("macro_scale", [])
    for name, mean, std in zip(market_order + macro_order, means, scales, strict=True):
        field = fields[name]
        if (field["representation"] != "zscore_clipped" or float(field["mean"]) != float(mean)
                or float(field["std"]) != float(std)):
            raise PilotBlocked("dictionary scale/mean differs from frozen scaler")
    if any(fields[name]["representation"] != "probability" for name in regimes):
        raise PilotBlocked("HMM posterior representation must remain probability")
    return market_order, macro_order + regimes


def quote_cost_micro(price: dict, input_tokens: int, output_tokens: int) -> int:
    # Rates are USD per million tokens. One rate * tokens is micro-USD.
    # Fraction keeps all declared decimal digits; Decimal arithmetic can round
    # before the ceiling when the active context precision is insufficient.
    inputs = _count(input_tokens, "input tokens")
    outputs = _count(output_tokens, "output tokens")
    amount = (_tariff(price["input_usd_per_million"]) * inputs
              + _tariff(price["output_usd_per_million"]) * outputs)
    return math.ceil(amount)


def prepare_manifest(config: dict, calendar: dict, dictionary: dict,
                     artifacts: dict[str, Path], *, now: datetime) -> dict:
    """Validate and materialize a freeze, without network or credential access.

    Calendar is an ex-ante exchange/session-calendar snapshot, NOT a mask selected
    using future price quality or PnL. Bad future sessions remain in the ledger.
    """
    if config.get("status") != "OPERATOR_APPROVED":
        raise PilotBlocked("operator approval / completed configuration required")
    if config.get("variants") != list(VARIANTS) or config.get("providers") is None:
        raise PilotBlocked("freeze all three ablations and both providers")
    if set(config["providers"]) != set(PROVIDERS):
        raise PilotBlocked("both providers required")
    if (_count(config.get("cohort_sessions"), "cohort sessions", minimum=1) != COHORT_SESSIONS
            or _count(config.get("bars_per_session"), "bars per session", minimum=1) != BARS):
        raise PilotBlocked("pilot is exactly 20 scheduled sessions x 59 decisions")
    if Decimal(str(config.get("budget_usd"))) != Decimal("100"):
        raise PilotBlocked("shared operator authority is USD 100")
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise PilotBlocked("freeze timezone required; host timezone cannot be inferred")
    freeze = now.astimezone(UTC)
    freeze_day = freeze.astimezone(ZoneInfo("America/Bogota")).date()
    validate_dictionary(dictionary)
    sessions = calendar.get("sessions", [])
    if not calendar.get("source") or instant(calendar["published_at_utc"]) > freeze:
        raise PilotBlocked("calendar must be sourced and available before freeze")
    dates = [date.fromisoformat(row["session_date"]) for row in sessions]
    if dates != sorted(set(dates)):
        raise PilotBlocked("calendar dates must be unique and sorted")
    for row in sessions:
        start, end = instant(row["open_utc"]), instant(row["close_utc"])
        if start >= end or start.astimezone(ZoneInfo("America/Bogota")).date().isoformat() != row["session_date"]:
            raise PilotBlocked("calendar session boundaries invalid")
    cohort = [r for r in sessions if date.fromisoformat(r["session_date"]) > freeze_day][:COHORT_SESSIONS]
    if len(cohort) != COHORT_SESSIONS:
        raise PilotBlocked("calendar needs first 20 future eligible sessions")
    sampling = config["sampling"]
    _count(sampling.get("max_tokens"), "output token cap", minimum=1)
    _count(sampling.get("max_retries"), "retry count")
    if sampling != {"temperature": 0.1, "top_p": 0.9, "max_tokens": 256, "max_retries": 1}:
        raise PilotBlocked("sampling/retry contract differs from frozen pilot")
    cap = _count(config["max_input_tokens"], "input token admission cap", minimum=1)
    allocation, rates = 0, {}
    for provider in PROVIDERS:
        spec = config["providers"][provider]
        if not all(spec.get(k) for k in ("requested_model", "expected_served_model", "pricing_model", "api_version")):
            raise PilotBlocked("requested deployment, expected served snapshot, pricing base-model and API version required")
        for field in ("requested_model", "expected_served_model", "pricing_model", "api_version"):
            _model_identifier(spec[field], field)
        allowed_url(spec.get("endpoint", ""), provider=provider)
        price = spec.get("pricing")
        if not price or not price.get("source_url") or not price.get("evidence_sha256"):
            raise PilotBlocked("current verifiable pricing snapshot required for both providers")
        _hash(price["evidence_sha256"], "pricing evidence")
        allowed_url(price["source_url"], provider=provider, pricing=True)
        evidence = artifacts.get("pricing_" + provider)
        if evidence is None or file_digest(evidence) != price["evidence_sha256"]:
            raise PilotBlocked("tariff evidence must match a frozen pricing_PROVIDER artifact")
        if price.get("model") != spec["pricing_model"]:
            raise PilotBlocked("price/model binding mismatch")
        if not instant(price["observed_at_utc"]) <= freeze <= instant(price["valid_until_utc"]):
            raise PilotBlocked("pricing snapshot is unavailable or stale")
        # Require an upper bound without cache discounts or unverified free tiers.
        rates[provider] = quote_cost_micro(price, cap, sampling["max_tokens"])
        allocation += rates[provider] * COHORT_SESSIONS * BARS * len(VARIANTS) * (sampling["max_retries"] + 1)
    if allocation > AUTHORITY_MICRO_USD:
        raise PilotBlocked(f"whole paired cohort incl retry reserves requires {allocation / 1e6:.6f} USD; cap=100")
    if not artifacts or not {"dataset", "schema", "scaler", "cost_contract"}.issubset(artifacts):
        raise PilotBlocked("dataset/schema/scaler/cost artifacts required")
    market_order, daily_order = bind_dictionary_to_scaler(dictionary, artifacts)
    frozen_artifacts = {key: {"path": str(path.resolve()), "sha256": file_digest(path)}
                        for key, path in artifacts.items()}
    manifest = {
        "contract": "CTR-RESEARCH-LLM-PILOT-002", "config": config,
        "freeze_utc": freeze.isoformat(), "cohort": cohort,
        "cohort_policy": "first_20_scheduled_sessions_after_freeze_local_date_no_quality_selection",
        "calendar_sha256": digest(calendar), "dictionary": dictionary,
        "dictionary_sha256": digest(dictionary), "artifacts": frozen_artifacts,
        "market_feature_order": market_order, "daily_feature_order": daily_order,
        "legacy_system_prompt": SYSTEM_PROMPT,
        "legacy_system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "implementation_sha256": file_digest(Path(__file__)),
        "scope": "prospective_pilot_not_confirmatory", "allocation_micro_usd": allocation,
        "per_attempt_reserve_micro_usd": rates,
    }
    manifest["manifest_sha256"] = digest(manifest)
    return manifest


def verify_manifest(manifest: dict, *, verify_files: bool = True) -> None:
    payload = dict(manifest)
    expected = payload.pop("manifest_sha256", None)
    if digest(payload) != expected:
        raise PilotBlocked("immutable manifest digest mismatch")
    if verify_files:
        for artifact in manifest["artifacts"].values():
            if file_digest(Path(artifact["path"])) != artifact["sha256"]:
                raise PilotBlocked("frozen input file changed")
        if file_digest(Path(__file__)) != manifest["implementation_sha256"]:
            raise PilotBlocked("implementation changed after freeze")


def freeze_file(path: Path, manifest: dict) -> None:
    verify_manifest(manifest)
    safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical(manifest) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class PilotStore:
    """One SQLite ledger shared by all providers/processes and pilot manifests.

    Cohort worst-case funds remain reserved even when actual billed usage is lower.
    Unknown billing never releases funds. This conservative choice makes crashes
    safe and prevents provider/variant-specific budget selection.
    """

    def __init__(self, path: Path):
        self.path = safe_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS cohorts (
                    manifest_hash TEXT PRIMARY KEY, allocation INTEGER NOT NULL,
                    manifest TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS calls (
                    call_id TEXT PRIMARY KEY, manifest_hash TEXT NOT NULL,
                    reserve INTEGER NOT NULL, status TEXT NOT NULL, outcome TEXT);
                CREATE TABLE IF NOT EXISTS decisions (
                    decision_id TEXT PRIMARY KEY, manifest_hash TEXT NOT NULL,
                    provider TEXT NOT NULL, variant TEXT NOT NULL,
                    session_date TEXT NOT NULL, bar INTEGER NOT NULL,
                    record TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS controls (name TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS served_models (
                    manifest_hash TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL,
                    PRIMARY KEY (manifest_hash,provider));
            """)

    def connect(self):
        db = sqlite3.connect(self.path, timeout=20, isolation_level=None)
        db.execute("PRAGMA busy_timeout=20000")
        db.execute("PRAGMA synchronous=FULL")
        return db

    def admit(self, manifest: dict) -> None:
        verify_manifest(manifest)
        key, amount = manifest["manifest_sha256"], manifest["allocation_micro_usd"]
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._not_halted(db)
            prior = db.execute("SELECT manifest FROM cohorts WHERE manifest_hash=?", (key,)).fetchone()
            if prior:
                if prior[0] != canonical(manifest):
                    raise PilotBlocked("manifest collision")
                db.commit()
                return
            reserved = db.execute("SELECT COALESCE(SUM(allocation),0) FROM cohorts").fetchone()[0]
            if reserved + amount > AUTHORITY_MICRO_USD:
                raise PilotBlocked("shared USD 100 budget exhausted by admitted cohorts")
            db.execute("INSERT INTO cohorts VALUES (?,?,?)", (key, amount, canonical(manifest)))
            db.commit()

    @staticmethod
    def _not_halted(db) -> None:
        if db.execute("SELECT 1 FROM controls WHERE name='global_halt'").fetchone():
            raise PilotBlocked("shared pilot is HALTED; billing/model provenance needs operator reconciliation")

    def halt(self, reason: str) -> None:
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO controls VALUES ('global_halt',?)", (reason,))

    def check_served_model(self, manifest: dict, provider: str, model: str) -> None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._not_halted(db)
            args = (manifest["manifest_sha256"], provider)
            row = db.execute("SELECT model FROM served_models WHERE manifest_hash=? AND provider=?", args).fetchone()
            if row and row[0] != model:
                db.execute("INSERT OR IGNORE INTO controls VALUES ('global_halt','served_model_drift')")
                db.commit()
                raise PilotBlocked("served model changed within frozen cohort")
            if not row:
                db.execute("INSERT INTO served_models VALUES (?,?,?)", (*args, model))
            db.commit()

    def records(self, manifest: dict) -> list[dict]:
        with self.connect() as db:
            return [json.loads(row[0]) for row in db.execute(
                "SELECT record FROM decisions WHERE manifest_hash=? ORDER BY session_date,bar,provider,variant",
                (manifest["manifest_sha256"],))]

    def decision(self, decision_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT record FROM decisions WHERE decision_id=?", (decision_id,)).fetchone()
            return json.loads(row[0]) if row else None

    def reserve_attempt(self, manifest: dict, call_id: str, provider: str) -> None:
        key = manifest["manifest_sha256"]
        amount = manifest["per_attempt_reserve_micro_usd"][provider]
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._not_halted(db)
            if db.execute("SELECT 1 FROM calls WHERE call_id=?", (call_id,)).fetchone():
                raise PilotBlocked("attempt already started; uncertain requests are never silently reissued")
            cohort = db.execute("SELECT allocation FROM cohorts WHERE manifest_hash=?", (key,)).fetchone()
            if not cohort:
                raise PilotBlocked("cohort not admitted to shared budget")
            used = db.execute("SELECT COALESCE(SUM(reserve),0) FROM calls WHERE manifest_hash=?", (key,)).fetchone()[0]
            if used + amount > cohort[0]:
                raise PilotBlocked("cohort retry reserve exhausted")
            db.execute("INSERT INTO calls VALUES (?,?,?,'started',NULL)", (call_id, key, amount))
            db.commit()

    def outcome(self, call_id: str, payload: dict) -> None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            changed = db.execute("UPDATE calls SET status='completed',outcome=? WHERE call_id=? AND status='started'",
                                 (canonical(payload), call_id)).rowcount
            if changed != 1:
                raise PilotBlocked("attempt outcome already sealed or missing")
            db.commit()

    def append_decision(self, record: dict) -> None:
        with self.connect() as db:
            db.execute("INSERT INTO decisions VALUES (?,?,?,?,?,?,?)", (
                record["decision_id"], record["manifest_sha256"], record["provider"],
                record["variant"], record["session_date"], record["bar"], canonical(record)))


def render_legacy(context: dict) -> str:
    """Exact historical user-template, now fed by an explicitly causal envelope."""
    rows = [f"barra={r['bar']}: " + ", ".join(f"{name}={float(value):.8g}" for name, value in r["features"].items())
            for r in context["market"]]
    daily = ", ".join(f"{name}={float(value):.8g}" for name, value in context["daily_features"].items())
    return (f"Sesión {context['session_date']}, barra de decisión {context['bar']}/58.\n"
            f"Contexto diario causal (disponible antes de la sesión): {daily}\n"
            "No hay documentos de noticias en este export; no asumas ninguno.\n"
            "Observaciones de mercado hasta ahora:\n" + "\n".join(rows))


def validate_context(context: dict, manifest: dict, now: datetime) -> None:
    cohort = {r["session_date"]: r for r in manifest["cohort"]}
    session = context.get("session_date")
    if context.get("retrospective") is not False or session not in cohort:
        raise PilotBlocked("only prospective frozen-cohort contexts allowed")
    bar = context.get("bar")
    if type(bar) is not int or not 0 <= bar < BARS:
        raise PilotBlocked("bar must be 0..58")
    cutoff, deadline = instant(context["cutoff_utc"]), instant(context["decision_deadline_utc"])
    created = instant(context["context_created_at_utc"])
    opened, closed = instant(cohort[session]["open_utc"]), instant(cohort[session]["close_utc"])
    if not instant(manifest["freeze_utc"]) < opened <= cutoff <= now < deadline <= closed:
        raise PilotBlocked("retrospective, future bar, or expired decision deadline")
    if cutoff != opened + timedelta(minutes=5 * (bar + 1)) or deadline != cutoff + timedelta(minutes=5):
        raise PilotBlocked("cutoff/deadline must follow exact session-open plus bar-close five-minute grid")
    if not cutoff <= created <= now:
        raise PilotBlocked("context must be assembled after observed cutoff and before request")
    if context.get("dataset_sha256") != manifest["artifacts"]["dataset"]["sha256"]:
        raise PilotBlocked("dataset binding mismatch")
    _hash(context.get("snapshot_sha256"), "live snapshot")
    snapshot_path = safe_path(Path(context["snapshot_path"]))
    if file_digest(snapshot_path) != context["snapshot_sha256"]:
        raise PilotBlocked("live snapshot content hash mismatch")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if snapshot != {key: context[key] for key in SNAPSHOT_FIELDS}:
        raise PilotBlocked("live snapshot content differs from received context")
    observed = list(context["market"])
    if [r["bar"] for r in observed] != list(range(max(0, bar - 23), bar + 1)):
        raise PilotBlocked("exact trailing up-to-24 observed bars required")
    names = set(context["daily_features"])
    if list(context["daily_features"]) != manifest["daily_feature_order"]:
        raise PilotBlocked("daily feature order differs from frozen template")
    for row in observed:
        if instant(row["observed_at_utc"]) != opened + timedelta(minutes=5 * (row["bar"] + 1)):
            raise PilotBlocked("observed bar-close timestamps are off grid")
        if not instant(row["observed_at_utc"]) <= instant(row["received_at_utc"]) <= created:
            raise PilotBlocked("bar observation/receipt exceeds context assembly time")
        if list(row["features"]) != manifest["market_feature_order"]:
            raise PilotBlocked("per-bar feature order differs from frozen template")
        names.update(row["features"])
        if not all(math.isfinite(float(v)) for v in row["features"].values()):
            raise PilotBlocked("nonfinite feature")
    if names != {f["name"] for f in manifest["dictionary"]["features"]}:
        raise PilotBlocked("dictionary does not cover exactly the presented features")
    if not all(math.isfinite(float(v)) for v in context["daily_features"].values()):
        raise PilotBlocked("nonfinite daily context")
    if instant(context["daily_available_at_utc"]) > min(opened, cutoff):
        raise PilotBlocked("daily context was not available before session open")
    positive(context["close"], "close")
    if context["cost_context"].get("unit") != "decimal_return_per_abs_delta_weight":
        raise PilotBlocked("explicit cost units required")
    positive(context["cost_context"]["one_way_return_per_unit"], "cost", zero=True)
    if not context["cost_context"].get("source"):
        raise PilotBlocked("declared cost source required (assumed is not measured)")
    if context["cost_context"].get("contract_sha256") != manifest["artifacts"]["cost_contract"]["sha256"]:
        raise PilotBlocked("cost contract binding mismatch")


def state_at(context: dict, prior: dict | None) -> dict:
    if context["bar"] == 0:
        if prior is not None:
            raise PilotBlocked("first bar cannot inherit yesterday's exposure")
        return {"position": 0.0, "session_pnl_decimal": 0.0, "bars_in_position": 0,
                "entry_price": None, "unrealized_pnl_decimal": 0.0}
    if prior is None or prior["bar"] != context["bar"] - 1:
        raise PilotBlocked("missing immediate prior decision; no state reset or bar skipping")
    if instant(prior["cutoff_utc"]) >= instant(context["cutoff_utc"]):
        raise PilotBlocked("nonmonotone cutoff")
    weight = prior["weight"]
    pnl = prior["state_after"]["session_pnl_decimal"] + weight * (context["close"] / prior["close"] - 1)
    entry = prior["state_after"]["entry_price"]
    return {"position": weight, "session_pnl_decimal": pnl,
            "bars_in_position": prior["state_after"]["bars_in_position"] + (1 if weight else 0),
            "entry_price": entry,
            "unrealized_pnl_decimal": weight * (context["close"] / entry - 1) if entry else 0.0}


def prompts(variant: str, context: dict, manifest: dict, state: dict) -> tuple[str, str]:
    if variant not in VARIANTS:
        raise PilotBlocked("unknown ablation")
    text = render_legacy(context)
    if variant in {"L1", "L2"}:
        text += ("\nDiccionario congelado: los z-scores no son magnitudes económicas nativas. "
                 "z=clip((x-media_desarrollo)/std_desarrollo,-5,5); negativos de volatilidad "
                 "normalizada no significan volatilidad negativa. Probabilidades HMM no se "
                 "estandarizan. No se modificó ningún valor del contexto anterior.\n"
                 + canonical(manifest["dictionary"]))
    if variant == "L2":
        text += "\nEstado PROPIO antes de esta decisión: " + canonical(state)
        text += "\nCostos declarados: " + canonical(context["cost_context"])
        text += "\nEl coste del cambio será |peso_nuevo-peso_previo| por el coste unitario; cierre terminal también se cobra."
    return manifest["legacy_system_prompt"], text


def sanitize(text: str) -> str:
    text = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[REDACTED_SECRET]", text)
    return re.sub(r"(?i)bearer\s+[A-Za-z0-9_.-]+", "Bearer [REDACTED_SECRET]", text)


def parse_response(text: str) -> tuple[bool, float, dict]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        value = json.loads(text)
        if not isinstance(value, dict) or set(value) != {"direccion", "tamano", "confianza"}:
            return False, 0.0, {}
        size, confidence = float(value["tamano"]), float(value["confianza"])
        sign = {"short": -1, "flat": 0, "long": 1}[value["direccion"]]
        valid = size in (0, 0.5, 1) and 0 <= confidence <= 1 and (sign != 0 or size == 0)
        return bool(valid), sign * size, value
    except (ValueError, TypeError, KeyError):
        return False, 0.0, {}


class ExplicitChatTransport:
    """No dotenv, implicit model, provider fallback, SDK retry, or hidden sampling.

    Official Chat Completions schema: https://developers.openai.com/api/reference/resources/chat
    API credentials are read from inherited process environment only on live call.
    """

    def generate(self, provider: str, spec: dict, request: dict) -> dict:
        allowed_url(spec["endpoint"], provider=provider)
        env_name = {"deepseek": "DEEPSEEK_API_KEY", "azure_openai": "USDCOP_AZURE_OPENAI_API_KEY"}[provider]
        key = os.environ.get(env_name)
        if not key:
            raise PilotBlocked("provider credential not present in inherited environment")
        base = spec["endpoint"].rstrip("/")
        if provider == "deepseek":
            url, headers = base + "/chat/completions", {"Authorization": "Bearer " + key}
        else:
            from urllib.parse import quote
            url = base + "/openai/deployments/" + quote(spec["requested_model"], safe="") + "/chat/completions?" + urlencode({"api-version": spec["api_version"]})
            headers = {"api-key": key}
        headers["Content-Type"] = "application/json"
        class NoRedirect(HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None

        # No cross-host redirect can forward Authorization/api-key headers.
        with build_opener(NoRedirect()).open(Request(url, data=canonical(request).encode(), headers=headers, method="POST"), timeout=45) as response:
            raw = response.read(1_000_001)
            if len(raw) > 1_000_000:
                raise PilotBlocked("provider response exceeds bounded envelope")
            payload = json.loads(raw)
            return {"content": payload["choices"][0]["message"].get("content") or "",
                    "served_model": payload.get("model"), "response_id": payload.get("id"),
                    "request_id": response.headers.get("x-request-id") or response.headers.get("apim-request-id"),
                    "system_fingerprint": payload.get("system_fingerprint"),
                    "finish_reason": payload["choices"][0].get("finish_reason"),
                    "usage": payload.get("usage"), "raw_sha256": hashlib.sha256(raw).hexdigest()}


class PilotRunner:
    def __init__(self, manifest: dict, store: PilotStore, sidecars: Path,
                 transport: object, *, clock: Callable[[], datetime] = utc_now):
        verify_manifest(manifest)
        self.manifest, self.store, self.sidecars = manifest, store, safe_path(sidecars)
        self.transport, self.clock = transport, clock

    def decide(self, context: dict, provider: str, variant: str) -> dict:
        verify_manifest(self.manifest)
        if provider not in PROVIDERS or variant not in VARIANTS:
            raise PilotBlocked("provider/variant not frozen")
        key = "::".join((self.manifest["manifest_sha256"], provider, variant,
                         context["session_date"], str(context["bar"])))
        existing = self.store.decision(key)
        if existing is not None:
            if existing["context_sha256"] != digest(context):
                raise PilotBlocked("cannot replace an existing decision using changed context")
            return existing
        now = self.clock()
        validate_context(context, self.manifest, now)
        spec = self.manifest["config"]["providers"][provider]
        if now > instant(spec["pricing"]["valid_until_utc"]):
            raise PilotBlocked("pricing has expired; new authority/freeze required before calls")
        prior_id = key.rsplit("::", 1)[0] + "::" + str(context["bar"] - 1)
        prior = self.store.decision(prior_id) if context["bar"] else None
        state = state_at(context, prior)
        system, user = prompts(variant, context, self.manifest, state)
        sampling = self.manifest["config"]["sampling"]
        request = {"model": spec["requested_model"], "messages": [
            {"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": sampling["temperature"], "top_p": sampling["top_p"],
            "max_tokens": sampling["max_tokens"]}
        if sanitize(canonical(request)) != canonical(request):
            raise PilotBlocked("request contains a secret-shaped token; no archive or API call")
        # UTF-8 byte length plus framing is a conservative admission bound, not a
        # tokenizer estimate. Do not silently truncate context to make budget fit.
        token_upper_bound = len(canonical(request).encode("utf-8")) + 512
        if token_upper_bound > self.manifest["config"]["max_input_tokens"]:
            raise PilotBlocked("prompt admission upper bound exceeds frozen token cap")
        self.store.admit(self.manifest)
        self.sidecars.mkdir(parents=True, exist_ok=True)
        request_file = self.sidecars / (hashlib.sha256(key.encode()).hexdigest() + ".request.json")
        request_archive = {"manifest_sha256": self.manifest["manifest_sha256"],
                           "context": context, "request": request, "state_before": state,
                           "archival_scope": "no_headers_no_credentials"}
        if request_file.exists():
            if json.loads(request_file.read_text(encoding="utf-8")) != request_archive:
                raise PilotBlocked("archived request differs; cannot overwrite")
        else:
            with request_file.open("x", encoding="utf-8") as handle:
                handle.write(canonical(request_archive) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        request_archive_sha256 = file_digest(request_file)
        attempts, valid, response, weight = [], False, {}, state["position"]
        hard_halt = False
        for attempt in range(sampling["max_retries"] + 1):
            started = self.clock()
            if started >= instant(context["decision_deadline_utc"]):
                raise PilotBlocked("decision deadline expired before provider request")
            call_id = key + "::attempt" + str(attempt)
            self.store.reserve_attempt(self.manifest, call_id, provider)
            error = None
            try:
                response = {}
                response = self.transport.generate(provider, spec, request)
                valid, candidate, parsed = parse_response(response.get("content", ""))
                usage = response.get("usage") or {}
                if not response.get("served_model") or not {"prompt_tokens", "completion_tokens"}.issubset(usage):
                    self.store.halt("missing_response_provenance")
                    hard_halt = True
                    raise PilotBlocked("response lacks served-model or actual usage provenance")
                if response["served_model"] != spec["expected_served_model"]:
                    self.store.halt("served_model_differs_from_preregistered_snapshot")
                    hard_halt = True
                    raise PilotBlocked("served model does not match frozen deployment/base-model expectation")
                if any(type(usage[k]) is not int or usage[k] < 0 for k in ("prompt_tokens", "completion_tokens")):
                    self.store.halt("invalid_actual_usage")
                    hard_halt = True
                    raise PilotBlocked("actual token usage must be nonnegative integers")
                if (int(usage["prompt_tokens"]) > self.manifest["config"]["max_input_tokens"]
                        or int(usage["completion_tokens"]) > sampling["max_tokens"]):
                    self.store.halt("actual_usage_above_reserved_cap")
                    hard_halt = True
                    raise PilotBlocked("actual usage exceeds reserved tariff cap; halt and reconcile billing")
                try:
                    self.store.check_served_model(self.manifest, provider, response["served_model"])
                except PilotBlocked:
                    hard_halt = True
                    raise
            except Exception as exc:
                # Never log exception text (HTTP clients may embed credentials).
                error, valid, parsed = type(exc).__name__, False, {}
            received = self.clock()
            sanitized = {k: v for k, v in response.items() if k != "content"}
            sanitized["content"] = sanitize(str(response.get("content", "")))
            sanitized["error_type"] = error
            self.sidecars.mkdir(parents=True, exist_ok=True)
            sidecar = self.sidecars / (hashlib.sha256(call_id.encode()).hexdigest() + ".json")
            with sidecar.open("x", encoding="utf-8") as handle:
                handle.write(canonical(sanitized) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            outcome = {"call_id": call_id, "attempt": attempt, "requested_at_utc": started.isoformat(),
                       "received_at_utc": received.isoformat(), "sidecar_path": str(sidecar.resolve()),
                       "sidecar_sha256": file_digest(sidecar), "error_type": error,
                       "valid_json": valid, "parsed": parsed,
                       "actual_usage": response.get("usage"),
                       "request_archive_path": str(request_file.resolve()),
                       "request_archive_sha256": request_archive_sha256,
                       "raw_response_sha256": response.get("raw_sha256")}
            self.store.outcome(call_id, outcome)
            attempts.append(outcome)
            if error is not None or valid:
                break
        eligible = valid and received < instant(context["decision_deadline_utc"])
        if eligible:
            weight = candidate
        change = weight - state["position"]
        cost = abs(change) * context["cost_context"]["one_way_return_per_unit"]
        after = {**state, "position": weight,
                 "session_pnl_decimal": state["session_pnl_decimal"] - cost}
        if change:
            old = state["position"]
            if not old or not weight or old * weight < 0:
                after["bars_in_position"] = 0
                after["entry_price"] = context["close"] if weight else None
            elif abs(weight) > abs(old):
                after["entry_price"] = (abs(old) * state["entry_price"]
                                         + abs(change) * context["close"]) / abs(weight)
            # Same-sign trimming preserves entry basis and holding age.
        after["unrealized_pnl_decimal"] = (weight * (context["close"] / after["entry_price"] - 1)
                                           if after["entry_price"] else 0.0)
        record = {
            "decision_id": key, "manifest_sha256": self.manifest["manifest_sha256"],
            "provider": provider, "variant": variant, "session_date": context["session_date"],
            "bar": context["bar"], "cutoff_utc": context["cutoff_utc"],
            "context_created_at_utc": context["context_created_at_utc"],
            "decision_deadline_utc": context["decision_deadline_utc"],
            "dataset_sha256": context["dataset_sha256"], "snapshot_sha256": context["snapshot_sha256"],
            "context_sha256": digest(context), "prompt_sha256": digest(request["messages"]),
            "request_sha256": digest(request), "requested_model": request["model"],
            "request_archive_path": str(request_file.resolve()),
            "request_archive_sha256": request_archive_sha256,
            "served_model": response.get("served_model"), "api_version": spec["api_version"],
            "expected_served_model": spec["expected_served_model"],
            "pricing_model": spec["pricing_model"],
            "pricing_basis": "frozen_tariff_upper_bound_not_provider_invoice",
            "sampling": {k: request[k] for k in ("temperature", "top_p", "max_tokens")},
            "requested_at_utc": attempts[0]["requested_at_utc"], "received_at_utc": received.isoformat(),
            "response_id": response.get("response_id"), "request_id": response.get("request_id"),
            "system_fingerprint": response.get("system_fingerprint"), "attempts": attempts,
            "valid_json": valid, "eligible_before_deadline": eligible, "weight": weight,
            "previous_weight": state["position"], "state_before": state, "state_after": after,
            "close": context["close"], "transaction_cost_return": cost,
            "status": "model_decision" if eligible else "unavailable_late_or_invalid_retained",
            "execution_basis": "decision_only_no_claim_of_fill_at_cutoff",
            "state_accounting_basis": "shadow_mark_to_observed_close_assumed_cost_not_realized_fill_pnl",
            "scope": "prospective_pilot_not_confirmatory",
        }
        self.store.append_decision(record)
        if hard_halt:
            raise PilotBlocked("shared pilot HALTED after archiving response; reconcile provenance/billing before proceeding")
        return record


def cohort_status(manifest: dict, records: list[dict]) -> dict:
    expected = {(p, v, s["session_date"], b) for p in PROVIDERS for v in VARIANTS
                for s in manifest["cohort"] for b in range(BARS)}
    keys = [(r["provider"], r["variant"], r["session_date"], r["bar"]) for r in records]
    complete = set(keys) == expected and len(keys) == len(expected)
    sound = all(r.get("manifest_sha256") == manifest["manifest_sha256"]
                and r.get("eligible_before_deadline") and r.get("valid_json") for r in records)
    return {"status": "COMPLETE_PILOT" if complete and sound else "INCOMPLETE_OR_INVALID",
            "expected_decisions": len(expected), "sealed_decisions": len(records),
            "missing_decisions": len(expected - set(keys)),
            "invalid_or_late": sum(not r.get("eligible_before_deadline") for r in records),
            "confirmatory": False, "scientific_edge_claim": False}
