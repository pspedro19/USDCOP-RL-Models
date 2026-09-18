"""Causal, auditable LLM decision adapter for the USD/COP thesis.

This module only supplies the experiment harness.  It does not load a ``.env`` file, place
orders, or decide the provider implicitly.  The caller must freeze provider/model/prompt before
opening any evaluation block and must provide a signed ledger path.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.analysis.llm_client import AzureOpenAIProvider, DeepSeekProvider, LLMProvider

PROMPT_VERSION = "thesis-llm-trader-v1"
ALLOWED_DIRECTIONS = {"short", "flat", "long"}
ALLOWED_SIZES = {0.0, 0.5, 1.0}


@dataclass(frozen=True)
class LLMDecision:
    direction: str
    size: float
    confidence: float
    weight: float
    valid: bool
    error: str | None = None


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _strip_code_fence(raw: str) -> str:
    """Quita la valla markdown que algunos modelos ponen alrededor del JSON.

    Medido el 2026-09-11: Azure `gpt-4o-mini` responde EN CONTRATO pero envuelto en una valla
    ```json ... ``` , mientras DeepSeek devuelve el objeto pelado. `json.loads` fallaba sobre la
    valla, el payload se degradaba a `{"_invalid_raw": ...}` y la decision salia
    `numeric_field_invalid`: el brazo de robustez entero habria sido 13.334 respuestas
    "invalidas" que en realidad eran validas, todas resueltas como `retain_previous_weight`.
    Un brazo asi no mide al modelo, mide al parser.

    Se aplica **igual a todos los proveedores**: es correccion del arnes, no ajuste por
    proveedor. Si solo se aplicara a Azure seria exactamente la clase de trato desigual que
    invalidaria la comparacion entre brazos.
    """
    text = raw.strip()
    if not text.startswith("```"):
        return text
    body = text[3:]
    newline = body.find("\n")
    if newline != -1 and body[:newline].strip().isalpha():
        body = body[newline + 1:]          # descarta el idioma de la valla (```json)
    if body.rstrip().endswith("```"):
        body = body.rstrip()[:-3]
    return body.strip()


def parse_decision(payload: Any, previous_weight: float) -> LLMDecision:
    """Validate the frozen JSON contract; invalid responses keep the previous exposure."""
    if not isinstance(payload, dict):
        return LLMDecision("flat", 0.0, 0.0, previous_weight, False, "response_not_object")
    direction = str(payload.get("direccion", "")).lower()
    try:
        size = float(payload.get("tamano"))
        confidence = float(payload.get("confianza"))
    except (TypeError, ValueError):
        return LLMDecision("flat", 0.0, 0.0, previous_weight, False, "numeric_field_invalid")
    if direction not in ALLOWED_DIRECTIONS:
        return LLMDecision("flat", 0.0, 0.0, previous_weight, False, "direction_invalid")
    if size not in ALLOWED_SIZES or not 0.0 <= confidence <= 1.0:
        return LLMDecision("flat", 0.0, 0.0, previous_weight, False, "size_or_confidence_invalid")
    sign = {"short": -1.0, "flat": 0.0, "long": 1.0}[direction]
    weight = sign * size
    return LLMDecision(direction, size, confidence, weight, True)


def provider_from_environment(provider: str) -> LLMProvider:
    """Construct exactly the requested provider; no automatic fallback across arms."""
    name = provider.strip().lower()
    if name == "deepseek":
        return DeepSeekProvider()
    if name in {"azure", "azure_openai"}:
        return AzureOpenAIProvider()
    raise ValueError("provider must be 'deepseek' or 'azure_openai'")


class ThesisLLMRunner:
    """One-provider runner with append-only, hash-based decision records."""

    def __init__(self, provider: LLMProvider, provider_name: str, model_id: str,
                 ledger_path: Path, prompt_version: str = PROMPT_VERSION):
        if not provider_name or not model_id or not prompt_version:
            raise ValueError("provider, model_id, and prompt_version must be frozen")
        self.provider = provider
        self.provider_name = provider_name
        self.model_id = model_id
        self.ledger_path = Path(ledger_path)
        self.prompt_version = prompt_version
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._decision_ids: set[str] = set()
        if self.ledger_path.is_file():
            with self.ledger_path.open(encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        prior = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"ledger line {line_no} is not valid JSON") from exc
                    decision_id = prior.get("decision_id")
                    if not isinstance(decision_id, str) or not decision_id:
                        raise ValueError(f"ledger line {line_no} has no decision_id")
                    self._decision_ids.add(decision_id)

    def decide(self, *, session_date: str, bar: int, system_prompt: str, user_prompt: str,
               previous_weight: float, max_tokens: int = 256, temperature: float = 0.10,
               attempt: int = 1, dataset_block: str | None = None,
               dataset_sha256: str | None = None, retrospective: bool | None = None,
               top_p: float = 0.90, max_retries_invalid_json: int = 1) -> LLMDecision:
        if not 0 <= bar <= 58:
            raise ValueError("bar must be in [0, 58]")
        if max_tokens <= 0 or not 0.0 <= temperature <= 2.0 or not 0.0 < top_p <= 1.0:
            raise ValueError("invalid frozen LLM sampling parameters")
        if max_retries_invalid_json < 0:
            raise ValueError("max_retries_invalid_json must be non-negative")
        decision_id = f"{session_date}::llm::{bar}"
        if decision_id in self._decision_ids:
            raise ValueError(f"duplicate decision_id refused: {decision_id}")
        started = time.perf_counter()
        raw = ""
        response: dict[str, Any] = {}
        error: str | None = None
        decision = LLMDecision("flat", 0.0, 0.0, previous_weight, False, "response_not_object")
        actual_attempt = attempt
        for retry_index in range(max_retries_invalid_json + 1):
            actual_attempt = attempt + retry_index
            try:
                response = self.provider.generate(system_prompt, user_prompt, max_tokens, temperature)
                raw = str(response.get("content", ""))
                try:
                    payload = json.loads(_strip_code_fence(raw))
                except json.JSONDecodeError:
                    payload = {"_invalid_raw": raw}
                decision = parse_decision(payload, previous_weight)
                if decision.valid or retry_index >= max_retries_invalid_json:
                    break
            except Exception as exc:  # fail closed: keep w_prev, record the reason
                error = type(exc).__name__
                decision = LLMDecision("flat", 0.0, 0.0, previous_weight, False, error)
                break
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        record = {
            "decision_id": decision_id,
            "session_date": session_date,
            "dataset_block": dataset_block,
            "dataset_sha256": dataset_sha256,
            "retrospective": retrospective,
            "bar": bar,
            "provider": self.provider_name,
            "model_id": self.model_id,
            "prompt_version": self.prompt_version,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "prompt_hash": _sha256_text(system_prompt + "\n" + user_prompt),
            "raw_response_sha256": _sha256_text(raw),
            "direction": decision.direction,
            "size": decision.size,
            "confidence": decision.confidence,
            "weight": decision.weight,
            "previous_weight": previous_weight,
            "valid_json": decision.valid,
            # Invalid JSON means the provider answered but violated the schema;
            # unavailable means the call itself failed (timeout, missing key, etc.).
            "unavailable": bool(error is not None and not raw),
            "error": decision.error or error,
            "attempt": actual_attempt,
            "latency_ms": elapsed_ms,
            "tokens_used": int(response.get("tokens_used", 0) or 0),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "cost_known": bool(response.get("cost_known", False)),
        }
        with self.ledger_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self._decision_ids.add(decision_id)
        return decision


def configured_provider_name() -> str:
    return os.environ.get("THESIS_LLM_PROVIDER", "deepseek")
