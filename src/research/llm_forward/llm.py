"""LLM client: deterministic settings, structured output, full call logging.

Design constraints that come from the experiment rather than from taste:

* **Determinism as far as the API allows.** ``temperature=0`` plus a fixed
  ``seed``. This never guarantees reproducibility — the provider can change the
  serving stack underneath you — which is why ``system_fingerprint`` is captured
  on every call. When it changes, the comparison across that boundary is no
  longer apples to apples, and you want that in the record, not in your memory.

* **Structured output, not parsing.** The score feeds a numeric pipeline. Asking
  for prose and regexing a number out of it introduces a failure mode that looks
  like model behaviour but is really a parser bug. A JSON schema pushes that
  failure to the API boundary where it raises instead of silently returning 0.0.

* **No retries on content.** A transport retry is fine. Re-prompting because you
  did not like the answer is p-hacking with extra steps.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

from .schema import Decision, LlmUsage

# Priced per million tokens. Verify against current provider pricing before
# quoting these in a thesis — they move, and a stale constant in a cost table is
# the kind of error a reviewer will find.
PRICE_PER_MTOK: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}

DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["score", "direction", "confidence", "rationale"],
    "properties": {
        "score": {
            "type": "number",
            "minimum": -1.0,
            "maximum": 1.0,
            "description": "Directional score for the session. -1 strongly "
                           "bearish USD/COP, +1 strongly bullish, 0 neutral.",
        },
        "direction": {"type": "string", "enum": ["long", "short", "flat"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "rationale": {
            "type": "string",
            "description": "Max 3 sentences. Cite doc_ids that drove the score.",
        },
    },
}


@dataclass
class LlmResult:
    decision: Decision
    usage: LlmUsage
    raw_response: dict[str, Any]


class LlmClient:
    """Thin wrapper over the OpenAI SDK, usable against Azure or api.openai.com.

    The provider is chosen by which environment variables are present, so the
    same code runs in both places without a flag to forget to set.
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        seed: int | None = 20260101,
    ) -> None:
        self.temperature = temperature
        self.seed = seed
        self.provider, self._client, self.model = self._build_client(model)

    @staticmethod
    def _build_client(model: str | None) -> tuple[str, Any, str]:
        azure_endpoint = _env("AZURE_OPENAI_ENDPOINT")

        if azure_endpoint:
            from openai import AzureOpenAI

            client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=_require_env("AZURE_OPENAI_API_KEY"),
                api_version=_env("AZURE_OPENAI_API_VERSION") or "2024-10-21",
            )
            # On Azure the "model" argument is the *deployment* name you chose in
            # the portal, not the model id. Mixing these up is the single most
            # common Azure setup error.
            deployment = model or _require_env("AZURE_OPENAI_DEPLOYMENT")
            return "azure", client, deployment

        from openai import OpenAI

        client = OpenAI(api_key=_require_env("OPENAI_API_KEY"))
        return "openai", client, model or "gpt-4o-mini"

    def decide(self, system_prompt: str, user_prompt: str) -> LlmResult:
        """Make one scored decision. Raises on malformed output rather than
        substituting a default."""
        started = time.perf_counter()

        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            seed=self.seed,
            top_p=1.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "session_decision",
                    "strict": True,
                    "schema": DECISION_SCHEMA,
                },
            },
        )

        latency_ms = int((time.perf_counter() - started) * 1000)
        payload = json.loads(response.choices[0].message.content)

        usage = response.usage
        decision = Decision(
            score=float(payload["score"]),
            direction=payload["direction"],
            confidence=float(payload["confidence"]),
            rationale=payload["rationale"],
        )

        return LlmResult(
            decision=decision,
            usage=LlmUsage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                latency_ms=latency_ms,
                cost_usd=estimate_cost(
                    self.model, usage.prompt_tokens, usage.completion_tokens
                ),
                system_fingerprint=getattr(response, "system_fingerprint", None),
            ),
            raw_response=response.model_dump(),
        )


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Approximate USD cost of one call. Returns 0.0 for unpriced models."""
    for known, (in_price, out_price) in PRICE_PER_MTOK.items():
        if known in model:
            return round(
                prompt_tokens / 1e6 * in_price + completion_tokens / 1e6 * out_price,
                6,
            )
    return 0.0


# Este repositorio nombra sus credenciales con prefijo `USDCOP_` (ver
# `docker-compose.compact.yml`), y el arnes venia escrito contra los nombres genericos.
# Se resuelve en UN solo punto en vez de duplicar la clave en dos variables: una clave
# copiada a dos sitios se desincroniza, y la que sobra acaba en un fichero que alguien
# commitea. Precedente vivo: BL-08, con `.env` real en el historial publico.
ENV_PREFIXES = ("USDCOP_", "")


def _env(name: str) -> str | None:
    """Primer valor no vacio entre `USDCOP_<name>` y `<name>`."""
    for prefix in ENV_PREFIXES:
        value = os.getenv(f"{prefix}{name}")
        if value:
            return value
    return None


def _require_env(name: str) -> str:
    value = _env(name)
    if not value:
        tried = " o ".join(f"{p}{name}" for p in ENV_PREFIXES)
        raise RuntimeError(
            f"{tried} no esta definida. Ponla en .env (que DEBE estar gitignorado) y "
            "cargala con python-dotenv. Nunca en el codigo."
        )
    return value
