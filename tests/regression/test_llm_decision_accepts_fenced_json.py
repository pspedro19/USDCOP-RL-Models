"""Una respuesta EN contrato no puede contarse como invalida por la valla que la envuelve.

El 2026-09-11, al abrir el brazo de robustez, Azure `gpt-4o-mini` devolvio esto:

    ```json
    {"direccion": "flat", "tamano": 0, "confianza": 0}
    ```

que es exactamente el contrato congelado, envuelto en una valla markdown. `json.loads` fallaba,
el payload se degradaba a `{"_invalid_raw": ...}` y la decision se sellaba como
`numeric_field_invalid` con la politica `retain_previous_weight`.

Consecuencia si no se arregla: **13.334 llamadas pagadas cuyo resultado sellado es "el modelo no
supo responder"**, cuando el modelo respondio bien las 13.334 veces. El brazo no habria medido al
modelo, habria medido al parser -- y la comparacion DeepSeek (JSON pelado) contra Azure (JSON
vallado) habria salido demoledora a favor de DeepSeek por una razon que no tiene nada que ver con
operar USD/COP.

Por eso la correccion se aplica a TODOS los proveedores por igual: aplicarla solo a Azure seria
trato desigual entre brazos, que es el mismo pecado por el otro lado.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.llm_trader import _strip_code_fence, parse_decision  # noqa: E402

FENCED = '```json\n{\n  "direccion": "short",\n  "tamano": 0.5,\n  "confianza": 0.7\n}\n```'
BARE = '{"direccion": "short", "tamano": 0.5, "confianza": 0.7}'


def test_fenced_and_bare_json_produce_the_same_decision() -> None:
    """Lo que decide es el contenido; la valla es envoltorio y no puede cambiar el resultado."""
    fenced = parse_decision(json.loads(_strip_code_fence(FENCED)), previous_weight=0.0)
    bare = parse_decision(json.loads(_strip_code_fence(BARE)), previous_weight=0.0)
    assert fenced == bare
    assert fenced.valid is True
    assert fenced.weight == -0.5


def test_bare_json_is_untouched() -> None:
    """DeepSeek responde pelado: la correccion no puede alterar su camino."""
    assert _strip_code_fence(BARE) == BARE


def test_a_fence_without_language_tag_also_parses() -> None:
    assert json.loads(_strip_code_fence('```\n{"direccion": "flat", "tamano": 0, "confianza": 0}\n```'))


def test_genuine_garbage_is_still_invalid() -> None:
    """La correccion NO puede convertir en valida una respuesta que de verdad no lo es."""
    for junk in ("lo siento, no puedo ayudar con eso", "```json\nno es json\n```", ""):
        try:
            payload = json.loads(_strip_code_fence(junk))
        except json.JSONDecodeError:
            payload = {"_invalid_raw": junk}
        assert parse_decision(payload, previous_weight=0.25).valid is False


def test_an_out_of_contract_field_is_still_refused() -> None:
    """Tamano fuera del conjunto congelado sigue siendo invalido aunque venga bien envuelto."""
    bad = '```json\n{"direccion": "long", "tamano": 0.37, "confianza": 0.9}\n```'
    decision = parse_decision(json.loads(_strip_code_fence(bad)), previous_weight=0.25)
    assert decision.valid is False
    assert decision.weight == 0.25, "una respuesta invalida conserva la exposicion previa"
