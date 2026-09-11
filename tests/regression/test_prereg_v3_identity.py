"""El pre-registro v3 congela una identidad; este test comprueba que la congela de verdad.

Un pre-registro vale exactamente lo que valga su tabla de identidad: si el hash que declara
no es el del artefacto, no hay nada congelado y la afirmacion "esto es lo que se entreno" se
vuelve incomprobable justo cuando alguien la audite.

Esto no es hipotetico. El 2026-09-11, la fila del schema declaraba

    a0568db4b953604cabb6d64eab73631419f65184d0fc46f35f36c695487484cb

mientras el artefacto declaraba

    a0568db4b953604cabb6d64eab73631419f65184d0fc96bd61a6f994f75d2e8b

Los 46 primeros caracteres coinciden y despues divergen: los ultimos 18 del valor publicado
son la cola de la identidad del dataset portable
(`c4d32158...2fd46f35f36c695487484cb`). Fue una pegada defectuosa, no una colision -- dos
sha256 independientes no comparten sufijo. El error sobrevivio a la redaccion y a la revision
porque un hash largo se lee como ruido: nadie compara sesenta y cuatro caracteres a ojo.

Por eso se compara aqui, y por eso el test tambien prohibe que dos hashes distintos del
documento compartan cola: es la firma exacta del defecto que lo motivo.
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / ".claude" / "specs" / "planes" / "06-PRE-REGISTRATION-v3.md"
SCHEMA = ROOT / "config" / "research" / "feature_schema_v2.json"
PORTABLE = ROOT / "data" / "thesis" / "research_data_portable_v2.pkl"

_SHA_RE = re.compile(r"\b([0-9a-f]{64})\b")


def _prereg_text() -> str:
    if not PREREG.is_file():
        pytest.skip("el pre-registro v3 no existe en este checkout")
    return PREREG.read_text(encoding="utf-8")


def _identity_table() -> str:
    """Solo la tabla de identidad congelada, no el cuerpo del documento.

    El bloque de correccion cita a proposito el hash equivocado para dejar constancia; si el
    test mirara el documento entero, esa cita lo pondria en rojo para siempre.
    """
    text = _prereg_text()
    start = text.index("## Identidad congelada")
    end = text.index("## Universo y cargos")
    return text[start:end]


def test_schema_hash_in_prereg_matches_the_artifact() -> None:
    if not SCHEMA.is_file():
        pytest.skip("feature_schema_v2.json no existe en este checkout")
    declared = json.loads(SCHEMA.read_text(encoding="utf-8"))["sha256"]
    assert declared in _identity_table(), (
        "la tabla de identidad del pre-registro v3 no contiene el sha256 que declara "
        f"{SCHEMA.relative_to(ROOT).as_posix()} ({declared}). Un congelamiento cuyo hash no "
        "coincide con el artefacto no congela nada."
    )


def test_portable_identity_in_prereg_matches_the_artifact() -> None:
    if not PORTABLE.is_file():
        pytest.skip("research_data_portable_v2.pkl no existe en este checkout")
    blob = pickle.loads(PORTABLE.read_bytes())
    identity = getattr(blob, "identity", None)
    if identity is None and isinstance(blob, dict):
        identity = blob.get("identity")
    if not identity:
        pytest.skip("el dataset portable no expone identidad")
    assert str(identity) in _identity_table(), (
        "la identidad del dataset portable no coincide con la publicada en el pre-registro v3"
    )


def test_no_two_frozen_hashes_share_a_tail() -> None:
    """La firma del defecto real: una cola pegada de otro hash.

    Dos sha256 de artefactos distintos comparten un sufijo largo con probabilidad
    despreciable. Si ocurre, es copia y pega, que es exactamente lo que paso.
    """
    hashes = sorted(set(_SHA_RE.findall(_identity_table())))
    for i, left in enumerate(hashes):
        for right in hashes[i + 1:]:
            assert left[-18:] != right[-18:], (
                f"dos hashes de la tabla de identidad comparten los ultimos 18 caracteres "
                f"({left} y {right}). No es una colision: es una pegada defectuosa."
            )
