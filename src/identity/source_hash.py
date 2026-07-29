"""Hash canónico LF de FUENTES — la ÚNICA implementación de producción (CXD-041/043).

Por qué existe este módulo (BL-13 red-team): el método de hashing que sella los
manifiestos congelados (`config/strategy_manifests/*.yaml`) vivía DENTRO de su propio
test de regresión. Un test que se verifica contra su propia implementación no tiene nada
de producción que mutar: la garantía era circular. La lógica se extrae aquí y TODOS los
consumidores la importan — el gate del catálogo de features
(`scripts/validation/validate_feature_catalog.py`) y los tests de regresión — de modo que
mutar esta función pone en rojo las murallas que dependen de ella.

El método: sha256 sobre los bytes del fichero con **CRLF -> LF normalizado**, es decir el
contenido del blob git bajo `.gitattributes` `* text=auto eol=lf`. Equivale a
`git show :<path> | sha256sum` en cualquier SO. Hashear los bytes crudos del working tree
(el método viejo) fijaba los CRLF de Windows en el hash declarado y un checkout limpio en
Linux NO podía reproducir el freeze.

ALCANCE: SOLO ficheros de código/texto. Los ARTEFACTOS BINARIOS (`.pkl`, parquet) se
hashean con sus bytes crudos — ahí la identidad ES el byte, y normalizar CRLF los
corrompería (ver `scripts/validation/bitcheck_v11_signal.py`).
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

__all__ = ["canonical_lf", "sha256_16", "file_code_hash", "files_code_hash"]

#: Longitud del prefijo de hash publicado en manifiestos/catálogos.
SHA16_LEN = 16


def canonical_lf(data: bytes) -> bytes:
    """Normalización CRLF -> LF: el flujo de bytes canónico para hashear fuentes."""
    return data.replace(b"\r\n", b"\n")


def sha256_16(data: bytes) -> str:
    """sha256[:16] de un flujo de bytes YA canonicalizado."""
    return hashlib.sha256(data).hexdigest()[:SHA16_LEN]


def file_code_hash(path: str | Path) -> str:
    """Hash canónico LF de UN fichero de código (reproducible desde el blob git)."""
    return sha256_16(canonical_lf(Path(path).read_bytes()))


def files_code_hash(paths: Iterable[str | Path]) -> str:
    """Hash canónico LF de una SECUENCIA ORDENADA de ficheros (concatenación).

    El orden es parte del hash: es el orden declarado en `files:` /
    `code_reference:` / `spec_fingerprint_inputs:` del manifiesto congelado.
    """
    digest = hashlib.sha256()
    for path in paths:
        digest.update(canonical_lf(Path(path).read_bytes()))
    return digest.hexdigest()[:SHA16_LEN]
