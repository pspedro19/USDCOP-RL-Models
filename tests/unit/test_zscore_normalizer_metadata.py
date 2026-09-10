# -*- coding: utf-8 -*-
"""El normalizador debe aceptar los `norm_stats` que el propio pipeline escribe.

`ZScoreNormalizer._load_stats` exigia `mean`+`std` a **todas** las claves de nivel
superior. Pero produccion escribe metadatos ahi:

* `src/data/ssot_dataset_builder.py:632`  -> `stats["_meta"] = {...}`
* `src/training/engine.py:705`            -> `norm_stats["_metadata"] = {...}`
* `scripts/data/generate_dataset_variants.py:355` -> `norm_stats["_metadata"]`

y otros consumidores ya las saltan a proposito (`l4_per_seed_asym.py:79`,
`ssot_lineage_integration.py:376`), o sea la convencion es real. El normalizador
rechazaba esos ficheros con `ValueError: Stats for _meta must have 'mean' and
'std' keys`. Lo destapo `tests/integration/test_determinism.py` al resucitarla en
`c665b539`; el alcance de la correccion lo fijo CODEX en CXD-316.

Alcance deliberadamente estrecho, y cada limite tiene su candado abajo:

1. Se aceptan **solo** `_meta` y `_metadata`, las dos que hay productores emitiendo.
   **NO** se salta cualquier clave con guion bajo: eso ocultaria features
   malformadas.
2. Ambas deben ser objetos. Un `_meta` escalar sigue siendo rojo.
3. Cualquier otra clave conserva la exigencia estricta.
4. No se canoniza cual de las dos convenciones es la buena, ni se tocan los
   productores: esa decision es de SSOT y no de este arreglo.
"""
from __future__ import annotations

import json

import pytest

from src.core.normalizers.zscore_normalizer import ZScoreNormalizer


def _write(tmp_path, payload) -> str:
    path = tmp_path / "norm_stats.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


_FEATURE = {"mean": 1.0, "std": 2.0}


@pytest.mark.parametrize("reserved", ["_meta", "_metadata"])
def test_reserved_metadata_keys_are_accepted(tmp_path, reserved):
    """Las dos convenciones que existen en produccion deben cargar."""
    path = _write(tmp_path, {"rsi_9": _FEATURE, reserved: {"created_at": "2026-01-01"}})
    normalizer = ZScoreNormalizer(stats_path=path)
    assert normalizer.normalize("rsi_9", 3.0) == pytest.approx(1.0)


@pytest.mark.parametrize("reserved", ["_meta", "_metadata"])
def test_reserved_metadata_is_not_exposed_as_a_feature(tmp_path, reserved):
    """No debe convertirse en pseudo-feature con mean=0/std=1 en el cache."""
    path = _write(tmp_path, {"rsi_9": _FEATURE, reserved: {"created_at": "2026-01-01"}})
    normalizer = ZScoreNormalizer(stats_path=path)
    assert reserved not in normalizer._stats_cache


@pytest.mark.parametrize("reserved", ["_meta", "_metadata"])
def test_reserved_metadata_must_still_be_an_object(tmp_path, reserved):
    """Fail-closed: un metadato escalar no se acepta por llamarse `_meta`."""
    path = _write(tmp_path, {"rsi_9": _FEATURE, reserved: "no-soy-un-objeto"})
    with pytest.raises(ValueError):
        ZScoreNormalizer(stats_path=path)


def test_other_underscore_keys_are_not_exempt(tmp_path):
    """El candado central: NO se salta cualquier `_...`.

    Saltar todo lo que empiece por guion bajo convertiria una feature malformada
    en invisible. Solo las dos claves con productor conocido son metadata.
    """
    path = _write(tmp_path, {"rsi_9": _FEATURE, "_notas": {"algo": 1}})
    with pytest.raises(ValueError):
        ZScoreNormalizer(stats_path=path)


def test_ordinary_feature_still_requires_mean_and_std(tmp_path):
    path = _write(tmp_path, {"rsi_9": {"mean": 1.0}})
    with pytest.raises(ValueError):
        ZScoreNormalizer(stats_path=path)


def test_ordinary_feature_must_be_a_mapping(tmp_path):
    path = _write(tmp_path, {"rsi_9": 3.0})
    with pytest.raises(ValueError):
        ZScoreNormalizer(stats_path=path)
