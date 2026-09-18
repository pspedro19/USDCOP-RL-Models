#!/usr/bin/env python
"""Contrafactual de REPRESENTACION del OHLC sobre las entradas del HMM.

## Que mide, dicho con precision

Construye el vector de observacion diario del HMM dos veces sobre el MISMO seed:

  * `real`      -- el OHLC tal cual esta en el seed;
  * `aplanado`  -- forzando `O = H = L = C`, que es la forma que el propio seed tiene en
                   2020-2022 (100 % de barras planas, rango intrabarra 0,000).

## Que NO demuestra

La diferencia mide **sensibilidad de las features a que exista rango intrabarra**. **No** es una
descomposicion de causas: aplanar elimina informacion real, asi que el delta observado **no
prueba** que esa fraccion se deba exclusivamente al cambio de proveedor. Separar volatilidad
real de artefacto de fuente exigiria una serie de referencia independiente para el mismo periodo,
que no se tiene.

Lo que si establece, y nada mas: **si las barras de un ano tuvieran la representacion de
2020-2022, estas features valdrian lo que dice la columna `aplanado`.** La fraccion del cambio
observado entre anos que corresponde al proveedor **no esta identificada**: hacerlo exigiria una
serie de referencia independiente para el mismo periodo, que no se tiene. Ni siquiera «en parte»
es afirmable, porque eso ya asertaria una fraccion positiva no identificada (CXD-861).

Correcciones pedidas por Codex sobre versiones previas: CXD-860 (presentaba el delta como
atribucion al proveedor) y CXD-861 (estadisticos pareados, cohortes, alcance del control).

El posterior del HMM **no** se compara aqui: `PortableRegimeModel.load` aborta hoy por deriva de
identidad y reconstruir el modelo a mano produciria un posterior que no corresponde a ningun
artefacto declarado.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.regime_hmm import build_regime_observations  # noqa: E402

RANGE_FEATURES = ("atr_norm", "range_over_atr")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paired(real: np.ndarray, flattened: np.ndarray) -> dict:
    """Distribucion pareada de la diferencia relativa, sesion a sesion."""
    ok = flattened != 0.0
    if not ok.any():
        return {"paired_available": False}
    rel = 100.0 * (real[ok] / flattened[ok] - 1.0)
    return {
        "paired_available": True,
        "paired_n": int(rel.size),
        "paired_median_pct": float(np.median(rel)),
        "paired_mean_pct": float(np.mean(rel)),
        "paired_p05_pct": float(np.percentile(rel, 5)),
        "paired_p95_pct": float(np.percentile(rel, 95)),
        "paired_max_pct": float(np.max(rel)),
        "paired_share_above_5pct": float(np.mean(rel > 5.0)),
    }


def measure(seed: Path, years: tuple[int, ...]) -> dict:
    m5 = pd.read_parquet(seed)
    flat = m5.copy()
    for column in ("open", "high", "low"):
        flat[column] = flat["close"]

    obs_real = build_regime_observations(m5)
    obs_flat = build_regime_observations(flat)
    if len(obs_real) != len(obs_flat):
        raise ValueError("las dos variantes deben producir el mismo numero de observaciones")

    by_year: dict[str, dict] = {}
    for year in years:
        sel_r = obs_real[obs_real.index.year == year]
        sel_f = obs_flat[obs_flat.index.year == year]
        entry: dict[str, dict] = {
            "n_sessions": int(len(sel_r)),
            # Cohorte explicita: estas son fechas de CALENDARIO agregadas por ano, NO la cohorte
            # enmascarada de 226 sesiones que usa la tesis. Mezclar los dos denominadores fue
            # una reserva de CXD-861 y se evita nombrandolos.
            "cohort": "calendar_year_of_all_regime_observations",
            "first_date": str(sel_r.index.min().date()) if len(sel_r) else None,
            "last_date": str(sel_r.index.max().date()) if len(sel_r) else None,
        }
        for feature in RANGE_FEATURES:
            r = sel_r[feature].to_numpy(dtype=float)
            f = sel_f[feature].to_numpy(dtype=float)
            ok = np.isfinite(r) & np.isfinite(f)
            r, f = r[ok], f[ok]
            entry[feature] = {
                "n_finite": int(ok.sum()),
                "mean_real": float(np.mean(r)) if len(r) else None,
                "mean_flattened": float(np.mean(f)) if len(f) else None,
                "median_real": float(np.median(r)) if len(r) else None,
                "median_flattened": float(np.median(f)) if len(f) else None,
                # `ratio_of_means` y `ratio_of_medians` son agregados; `paired_*` son la
                # distribucion de la diferencia relativa SESION A SESION. No son lo mismo y
                # confundirlos fue una reserva de CXD-861: la cercania entre los dos agregados
                # no dice nada sobre outliers, y los pareados si.
                "ratio_of_means_minus_one_pct": (float(100.0 * (np.mean(r) / np.mean(f) - 1.0))
                                                 if len(f) and np.mean(f) else None),
                "ratio_of_medians_minus_one_pct": (float(100.0 * (np.median(r) / np.median(f) - 1.0))
                                                   if len(f) and np.median(f) else None),
                **_paired(r, f),
            }
        by_year[str(year)] = entry

    flat_share = {}
    t = pd.to_datetime(m5["time"])
    is_flat = ((m5["open"] == m5["high"]) & (m5["high"] == m5["low"])
               & (m5["low"] == m5["close"]))
    for year, share in is_flat.groupby(t.dt.year).mean().items():
        flat_share[str(int(year))] = float(share)

    return {
        "contract": "CTR-RESEARCH-OHLC-REPRESENTATION-COUNTERFACTUAL-001",
        "scope": "representation_counterfactual",
        "proves": [
            "range_dependent_HMM_inputs_change_when_intrabar_range_is_removed",
        ],
        "does_not_prove": [
            "attribution_of_the_delta_to_the_data_vendor",
            "separation_of_real_volatility_from_source_artefact",
            "any_effect_on_the_regime_posterior",
            "any_effect_on_decisions_or_returns",
        ],
        "seed_path": str(seed),
        "seed_sha256": _sha256(seed),
        # Autocontenido: quien lea el artefacto puede verificar el codigo que lo produjo y el
        # constructor de features, sin depender de este mensaje (CXD-861).
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "feature_builder_sha256": _sha256(ROOT / "src" / "research" / "regime_hmm.py"),
        "n_observations": int(len(obs_real)),
        "features": list(RANGE_FEATURES),
        "flat_ohlc_fraction_by_year": flat_share,
        "by_year": by_year,
        "runtime": {
            "python": platform.python_version(),
            "executable": sys.executable,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "measured_at_utc": datetime.now(UTC).isoformat(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=Path,
                    default=ROOT / "seeds" / "latest" / "usdcop_m5_ohlcv.parquet")
    ap.add_argument("--years", type=int, nargs="+", default=[2021, 2023, 2026])
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    if args.output.exists():
        # Sobrescribir un artefacto es perder la version anterior sin dejar rastro.
        print(f"rehusado: {args.output} ya existe; elige otra ruta o archiva la anterior",
              file=sys.stderr)
        return 2
    report = measure(args.seed, tuple(args.years))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({"output": str(args.output),
                      "self_sha256": _sha256(args.output),
                      "years": args.years}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
