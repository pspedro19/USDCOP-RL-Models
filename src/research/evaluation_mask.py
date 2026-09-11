"""Mascara de evaluacion comun: que sesiones cuentan, y por que.

Contract: CTR-RESEARCH-EVALMASK-001 · Date: 2026-08-24

Implementa §9.5 del plan de tesis (`.claude/specs/planes/06-tesis-rl-llm-hibrido.md`) y su
test 15: **todos los sistemas se evaluan sobre el mismo conjunto de sesiones validas;
ninguna sesion invalida entra como retorno 0 ni cuenta en `n`**.

## Por que hace falta

Sin mascara, una sesion que el proveedor invento (festivo con 60 barras "perfectas") o una
a la que le faltan barras entra en la serie diaria como un retorno cualquiera. Dos danos
distintos:

  1. **Sesga la media**: un festivo relleno suele tener retorno ~0, lo que baja la
     volatilidad y sube el Sharpe de TODAS las estrategias por igual — incluidos los
     baselines, asi que el efecto no se cancela en la diferencia pareada.
  2. **Infla `n`**: el bootstrap y los IC se calculan sobre un tamano de muestra que no
     existe, y el analisis de poder de §11.2 pasa a mentir.

## Que excluye (criterio fijado ex-ante en el pre-registro)

- **Festivos colombianos**, tengan sesion completa o residuo parcial. Medido el 2026-08-24:
  11 festivos con 60 barras (relleno sintetico puro) y el resto con 2-18 barras.
- **Sesiones incompletas** (< `BARS_PER_SESSION`). 288 de 1.723 en la serie reparada.
- **Fines de semana**, que no deberian existir tras `fix_tz_wall_cot_seed.py` pero se
  filtran igual: una mascara que confia en que el dato de entrada esta limpio no es una
  mascara.

Lo que NO excluye: sesiones completas en dia habil, aunque su retorno sea raro. Filtrar por
el VALOR del retorno seria seleccionar sobre el resultado.

## Salida

`EvaluationMask` con las fechas validas y un `sha256` reproducible del conjunto. Ese hash se
publica junto a los resultados: dos corridas que declaren el mismo hash evaluaron sobre las
mismas sesiones, y una diferencia de hash invalida la comparacion.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd
import numpy as np

REPO = Path(__file__).resolve().parents[2]
CALENDAR = REPO / "config" / "trading_calendar.json"
PARTITION = REPO / "config" / "research" / "partition.yaml"
DEFAULT_SEED = REPO / "seeds" / "latest" / "usdcop_m5_ohlcv.parquet"

BARS_PER_SESSION = 60          # 08:00-12:55 COT en pasos de 5 minutos
SESSION_LO, SESSION_HI = 8, 12


# Festivos colombianos que la lista canonica de `config/trading_calendar.json` no cubre
# (solo trae 2025). Derivados de los que aparecen CON DATOS en la serie, cruzados a mano
# con el calendario oficial. Sin esta lista, un festivo de 2024 con 60 barras entraria
# como sesion valida.
EXTRA_COLOMBIA_HOLIDAYS = (
    # 2020
    "2020-01-01", "2020-01-06", "2020-03-23", "2020-04-09", "2020-04-10",
    "2020-05-01", "2020-05-25", "2020-06-15", "2020-06-22", "2020-06-29",
    "2020-07-20", "2020-08-07", "2020-08-17", "2020-10-12", "2020-11-02",
    "2020-11-16", "2020-12-08", "2020-12-25",
    # 2021
    "2021-01-01", "2021-01-11", "2021-03-22", "2021-04-01", "2021-04-02",
    "2021-05-17", "2021-06-07", "2021-06-14", "2021-07-05", "2021-07-20",
    "2021-08-16", "2021-10-18", "2021-11-01", "2021-11-15", "2021-12-08",
    "2021-12-25",
    # 2022
    "2022-01-01", "2022-01-10", "2022-03-21", "2022-04-14", "2022-04-15",
    "2022-05-30", "2022-06-20", "2022-06-27", "2022-07-04", "2022-07-20",
    "2022-08-15", "2022-10-17", "2022-11-07", "2022-11-14", "2022-12-08",
    "2022-12-25",
    # 2023
    "2023-01-01", "2023-01-09", "2023-03-20", "2023-04-06", "2023-04-07",
    "2023-05-01", "2023-05-22", "2023-06-12", "2023-06-19", "2023-07-03",
    "2023-07-20", "2023-08-07", "2023-08-21", "2023-10-16", "2023-11-06",
    "2023-11-13", "2023-12-08", "2023-12-25",
    # 2024
    "2024-01-01", "2024-01-08", "2024-03-25", "2024-03-28", "2024-03-29",
    "2024-05-01", "2024-05-13", "2024-06-03", "2024-06-10", "2024-07-01",
    "2024-07-20", "2024-08-07", "2024-08-19", "2024-10-14", "2024-11-04",
    "2024-11-11", "2024-12-08", "2024-12-25",
    # 2026
    "2026-01-01", "2026-01-12", "2026-03-23", "2026-04-02", "2026-04-03",
    "2026-05-01", "2026-05-18", "2026-06-08", "2026-06-15", "2026-06-29",
    "2026-07-20", "2026-08-07", "2026-08-17",
)


@dataclass(frozen=True)
class EvaluationMask:
    """Sesiones validas + la razon de cada exclusion."""

    valid: tuple[date, ...]
    excluded: dict[str, tuple[date, ...]] = field(default_factory=dict)
    source: str = ""
    flat_ohlc_pct: dict[date, float] = field(default_factory=dict)

    @property
    def sha256(self) -> str:
        payload = "\n".join(d.isoformat() for d in self.valid)
        return hashlib.sha256(payload.encode()).hexdigest()

    def __len__(self) -> int:
        return len(self.valid)

    def in_block(self, start: str | date, end: str | date) -> tuple[date, ...]:
        a = pd.Timestamp(start).date() if isinstance(start, str) else start
        b = pd.Timestamp(end).date() if isinstance(end, str) else end
        return tuple(d for d in self.valid if a <= d <= b)

    def to_dict(self) -> dict:
        return {
            "contract": "CTR-RESEARCH-EVALMASK-001",
            "source": self.source,
            "bars_per_session": BARS_PER_SESSION,
            "n_valid": len(self.valid),
            "sha256": self.sha256,
            "first": self.valid[0].isoformat() if self.valid else None,
            "last": self.valid[-1].isoformat() if self.valid else None,
            "excluded": {k: [d.isoformat() for d in v] for k, v in self.excluded.items()},
            "excluded_counts": {k: len(v) for k, v in self.excluded.items()},
            "flat_ohlc_pct": {d.isoformat(): float(v) for d, v in self.flat_ohlc_pct.items()},
        }


def colombia_holidays() -> set[str]:
    """Union de la lista canonica (solo 2025) y la derivada multi-anio."""
    out = set(EXTRA_COLOMBIA_HOLIDAYS)
    if CALENDAR.is_file():
        cal = json.loads(CALENDAR.read_text(encoding="utf-8"))
        for key, vals in cal.items():
            if key.startswith("holidays_") and key.endswith("_colombia"):
                out |= set(vals)
    return out


def usa_holidays() -> set[str]:
    """Federal US holidays for the observed data span, plus SSOT entries.

    The market is OTC, but the macro inputs include US releases and the frozen
    contract explicitly requires the Colombia ∪ USA calendar.  The generated
    federal calendar supplies years not yet present in the JSON SSOT.
    """
    out: set[str] = set()
    try:
        from pandas.tseries.holiday import USFederalHolidayCalendar

        dates = USFederalHolidayCalendar().holidays(
            start="2019-01-01", end="2030-12-31"
        )
        out.update(pd.Timestamp(d).date().isoformat() for d in dates)
    except ImportError:
        # pandas is a hard dependency of this module; retain a clear empty set
        # only for constrained import-time tooling.
        pass
    if CALENDAR.is_file():
        cal = json.loads(CALENDAR.read_text(encoding="utf-8"))
        for key, vals in cal.items():
            if key.startswith("holidays_") and key.endswith("_usa"):
                out |= set(vals)
    return out


def build_mask(seed: Path | None = None) -> EvaluationMask:
    """Construye la mascara desde la serie de 5 minutos ya reparada."""
    path = seed or DEFAULT_SEED
    df = pd.read_parquet(path)
    t = pd.to_datetime(df["time"])
    if "symbol" in df.columns:
        keep = df["symbol"].astype(str).str.upper().str.replace("/", "", regex=False) == "USDCOP"
        df, t = df[keep], t[keep]
    d = t.dt.date
    hours = t.dt.hour

    per_day = pd.DataFrame({"d": d, "h": hours}).groupby("d").agg(
        n=("h", "size"), h_min=("h", "min"), h_max=("h", "max"))

    # Validate the complete OHLC contract before counting a session.  A single
    # malformed bar invalidates the session; it must never become a zero return.
    numeric = df[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    invalid = (
        ~np.isfinite(numeric.to_numpy()).all(axis=1)
        | (numeric <= 0).any(axis=1)
        | (numeric["high"] < numeric[["open", "close"]].max(axis=1))
        | (numeric["low"] > numeric[["open", "close"]].min(axis=1))
        | (numeric["high"] < numeric["low"])
    )
    invalid_days = set(d[invalid].tolist())

    # Outlier suspicion is a quality flag, not a return-based selection rule:
    # the threshold is estimated from the preceding 1,560 bars only.
    close = numeric["close"]
    logret = np.log(close).diff()
    rolling_sigma = logret.rolling(1560, min_periods=100).std().shift(1)
    outlier_days = set(d[(logret.abs() > 8.0 * rolling_sigma).fillna(False)].tolist())

    flat_by_day = (
        (numeric["open"] == numeric["high"])
        & (numeric["high"] == numeric["low"])
        & (numeric["low"] == numeric["close"])
    ).groupby(d).mean().mul(100.0)

    holidays = colombia_holidays()
    holidays_usa = usa_holidays()
    excluded: dict[str, list[date]] = {"holiday": [], "us_holiday": [], "incomplete": [],
                                       "weekend": [], "out_of_window": [],
                                       "invalid_ohlc": [], "outlier_suspect": []}
    valid: list[date] = []

    for day, row in per_day.iterrows():
        if day.weekday() >= 5:
            excluded["weekend"].append(day)
        elif day.isoformat() in holidays:
            # Se excluye tenga 60 barras (relleno sintetico) o 3 (residuo): en ambos casos
            # el mercado colombiano estuvo cerrado.
            excluded["holiday"].append(day)
        elif day.isoformat() in holidays_usa:
            excluded["us_holiday"].append(day)
        elif not (SESSION_LO <= row.h_min and row.h_max <= SESSION_HI):
            excluded["out_of_window"].append(day)
        elif day in invalid_days:
            excluded["invalid_ohlc"].append(day)
        elif day in outlier_days:
            excluded["outlier_suspect"].append(day)
        elif row.n < BARS_PER_SESSION:
            excluded["incomplete"].append(day)
        else:
            valid.append(day)

    return EvaluationMask(
        valid=tuple(sorted(valid)),
        excluded={k: tuple(sorted(v)) for k, v in excluded.items() if v},
        source=path.relative_to(REPO).as_posix(),
        flat_ohlc_pct={d: float(flat_by_day.get(d, 0.0)) for d in per_day.index},
    )


def main() -> int:  # pragma: no cover - CLI
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, help="escribe el JSON de la mascara aqui")
    a = ap.parse_args()

    mask = build_mask()
    info = mask.to_dict()
    print(f"mascara: {info['n_valid']} sesiones validas  "
          f"({info['first']} -> {info['last']})")
    print(f"sha256 : {info['sha256'][:16]}")
    for reason, n in sorted(info["excluded_counts"].items(), key=lambda kv: -kv[1]):
        print(f"  excluidas por {reason:14s}: {n}")

    try:
        import yaml
        blocks = yaml.safe_load(PARTITION.read_text(encoding="utf-8"))["blocks"]
        print("\npor bloque de la particion:")
        for name, b in blocks.items():
            n = len(mask.in_block(b["start"], b["end"]))
            print(f"  {name:12s} {n:5d} validas de {b['sessions']:5d} "
                  f"({100 * n / b['sessions']:.1f}%)")
    except Exception as exc:  # noqa: BLE001
        print(f"(no se pudo leer la particion: {exc})")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        print(f"\nescrito {a.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
