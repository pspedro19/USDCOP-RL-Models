#!/usr/bin/env python3
"""Paridad camino-legacy vs motor de políticas (BL-47 R6/R7, criterio de corte).

El strangler NO apaga nada hasta que esta comparación sea verde: la serie de
exposición en FECHA DE DECISIÓN debe ser EXACTAMENTE igual (float64, sin
tolerancia) a la que produce el código congelado. Si difiere, la migración
cambió una decisión => es modelado, se para y se reporta (guardarraíl BL-47).

Cada comprobación importa el productor congelado y lo ejecuta; ninguna
reimplementa la estrategia (reimplementarla haría que la paridad se comparase
contra sí misma).

Datos: seeds locales / snapshot del activo. Sin Docker, sin DB, sin red.
Si falta el dato, sale con código 2 (SKIP honesto), nunca inventa una serie.

Uso:
    python scripts/validation/check_policy_parity.py                 # todas
    python scripts/validation/check_policy_parity.py --policy btc_hodl_b1
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contracts.policy import PolicyContext  # noqa: E402
from src.strategies.policies.loader import (  # noqa: E402
    build_policy,
    load_all_policy_specs,
    load_policy_spec,
    policy_specs_dir,
)

np = None
pd = None


class DataUnavailable(RuntimeError):
    """El insumo congelado no está en disco. SKIP honesto, jamás un sustituto."""


def _ensure_numeric_runtime() -> None:
    """Import heavy parity dependencies only when a policy is actually eligible."""
    global np, pd
    if np is not None and pd is not None:
        return
    try:
        import numpy as numpy_runtime
        import pandas as pandas_runtime
    except ImportError as exc:
        raise DataUnavailable(f"runtime de paridad no disponible: {exc}") from exc
    np = numpy_runtime
    pd = pandas_runtime


def _policy_exposures(spec: dict, snapshots: list[dict | None],
                      as_of: list[str]) -> np.ndarray:
    """Exposición en fecha de decisión, barra a barra, por el motor único.

    ``None`` = barra sin snapshot válido (warmup/feature no finita): el runner
    no emite decisión y la exposición es 0 — exactamente lo que el camino
    legacy obtiene de un NaN propagado a fillna(0).
    """
    policy = build_policy(spec)
    out = np.zeros(len(snapshots), dtype=float)
    for i, snap in enumerate(snapshots):
        if snap is None:
            continue
        decision = policy.evaluate(snap, PolicyContext(as_of=as_of[i]))
        out[i] = decision.target_exposure
    return out


def _rows(df: pd.DataFrame, features: list[str]) -> list[dict | None]:
    values = {f: df[f].to_numpy(float) for f in features}
    rows: list[dict | None] = []
    for i in range(len(df)):
        row = {f: float(values[f][i]) for f in features}
        rows.append(None if any(not math.isfinite(v) for v in row.values()) else row)
    return rows


# --------------------------------------------------------------------------- SPX
def parity_spx500_ma200(spec: dict):
    pkg = ROOT / "src" / "strategies" / "spx500_regime_gated_v1"
    if str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))
    try:
        from src.strategies.spx500_regime_gated_v1.load_real import load_real
        df = load_real()
    except Exception as exc:  # noqa: BLE001
        raise DataUnavailable(f"snapshot SPX no disponible: {exc}") from exc

    from src.features.spx500_ma200 import compute_ma_200

    close = df["close"].astype(float)
    # Legacy (profitability_adapters.spx500 / publish_spx500_bundles, ANTES del
    # np.roll de ejecución): la exposición decidida en t.
    #
    # La media YA NO se calcula aquí. Antes esta línea era
    # `close.rolling(200, min_periods=200).mean()` — idéntica a la del publisher
    # legacy, pero por casualidad y no por contrato: dos definiciones que coinciden
    # hoy pueden divergir mañana sin que nada lo note, y `ma_200` no estaba
    # catalogada ni tenía productor declarado. Ahora el harness CONSUME el productor
    # único (`spx500.ma_200`, catálogo BL-39) — que es lo que hace de esta paridad
    # una prueba de la feature publicada y no de una copia local suya.
    ma200 = compute_ma_200(close)
    legacy = (close > ma200).astype(float).to_numpy()

    frame = pd.DataFrame({"close": close.to_numpy(float), "ma_200": ma200.to_numpy(float)})
    stamps = [str(t)[:10] for t in pd.to_datetime(df["timestamp"])]
    engine = _policy_exposures(spec, _rows(frame, ["close", "ma_200"]), stamps)
    return legacy, engine


# --------------------------------------------------------------------------- Oro
def parity_gold_trend_simple(spec: dict):
    seed = ROOT / "seeds/latest/xauusd_daily_ohlcv.parquet"
    if not seed.is_file():
        raise DataUnavailable(f"falta {seed}")
    from scripts.analysis.gold_trend_simple import simulate
    from src.gold_rl.indicators import build_daily_features

    df = pd.read_parquet(seed).sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"])
    feat = build_daily_features(df)
    legacy_run = simulate(feat)              # camino A: el del bundle publicado
    # simulate() devuelve la posición YA ejecutada (shift(1)); la decisión de t
    # es la posición de t+1.
    legacy = np.roll(legacy_run["position"].to_numpy(float), -1)
    legacy[-1] = np.nan                      # última decisión no observable

    from src.features.xauusd_trend_smas import build_trend_smas

    d = legacy_run.reset_index(drop=True)
    # Las SMA YA NO se recalculan aqui. Antes esta funcion tenia su propia copia de
    # `close.rolling(w).mean()` para 63/126/252 -- identica a la del voto legacy pero
    # por casualidad, no por contrato, y ninguna declarada como feature. Ahora consume
    # el productor UNICO (`xauusd.sma_*`, catalogo BL-39), que es lo que convierte esta
    # paridad en una prueba de la feature publicada y no de una copia local suya.
    con_smas = build_trend_smas(d)
    frame = pd.DataFrame({
        "close": d["close"].to_numpy(float),
        "sma_63": con_smas["sma_63"].to_numpy(float),
        "sma_126": con_smas["sma_126"].to_numpy(float),
        "sma_252": con_smas["sma_252"].to_numpy(float),
        "realized_vol_20": d["realized_vol_20"].to_numpy(float),
    })
    stamps = [str(t)[:10] for t in d["time"]]
    engine = _policy_exposures(
        spec, _rows(frame, list(frame.columns)), stamps)
    mask = np.isfinite(legacy)
    return legacy[mask], engine[mask]


# --------------------------------------------------------------------------- BTC
def parity_btc_hodl(spec: dict):
    seed = ROOT / "seeds/latest/btcusdt_daily_ohlcv.parquet"
    if not seed.is_file():
        raise DataUnavailable(f"falta {seed}")
    from src.btc_strategy.indicators import build_daily_features
    from src.btc_strategy.strategies import STRATEGIES, build_positions

    df = pd.read_parquet(seed).sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"])
    feat = build_daily_features(df)
    d = build_positions(feat, STRATEGIES["btc_hodl_b1"][1]).reset_index(drop=True)

    legacy = np.roll(d["position"].to_numpy(float), -1)
    legacy[-1] = np.nan

    cols = {"realized_vol_20": d["realized_vol_20"].to_numpy(float)}
    # El track BTC no emite regime_risk_mult (vol_target_size cae en su default
    # 1.0). Se incluye SOLO si el productor la trae, para no cambiar semántica.
    if "regime_risk_mult" in d.columns:
        cols["regime_risk_mult"] = d["regime_risk_mult"].to_numpy(float)
    frame = pd.DataFrame(cols)
    stamps = [str(t)[:10] for t in d["time"]]
    engine = _policy_exposures(spec, _rows(frame, list(frame.columns)), stamps)
    mask = np.isfinite(legacy)
    return legacy[mask], engine[mask]


CHECKS = {
    "spx500_daily_ma200_v1": parity_spx500_ma200,
    "gold_trend_simple": parity_gold_trend_simple,
    "btc_hodl_b1": parity_btc_hodl,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    target = ap.add_mutually_exclusive_group()
    target.add_argument("--policy", default=None, help="id de una sola política")
    target.add_argument(
        "--ci-eligible",
        action="store_true",
        help="verifica estrictamente solo PARITY_GREEN/CUTOVER; nunca convierte SKIP en verde",
    )

    ap.add_argument(
        "--allow-empty",
        action="store_true",
        help=(
            "acepta 0 specs elegibles como verde. Se exige explícito para que un gate "
            "sin sujeto sea una DECISIÓN visible en el llamador y no un silencio"
        ),
    )
    args = ap.parse_args(argv)

    if args.ci_eligible:
        specs = load_all_policy_specs()
        if not specs:
            print("[FAIL] registro de policies vacío — directorio/loader roto, no cero gobernado")
            return 1
        targets = [
            str(spec["id"])
            for spec in specs
            if (spec.get("migration") or {}).get("status") in {"PARITY_GREEN", "CUTOVER"}
        ]
        if not targets:
            if not CHECKS:
                print("[FAIL] registro de arneses vacío — paridad no observable")
                return 1
            # Cero sujetos NO es verde. El mensaje de abajo era honesto —decía que no
            # había verificado nada— pero CI lee el EXIT CODE, no el texto: el paso
            # corría, salía en verde y no comprobaba una sola policy. Estado real que lo
            # produjo: {PARITY_PENDING: 3, SPEC_ONLY: 1}, cero elegibles, y así habría
            # seguido indefinidamente.
            #
            # Ahora el vacío es rojo salvo que el llamador lo DECLARE con `--allow-empty`.
            # No se trata de tener sujeto a toda costa: promover una policy para darle
            # trabajo al gate sería la trampa que esto denuncia.
            if not args.allow_empty:
                print(
                    "[FAIL] 0 specs elegibles y --allow-empty no fue declarado. "
                    "Un gate sin sujeto no prueba nada: o hay una policy en "
                    "PARITY_GREEN/CUTOVER, o el llamador declara que hoy no la hay"
                )
                return 1
            print(
                "[OK] 0 specs elegibles — nada verificado, VACÍO DECLARADO por "
                "--allow-empty (SPEC_ONLY/PARITY_PENDING son inertes)"
            )
            return 0
    else:
        targets = [args.policy] if args.policy else list(CHECKS)

    skipped, failed = 0, 0
    for policy_id in targets:
        if policy_id not in CHECKS:
            print(f"[FAIL] {policy_id}: sin arnés de paridad (¿es SPEC_ONLY?)")
            failed += 1
            continue
        try:
            _ensure_numeric_runtime()
        except DataUnavailable as exc:
            print(f"[FAIL] {policy_id}: {exc}")
            failed += 1
            continue
        spec = load_policy_spec(policy_specs_dir() / f"{policy_id}.yaml")
        try:
            legacy, engine = CHECKS[policy_id](spec)
        except DataUnavailable as exc:
            if args.ci_eligible:
                print(f"[FAIL] {policy_id}: {exc}")
                failed += 1
            else:
                print(f"[SKIP] {policy_id}: {exc}")
                skipped += 1
            continue
        # La ventana de calentamiento se compara aparte: ahí el camino congelado
        # puede tratar un NaN como voto negativo mientras la política falla
        # cerrada (divergencia declarada en el spec, no un fallo del motor).
        warmup = int(spec["inputs"].get("warmup_bars") or 0)
        if warmup:
            warm = np.flatnonzero(legacy[:warmup] != engine[:warmup])
            if warm.size:
                print(f"[WARN] {policy_id}: {warm.size} barras divergen DENTRO del "
                      f"calentamiento (<{warmup}) — divergencia declarada en el spec")
            legacy, engine = legacy[warmup:], engine[warmup:]
        diff = np.flatnonzero(legacy != engine)
        if diff.size == 0:
            print(f"[OK]   {policy_id}: {len(legacy)} barras, exposición IDÉNTICA (float64)")
            continue
        i = int(diff[0])
        print(f"[FAIL] {policy_id}: {diff.size}/{len(legacy)} barras divergen. "
              f"Primera en idx={i}: legacy={legacy[i]!r} motor={engine[i]!r}")
        print("       Divergencia => es MODELADO, no migración: parar y escalar (BL-47).")
        failed += 1

    if failed:
        return 1
    if skipped and skipped == len(targets):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
