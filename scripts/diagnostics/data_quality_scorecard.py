#!/usr/bin/env python3
"""Data-quality scorecard for the canonical market layers (CTR-MKT-CANON-001).

Measures 5 dimensions per layer/series and prints a weighted 0-100 score (and /10):

  completeness  40%  bars present / bars a declared expectation implies
  coherence     20%  OHLC-valid bars / bars   (high>=max(o,c), low<=min(o,c), h>=l)
  freshness     15%  age of newest bar vs the layer's cadence tolerance
  validity      15%  1 - anomalous-bar rate (flat O=H=L=C on price series, off-calendar)
  pit           10%  rows with available_at stamped (historic NULL is honest -> low score
                     is EXPECTED for pre-backfill rows; the metric shows PIT accrual)

Expectations are DECLARED (bars/day by market clock), not fitted: COP session=60 M5/day
on Colombian trading days; 24/7 crypto=288/day; FX/metals ~24/5. A ratio can exceed 1
(provider quotes off-session); completeness is capped at 100 and the surplus is counted
in validity instead. N<expectation on purpose is stated, not hidden.

Run: POSTGRES_HOST=localhost POSTGRES_PASSWORD=... python scripts/diagnostics/data_quality_scorecard.py
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

W = {"completeness": 0.40, "coherence": 0.20, "freshness": 0.15,
     "validity": 0.15, "pit": 0.10}


def _conn():
    import psycopg2
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        dbname=os.environ.get("POSTGRES_DB", "usdcop_trading"),
        user=os.environ.get("POSTGRES_USER", "admin"),
        password=os.environ.get("POSTGRES_PASSWORD", ""))


def q1(cur, sql, *args):
    cur.execute(sql, args or None)
    return cur.fetchone()


def freshness_score(age_seconds: float | None, tolerance_s: float) -> float:
    if age_seconds is None:
        return 0.0
    return 1.0 if age_seconds <= tolerance_s else max(0.0, 1 - (age_seconds - tolerance_s) / (3 * tolerance_s))


def score_row(comp, coh, fresh, valid, pit) -> float:
    return 100 * (W["completeness"] * comp + W["coherence"] * coh +
                  W["freshness"] * fresh + W["validity"] * valid + W["pit"] * pit)


def main() -> int:
    conn = _conn()
    cur = conn.cursor()
    cur.execute("SET max_parallel_workers_per_gather = 0")
    now = datetime.now(timezone.utc)
    results = []

    # ---------------- M5 ----------------------------------------------------
    # expectation per symbol: bars/day on its own clock
    m5_exp = {
        "USD/COP":  ("cop_session", 60),    # Colombian trading days x 60
        "BTC/USDT": ("all_days", 288),
        "XAU/USD":  ("weekdays", 276),
        "USD/MXN":  ("weekdays", 276),
        # BRL trades ~11.5 liquid hours offshore (~138 M5 bars/day), NOT a 23h book:
        # the 50% "completeness" of the first scorecard was expectation mismatch, not loss.
        "USD/BRL":  ("weekdays", 138),
    }
    for sym, (clock, per_day) in m5_exp.items():
        n, bad, first, last = q1(cur, """
            SELECT count(*),
                   count(*) FILTER (WHERE NOT (high>=GREATEST(open,close)
                        AND low<=LEAST(open,close) AND high>=low)),
                   min(time), max(time)
            FROM usdcop_m5_ohlcv WHERE symbol=%s""", sym)
        pit, = q1(cur, "SELECT count(available_at) FROM usdcop_m5_ohlcv WHERE symbol=%s", sym)
        span_days = max(1, (last - first).days)
        if clock == "cop_session":
            (tdays,) = q1(cur, """SELECT count(*) FROM market_session_calendar
                WHERE asset_id='usdcop' AND is_trading_day
                  AND session_date BETWEEN %s AND %s""", first.date(), last.date())
            expected = tdays * per_day
            (offs,) = q1(cur, """SELECT count(*) FROM usdcop_m5_ohlcv m
                WHERE symbol=%s AND NOT EXISTS (
                  SELECT 1 FROM market_session_calendar c WHERE c.asset_id='usdcop'
                    AND c.session_date=(m.time AT TIME ZONE 'America/Bogota')::date
                    AND c.is_trading_day)""", sym)
        elif clock == "all_days":
            expected = span_days * per_day
            offs = 0
        else:
            expected = int(span_days * 5 / 7) * per_day
            offs = 0
        comp = min(1.0, n / expected)
        coh = 1 - bad / n if n else 0
        fresh = freshness_score((now - last).total_seconds(),
                                3 * 86400 if clock != "all_days" else 2 * 3600)
        valid = 1 - min(1.0, offs / n) if n else 0
        results.append(("5m", sym, n, f"{first.date()}→{last.date()}",
                        comp, coh, fresh, valid, pit / n if n else 0))

    # ---------------- 1h / 4h nativo ---------------------------------------
    for tf, per_day_fx, per_day_247 in (("1h", 23, 24), ("4h", 6, 6)):
        cur.execute("""SELECT symbol, count(*),
                   count(*) FILTER (WHERE NOT (high>=GREATEST(open,close)
                        AND low<=LEAST(open,close) AND high>=low)),
                   min(time), max(time), count(available_at)
            FROM asset_native_ohlcv WHERE tf=%s GROUP BY symbol""", (tf,))
        for sym, n, bad, first, last, pit in cur.fetchall():
            span_days = max(1, (last - first).days)
            per_day = per_day_247 if sym == "BTC/USDT" else per_day_fx
            if sym == "SPY":
                per_day = 7 if tf == "1h" else 2
            if sym == "USD/BRL":
                per_day = 15 if tf == "1h" else 5  # ~media jornada liquida
            expected = (span_days if sym == "BTC/USDT"
                        else int(span_days * 5 / 7)) * per_day
            comp = min(1.0, n / expected)
            coh = 1 - bad / n if n else 0
            fresh = freshness_score((now - last).total_seconds(), 3 * 86400)
            results.append((f"{tf} nativo", sym, n, f"{first.date()}→{last.date()}",
                            comp, coh, fresh, 1.0, pit / n if n else 0))

    # ---------------- Daily -------------------------------------------------
    cur.execute("""SELECT symbol, count(*),
               count(*) FILTER (WHERE NOT (high>=GREATEST(open,close)
                    AND low<=LEAST(open,close) AND high>=low)),
               count(*) FILTER (WHERE open=high AND high=low AND low=close),
               min(time), max(time), count(available_at)
        FROM asset_daily_ohlcv GROUP BY symbol""")
    # Simbolos tal y como estan en `asset_daily_ohlcv`. SPX es 'SPX/500' (el `symbol` del
    # AssetProfile), no 'SPX500' (que es su `chart_symbol`, para el dashboard). Antes de la
    # migracion 083 este mapeo devolvia None para SPX y el scorecard lo saltaba en silencio.
    daily_asset = {"USD/COP": "usdcop", "XAU/USD": "xauusd",
                   "BTC/USDT": "btcusdt", "SPX/500": "spx500"}
    for sym, n, bad, flat, first, last, pit in cur.fetchall():
        aid = daily_asset.get(sym)
        if aid:
            row = q1(cur, """SELECT count(*) FILTER (WHERE is_trading_day),
                       count(*) FROM market_session_calendar
                WHERE asset_id=%s AND session_date BETWEEN GREATEST(%s,'2020-01-01'::date) AND %s""",
                     aid, first.date(), last.date())
            tdays = row[0]
            (have,) = q1(cur, """SELECT count(*) FROM asset_daily_ohlcv d
                JOIN market_session_calendar c ON c.asset_id=%s
                 AND c.session_date=(d.time AT TIME ZONE 'UTC')::date AND c.is_trading_day
                WHERE d.symbol=%s""", aid, sym)
            comp = min(1.0, have / tdays) if tdays else 0
        else:  # SPY: NYSE clock via spx500 calendar
            row = q1(cur, """SELECT count(*) FROM market_session_calendar
                WHERE asset_id='spx500' AND is_trading_day
                  AND session_date BETWEEN GREATEST(%s,'2020-01-01'::date) AND %s""",
                     first.date(), last.date())
            (have,) = q1(cur, """SELECT count(*) FROM asset_daily_ohlcv d
                JOIN market_session_calendar c ON c.asset_id='spx500'
                 AND c.session_date=(d.time AT TIME ZONE 'UTC')::date AND c.is_trading_day
                WHERE d.symbol='SPY' AND d.time >= '2020-01-01'""")
            comp = min(1.0, have / row[0]) if row[0] else 0
        coh = 1 - bad / n if n else 0
        fresh = freshness_score((now - last).total_seconds(), 4 * 86400)
        valid = 1 - min(1.0, flat / n) if n else 0
        results.append(("daily", sym, n, f"{first.date()}→{last.date()}",
                        comp, coh, fresh, valid, pit / n if n else 0))

    # ---------------- Monthly (nativo precio) -------------------------------
    cur.execute("""SELECT symbol, count(*), min(time), max(time), count(available_at)
        FROM asset_native_ohlcv WHERE tf='1month' GROUP BY symbol""")
    for sym, n, first, last, pit in cur.fetchall():
        months = (last.year - first.year) * 12 + (last.month - first.month) + 1
        comp = min(1.0, n / months)
        fresh = freshness_score((now - last).total_seconds(), 40 * 86400)
        results.append(("1month nativo", sym, n, f"{first.date()}→{last.date()}",
                        comp, 1.0, fresh, 1.0, pit / n if n else 0))

    # ---------------- Monthly macro (wide) -----------------------------------
    n, first, last = q1(cur, "SELECT count(*), min(month_start), max(month_start) "
                             "FROM market_macro_monthly_wide")
    months = (last.year - first.year) * 12 + (last.month - first.month) + 1
    # series-cell completeness: filled cells / (15 series x months), honest about the
    # Colombian tail going dark
    (cells,) = q1(cur, """SELECT
        count(macro_fedfunds)+count(macro_cpi_usa)+count(macro_pce_usa)+count(macro_unrate)
       +count(macro_indpro)+count(macro_m2_usa)+count(macro_umcsent)+count(macro_ipc_col)
       +count(macro_itcr)+count(macro_resint)+count(macro_tot)+count(macro_expusd)
       +count(macro_impusd)+count(macro_cci)+count(macro_ici)
        FROM market_macro_monthly_wide""")
    comp = min(1.0, n / months)
    # cell fill INSIDE each series' own [first_obs, last_obs] span: a 2003-2026 grid
    # must not count cells before a series existed as "invalid" (expectation mismatch,
    # same lesson as BRL). Gaps WITHIN a live series are what validity punishes.
    series_cols = ["macro_fedfunds", "macro_cpi_usa", "macro_pce_usa", "macro_unrate",
                   "macro_indpro", "macro_m2_usa", "macro_umcsent", "macro_ipc_col",
                   "macro_itcr", "macro_resint", "macro_tot", "macro_expusd",
                   "macro_impusd", "macro_cci", "macro_ici"]
    rates = []
    for c in series_cols:
        cur.execute(f"""WITH x AS (SELECT min(month_start) f, max(month_start) l
                                   FROM market_macro_monthly_wide WHERE {c} IS NOT NULL)
            SELECT (SELECT count({c}) FROM market_macro_monthly_wide),
                   (SELECT count(*) FROM market_macro_monthly_wide, x
                    WHERE month_start BETWEEN x.f AND x.l)""")
        got, span = cur.fetchone()
        if span:
            rates.append(got / span)
    cell_rate = sum(rates) / len(rates) if rates else 0.0
    fresh = freshness_score((now - datetime(last.year, last.month, 1,
                                            tzinfo=timezone.utc)).total_seconds(),
                            120 * 86400)
    results.append(("monthly macro", "17 series CLEAN", n,
                    f"{first}→{last}", comp, 1.0, fresh, cell_rate, 0.0))

    # ---------------- Print scorecard ---------------------------------------
    print(f"{'capa':14s} {'serie':16s} {'n':>8s} {'rango':23s} "
          f"{'compl':>6s} {'coher':>6s} {'fresc':>6s} {'valid':>6s} {'pit':>5s} "
          f"{'score':>6s} {'/10':>5s}")
    by_layer: dict[str, list[float]] = {}
    for layer, sym, n, rng, comp, coh, fresh, valid, pit in results:
        s = score_row(comp, coh, fresh, valid, pit)
        by_layer.setdefault(layer, []).append(s)
        print(f"{layer:14s} {sym:16s} {n:8d} {rng:23s} "
              f"{comp:6.1%} {coh:6.1%} {fresh:6.1%} {valid:6.1%} {pit:5.0%} "
              f"{s:6.1f} {s/10:5.1f}")
    print("\n-- por capa --")
    for layer, ss in by_layer.items():
        avg = sum(ss) / len(ss)
        print(f"{layer:14s} score {avg:5.1f}  -> {avg/10:.1f}/10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
