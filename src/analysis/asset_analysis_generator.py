"""
Multi-Asset Weekly Analysis Generator
=====================================
Produces per-week weekly/daily analysis for the onboarded science-stack assets
(Gold, BTC) using ONLY real data:

  - Real daily OHLCV (seeds/latest/<asset>_daily_ohlcv.parquet)
  - Real technical indicators computed from that price series
  - Real strategy positioning (published backtest trades, e.g. gold_trend_b2)
  - Real news scraped live from GDELT DOC 2.0 (public API, no key)

It emits the SAME WeeklyViewData JSON contract the /analysis page already
consumes, written to the per-asset namespace
(public/data/analysis/<asset>/weekly_YYYY_WXX.json) + a per-asset
analysis_index.json — so the frontend asset selector renders Gold/BTC exactly
like USD/COP with no component changes.

Fully driven by config/analysis/analysis_assets.yaml (SSOT). No per-asset code.

USD/COP is intentionally NOT handled here — it keeps its richer macro-driven
LangGraph pipeline (generate_weekly_analysis.py) served from the legacy root.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.analysis.news_sources import (
    AssetNewsFetcher,
    NewsQuery,
    build_news_sources,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _sanitize(obj: Any) -> Any:
    """Recursively convert NaN/Inf → None so the JSON is always valid."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 6)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


@dataclass
class AssetProfile:
    asset_id: str
    symbol: str
    chart_symbol: str
    display_name: str
    asset_class: str
    ohlcv_seed: str
    ohlcv_symbol: str | None
    strategy_id: str
    strategy_display: str
    annualization_days: int
    google_news_query: str
    news_query: str
    news_language: str
    price_unit: str


class AssetAnalysisGenerator:
    def __init__(self, config_path: str | None = None):
        config_path = config_path or str(PROJECT_ROOT / "config/analysis/analysis_assets.yaml")
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.output_root = PROJECT_ROOT / self.config["output"]["root_dir"]
        self.bundle_root = PROJECT_ROOT / self.config["strategies"]["bundle_root"]
        self.news_cfg = self.config.get("news", {})
        # Pluggable, SSOT-driven news module (ports & adapters). Built once and
        # reused across weeks so any per-source pacing/state is shared.
        self.news_fetcher = AssetNewsFetcher(
            sources=build_news_sources(self.news_cfg),
            strategy=self.news_cfg.get("strategy", "first_nonempty"),
        )
        self._ohlcv_cache: dict[str, pd.DataFrame] = {}
        self._trades_cache: dict[str, list[dict]] = {}

    # ------------------------------------------------------------------ profiles
    def profile(self, asset_id: str) -> AssetProfile:
        assets = self.config["assets"]
        if asset_id not in assets:
            raise KeyError(
                f"Unknown analysis asset '{asset_id}'. Known: {sorted(assets)}"
            )
        a = assets[asset_id]
        return AssetProfile(
            asset_id=asset_id,
            symbol=a["symbol"],
            chart_symbol=a["chart_symbol"],
            display_name=a["display_name"],
            asset_class=a["asset_class"],
            ohlcv_seed=a["ohlcv_seed"],
            ohlcv_symbol=a.get("ohlcv_symbol"),
            strategy_id=a["strategy_id"],
            strategy_display=a["strategy_display"],
            annualization_days=int(a.get("annualization_days", 252)),
            google_news_query=a.get("google_news_query", a["news_query"]),
            news_query=a["news_query"],
            news_language=a.get("news_language", "english"),
            price_unit=a.get("price_unit", "USD"),
        )

    def known_assets(self) -> list[str]:
        return sorted(self.config["assets"].keys())

    # ------------------------------------------------------------------ data load
    def _load_ohlcv(self, p: AssetProfile) -> pd.DataFrame:
        if p.asset_id in self._ohlcv_cache:
            return self._ohlcv_cache[p.asset_id]
        df = pd.read_parquet(PROJECT_ROOT / p.ohlcv_seed)
        if p.ohlcv_symbol and "symbol" in df.columns:
            df = df[df["symbol"] == p.ohlcv_symbol].copy()
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.sort_values("time").reset_index(drop=True)
        df["d"] = df["time"].dt.date
        self._ohlcv_cache[p.asset_id] = df
        return df

    def _load_trades(self, p: AssetProfile) -> list[dict]:
        if p.strategy_id in self._trades_cache:
            return self._trades_cache[p.strategy_id]
        trades: list[dict] = []
        bt_dir = self.bundle_root / p.strategy_id / "backtests"
        if bt_dir.exists():
            for tf in sorted(bt_dir.glob("*/trades_*.json")):
                try:
                    with open(tf, encoding="utf-8") as f:
                        raw = json.load(f)
                    items = raw if isinstance(raw, list) else (raw.get("trades") or [])
                    trades.extend(items)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Could not read trades %s: %s", tf, e)
        # normalise timestamps
        for t in trades:
            t["_entry"] = pd.to_datetime(t.get("timestamp"), utc=True, errors="coerce")
            t["_exit"] = pd.to_datetime(t.get("exit_timestamp"), utc=True, errors="coerce")
        self._trades_cache[p.strategy_id] = trades
        return trades

    # ------------------------------------------------------------------ technicals
    @staticmethod
    def _rsi_wilder(close: pd.Series, period: int = 14) -> float | None:
        if len(close) < period + 1:
            return None
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        # Wilder's smoothing (alpha = 1/period)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100 - 100 / (1 + rs))

    @staticmethod
    def _macd(close: pd.Series) -> tuple[float | None, float | None, float | None]:
        if len(close) < 35:
            return None, None, None
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        sig = macd.ewm(span=9, adjust=False).mean()
        hist = macd - sig
        return float(macd.iloc[-1]), float(sig.iloc[-1]), float(hist.iloc[-1])

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> float | None:
        if len(df) < period + 1:
            return None
        h, l, c = df["high"], df["low"], df["close"]
        prev_c = c.shift(1)
        tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        return float(tr.ewm(alpha=1 / period, adjust=False).mean().iloc[-1])

    def _technical_analysis(self, hist: pd.DataFrame, p: AssetProfile) -> dict:
        """Technical block computed from the asset's own price up to week end."""
        close = hist["close"].astype(float)
        last = float(close.iloc[-1])
        sma20 = float(close.tail(20).mean()) if len(close) >= 20 else None
        sma50 = float(close.tail(50).mean()) if len(close) >= 50 else None
        rsi = self._rsi_wilder(close)
        macd, macd_sig, macd_hist = self._macd(close)
        atr = self._atr(hist)
        win = hist.tail(20)
        resistance = float(win["high"].max()) if len(win) else last
        support = float(win["low"].min()) if len(win) else last

        # Trend label
        trend = "neutral"
        if sma20 and sma50:
            if last > sma20 > sma50:
                trend = "alcista"
            elif last < sma20 < sma50:
                trend = "bajista"
            elif last > sma20:
                trend = "alcista_debil"
            else:
                trend = "bajista_debil"

        signal_state = "neutral"
        if rsi is not None:
            if rsi >= 70:
                signal_state = "sobrecompra"
            elif rsi <= 30:
                signal_state = "sobreventa"

        # No-trade zone around mid (ATR band)
        band = atr if atr else (resistance - support) * 0.15
        no_trade_zone = {"low": round(last - band * 0.5, 4), "high": round(last + band * 0.5, 4)}

        # Two concrete scenarios anchored on real S/R levels
        scenarios = [
            {
                "name": "Continuación de tendencia",
                "direction": "long" if trend.startswith("alcista") else "short",
                "trigger": round(resistance, 4) if trend.startswith("alcista") else round(support, 4),
                "target": round(resistance + band, 4) if trend.startswith("alcista") else round(support - band, 4),
                "invalidation": round(support, 4) if trend.startswith("alcista") else round(resistance, 4),
                "rationale": f"Ruptura {'alcista' if trend.startswith('alcista') else 'bajista'} con RSI={rsi:.0f}"
                if rsi is not None else "Ruptura de nivel clave",
            },
            {
                "name": "Reversión a la media",
                "direction": "short" if trend.startswith("alcista") else "long",
                "trigger": round(resistance, 4) if trend.startswith("alcista") else round(support, 4),
                "target": round(sma20, 4) if sma20 else round(last, 4),
                "invalidation": round(resistance + band, 4) if trend.startswith("alcista") else round(support - band, 4),
                "rationale": f"Precio {'extendido sobre' if trend.startswith('alcista') else 'bajo'} SMA20",
            },
        ]

        return {
            "as_of": str(hist["d"].iloc[-1]),
            "last_price": round(last, 4),
            "trend": trend,
            "signal": signal_state,
            "indicators": {
                "rsi_14": round(rsi, 2) if rsi is not None else None,
                "sma_20": round(sma20, 4) if sma20 else None,
                "sma_50": round(sma50, 4) if sma50 else None,
                "macd_line": round(macd, 4) if macd is not None else None,
                "macd_signal": round(macd_sig, 4) if macd_sig is not None else None,
                "macd_histogram": round(macd_hist, 4) if macd_hist is not None else None,
                "atr_14": round(atr, 4) if atr is not None else None,
            },
            "support_resistance": {
                "support": round(support, 4),
                "resistance": round(resistance, 4),
                "no_trade_zone": no_trade_zone,
            },
            "scenarios": scenarios,
        }

    # ------------------------------------------------------------------ signal
    def _signal_for_week(self, trades: list[dict], w_start: date, w_end: date, p: AssetProfile) -> dict:
        """Map the strategy's active position during the week into the h5 slot."""
        ws = pd.Timestamp(w_start, tz="UTC")
        we = pd.Timestamp(w_end, tz="UTC") + pd.Timedelta(days=1)
        active = [
            t for t in trades
            if pd.notna(t.get("_entry")) and t["_entry"] < we
            and (pd.isna(t.get("_exit")) or t["_exit"] >= ws)
        ]
        if not active:
            return {
                "h5": {"direction": "HOLD", "note": "Estrategia en efectivo (sin posición)"},
                "h1": {"direction": "N/A", "note": "N/A para este activo"},
            }
        t = active[-1]
        side = str(t.get("side", "")).upper()
        regime = t.get("regime")
        lev = t.get("leverage")
        return {
            "h5": {
                "direction": side if side in ("LONG", "SHORT") else "HOLD",
                "confidence": "media",
                "leverage": float(lev) if lev is not None else None,
                "predicted_return": float(t.get("pnl_pct")) if t.get("pnl_pct") is not None else None,
                "note": f"{p.strategy_display} · régimen: {regime}" if regime else p.strategy_display,
                "exit_reason": t.get("exit_reason"),
            },
            "h1": {"direction": "N/A", "note": "N/A para este activo"},
        }

    # ------------------------------------------------------------------ news
    def _fetch_news(self, p: AssetProfile, w_start: date, w_end: date) -> list[dict]:
        """Fetch real weekly articles via the pluggable news module (SSOT-driven).

        Sourcing/adapters/pacing live in src/analysis/news_sources.py; this method
        just builds the NewsQuery and delegates. Add sources or assets there / in
        config — no change here.
        """
        if not self.news_cfg.get("enabled", True):
            return []
        q = NewsQuery(
            asset_id=p.asset_id,
            google_query=p.google_news_query,
            gdelt_query=p.news_query,
            w_start=w_start,
            w_end=w_end,
        )
        return [a.to_dict() for a in self.news_fetcher.fetch(q)]

    @staticmethod
    def _cluster_news(arts: list[dict]) -> dict:
        by_source: dict[str, int] = {}
        for a in arts:
            src = a.get("source") or "unknown"
            by_source[src] = by_source.get(src, 0) + 1
        # Lightweight clustering by top source domains
        clusters = []
        for src, cnt in sorted(by_source.items(), key=lambda kv: -kv[1])[:6]:
            sample = [a for a in arts if (a.get("source") or "unknown") == src][:4]
            clusters.append({
                "theme": src,
                "count": cnt,
                "articles": [{"title": s.get("title"), "url": s.get("url")} for s in sample],
            })
        return {
            "total_articles": len(arts),
            "clusters": clusters,
            "source_breakdown": by_source,
        }

    # ------------------------------------------------------------------ narrative
    @staticmethod
    def _sentiment(change_pct: float | None) -> str:
        if change_pct is None:
            return "neutral"
        if change_pct > 0.5:
            return "bullish"
        if change_pct < -0.5:
            return "bearish"
        return "mixed"

    def _weekly_narrative(self, p: AssetProfile, wk: dict, ta: dict, sig: dict,
                          news: dict, w_start: date, w_end: date) -> str:
        o = wk
        h5 = sig["h5"]
        pos = h5["direction"]
        pos_txt = {"LONG": "posición larga", "SHORT": "posición corta", "HOLD": "en efectivo (sin exposición)"}.get(pos, pos)
        trend_txt = ta["trend"].replace("_", " ")
        rsi = ta["indicators"]["rsi_14"]
        lines = [
            f"### {p.display_name} — Semana {w_start.isocalendar()[1]:02d} ({w_start} → {w_end})",
            "",
            f"El cierre semanal de {p.symbol} fue **{o['close']:.2f} {p.price_unit}**, "
            f"un cambio de **{o['change_pct']:+.2f}%** en la semana "
            f"(rango {o['low']:.2f}–{o['high']:.2f}). "
            f"El sentimiento fue **{self._sentiment(o['change_pct'])}**.",
            "",
            "### Técnico",
            f"La tendencia es **{trend_txt}** con precio en {ta['last_price']:.2f}. "
            + (f"RSI(14)={rsi:.0f} ({ta['signal']}). " if rsi is not None else "")
            + f"Soporte {ta['support_resistance']['support']:.2f}, "
            f"resistencia {ta['support_resistance']['resistance']:.2f}.",
            "",
            "### Estrategia",
            f"La estrategia **{p.strategy_display}** estuvo **{pos_txt}** esta semana"
            + (f" (apalancamiento {h5['leverage']:.2f}x)." if h5.get("leverage") else ".")
            + (f" Régimen: {h5['note'].split('régimen: ')[-1]}." if "régimen" in (h5.get("note") or "") else ""),
            "",
            "### Noticias",
            f"Se recopilaron **{news['total_articles']} artículos** reales (GDELT) sobre {p.display_name} esta semana"
            + (f", principales fuentes: {', '.join(list(news['source_breakdown'])[:3])}." if news["source_breakdown"] else "."),
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------ per-week
    def generate_week(self, asset_id: str, iso_year: int, iso_week: int,
                      skip_news: bool = False) -> dict | None:
        p = self.profile(asset_id)
        df = self._load_ohlcv(p)
        w_start = date.fromisocalendar(iso_year, iso_week, 1)  # Monday
        w_end = w_start + timedelta(days=6)                    # Sunday

        wk_bars = df[(df["d"] >= w_start) & (df["d"] <= w_end)]
        if wk_bars.empty:
            logger.info("No %s bars for %s-W%02d — skipping", asset_id, iso_year, iso_week)
            return None

        hist = df[df["d"] <= w_end]  # everything up to week end (for indicators)
        prev = df[df["d"] < w_start]
        prev_close = float(prev["close"].iloc[-1]) if not prev.empty else float(wk_bars["open"].iloc[0])

        close = float(wk_bars["close"].iloc[-1])
        wk = {
            "open": round(float(wk_bars["open"].iloc[0]), 4),
            "high": round(float(wk_bars["high"].max()), 4),
            "low": round(float(wk_bars["low"].min()), 4),
            "close": round(close, 4),
            "change_pct": round((close / prev_close - 1) * 100, 4) if prev_close else 0.0,
        }

        ta = self._technical_analysis(hist, p)
        trades = self._load_trades(p)
        sig = self._signal_for_week(trades, w_start, w_end, p)

        arts = [] if skip_news else self._fetch_news(p, w_start, w_end)
        news = self._cluster_news(arts)

        # Daily entries (real per-day OHLCV + concise note)
        daily_entries = []
        wk_sorted = wk_bars.sort_values("time")
        pc = prev_close
        for _, r in wk_sorted.iterrows():
            d = r["d"]
            c = float(r["close"])
            chg = round((c / pc - 1) * 100, 4) if pc else 0.0
            daily_entries.append({
                "analysis_date": str(d),
                "iso_year": iso_year,
                "iso_week": iso_week,
                "day_of_week": d.weekday(),
                "headline": f"{p.display_name} {d}: cierre {c:.2f} ({chg:+.2f}%)",
                "summary_markdown": (
                    f"## {p.symbol} — {d}\n\n"
                    f"Cierre **{c:.2f} {p.price_unit}** ({chg:+.2f}%). "
                    f"Rango {float(r['low']):.2f}–{float(r['high']):.2f}."
                ),
                "usdcop_close": round(c, 4),
                "usdcop_change_pct": chg,
                "usdcop_high": round(float(r["high"]), 4),
                "usdcop_low": round(float(r["low"]), 4),
                "h1_signal": {},
                "h5_status": sig["h5"],
            })
            pc = c

        sentiment = self._sentiment(wk["change_pct"])
        h5 = sig["h5"]
        regime_label = "trend" if h5["direction"] in ("LONG", "SHORT") else "flat"
        macd_h = ta["indicators"]["macd_histogram"]

        view = {
            "asset_id": p.asset_id,
            "symbol": p.symbol,
            "chart_symbol": p.chart_symbol,
            "display_name": p.display_name,
            "iso_year": iso_year,
            "iso_week": iso_week,
            "week_start": str(w_start),
            "week_end": str(w_end),
            "weekly_summary": {
                "headline": f"{p.display_name} · Semana {iso_week:02d} de {iso_year}",
                "markdown": self._weekly_narrative(p, wk, ta, sig, news, w_start, w_end),
                "sentiment": sentiment,
                "ohlcv": wk,
                "themes": [ta["trend"], f"RSI {ta['indicators']['rsi_14']}" if ta["indicators"]["rsi_14"] else "RSI n/d"],
            },
            "daily_entries": daily_entries,
            "macro_snapshots": {},
            "macro_charts": {},
            "signals": sig,
            "technical_analysis": ta,
            "news_context": {
                "article_count": news["total_articles"],
                "avg_sentiment": None,
                "source_breakdown": news["source_breakdown"],
            },
            "news_intelligence": {
                "total_articles": news["total_articles"],
                "clusters": news["clusters"],
            },
            "macro_regime": {
                "regime": {
                    "label": regime_label,
                    "confidence": 0.6 if regime_label == "trend" else 0.4,
                    "transition_probabilities": {},
                },
                "note": h5.get("note"),
            },
            "upcoming_events": [],
            "quality_score": self._quality_score(wk_bars, news, ta),
            "generated_at": None,  # stamped by caller/CLI
        }
        return _sanitize(view)

    @staticmethod
    def _quality_score(wk_bars: pd.DataFrame, news: dict, ta: dict) -> float:
        score = 0.0
        score += min(len(wk_bars) / 5.0, 1.0) * 0.4          # data completeness
        score += min(news["total_articles"] / 20.0, 1.0) * 0.3  # news coverage
        score += 0.3 if ta["indicators"]["rsi_14"] is not None else 0.0  # indicators available
        return round(score, 3)

    # ------------------------------------------------------------------ writing
    def _asset_dir(self, asset_id: str) -> Path:
        d = self.output_root / asset_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_week(self, asset_id: str, view: dict, generated_at: str) -> Path:
        view = dict(view)
        view["generated_at"] = generated_at
        for de in view.get("daily_entries", []):
            de.setdefault("generated_at", generated_at)
        out = self._asset_dir(asset_id) / f"weekly_{view['iso_year']}_W{view['iso_week']:02d}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(_sanitize(view), f, ensure_ascii=False, indent=2)
        return out

    def rebuild_index(self, asset_id: str) -> Path:
        d = self._asset_dir(asset_id)
        weeks = []
        for wf in sorted(d.glob("weekly_*.json")):
            try:
                with open(wf, encoding="utf-8") as f:
                    v = json.load(f)
                ws = v.get("weekly_summary", {})
                weeks.append({
                    "year": v["iso_year"],
                    "week": v["iso_week"],
                    "start": v.get("week_start"),
                    "end": v.get("week_end"),
                    "sentiment": ws.get("sentiment"),
                    "headline": ws.get("headline"),
                    "has_weekly": True,
                    "daily_count": len(v.get("daily_entries", [])),
                })
            except Exception as e:  # noqa: BLE001
                logger.warning("Index skip %s: %s", wf, e)
        # newest first (matches frontend expectation: weeks[0] = most recent)
        weeks.sort(key=lambda w: (w["year"], w["week"]), reverse=True)
        idx = {"asset_id": asset_id, "weeks": weeks}
        out = d / "analysis_index.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(idx, f, ensure_ascii=False, indent=2)
        # ensure an (empty) events file exists so the calendar route 200s
        ev = d / "upcoming_events.json"
        if not ev.exists():
            with open(ev, "w", encoding="utf-8") as f:
                json.dump({"events": [], "generated_at": None}, f)
        return out
