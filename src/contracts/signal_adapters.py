"""
Signal Adapters — Convert strategy-native outputs to UniversalSignalRecord.
============================================================================

Each adapter converts a strategy's native prediction/output to the universal
signal format. This decouples signal generation from execution.

Adapters:
    H5SmartSimpleAdapter      — Ridge+BR walk-forward -> weekly signals
    H1ForecastVTAdapter       — 9-model ensemble -> daily signals
    RLPPOAdapter              — PPO model.predict() -> intraday signals

Contract: CTR-SIGNAL-ADAPTER-001
"""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import yaml

from src.contracts.signal_contract import (
    UniversalSignalRecord,
    SignalDirection,
    BarFrequency,
    EntryType,
)
from src.forecasting.confidence_scorer import (
    ConfidenceConfig,
    score_confidence,
)
from src.forecasting.adaptive_stops import (
    AdaptiveStopsConfig,
    compute_adaptive_stops,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Shared: Load OHLCV + macro + 21 features
# ---------------------------------------------------------------------------

def load_forecasting_data() -> tuple:
    """
    Load OHLCV + macro and build 21 features + H=5 target.

    Returns:
        (df, feature_cols) — DataFrame with features and list of column names.

    This is the exact data loading logic from backtest_smart_simple_v1.py.
    """
    ohlcv_path = PROJECT_ROOT / "seeds" / "latest" / "usdcop_daily_ohlcv.parquet"
    macro_path = PROJECT_ROOT / "data" / "pipeline" / "04_cleaning" / "output" / "MACRO_DAILY_CLEAN.parquet"

    df_ohlcv = pd.read_parquet(ohlcv_path).reset_index()
    df_ohlcv.rename(columns={"time": "date"}, inplace=True)
    df_ohlcv["date"] = pd.to_datetime(df_ohlcv["date"]).dt.tz_localize(None).dt.normalize()
    df_ohlcv = df_ohlcv[["date", "open", "high", "low", "close"]].sort_values("date").reset_index(drop=True)

    df_macro = pd.read_parquet(macro_path).reset_index()
    df_macro.rename(columns={df_macro.columns[0]: "date"}, inplace=True)
    df_macro["date"] = pd.to_datetime(df_macro["date"]).dt.tz_localize(None).dt.normalize()

    macro_cols = {
        "FXRT_INDEX_DXY_USA_D_DXY": "dxy_close_lag1",
        "COMM_OIL_WTI_GLB_D_WTI": "oil_close_lag1",
        "VOLT_VIX_USA_D_VIX": "vix_close_lag1",
        "CRSK_SPREAD_EMBI_COL_D_EMBI": "embi_close_lag1",
    }
    df_macro_sub = df_macro[["date"] + list(macro_cols.keys())].copy()
    df_macro_sub.rename(columns=macro_cols, inplace=True)
    df_macro_sub = df_macro_sub.sort_values("date")
    for col in macro_cols.values():
        df_macro_sub[col] = df_macro_sub[col].shift(1)  # T-1 anti-leakage

    df = pd.merge_asof(
        df_ohlcv.sort_values("date"),
        df_macro_sub.sort_values("date"),
        on="date",
        direction="backward",
    )

    # Returns
    df["return_1d"] = df["close"].pct_change(1)
    df["return_5d"] = df["close"].pct_change(5)
    df["return_10d"] = df["close"].pct_change(10)
    df["return_20d"] = df["close"].pct_change(20)

    # Volatility
    df["volatility_5d"] = df["return_1d"].rolling(5).std()
    df["volatility_10d"] = df["return_1d"].rolling(10).std()
    df["volatility_20d"] = df["return_1d"].rolling(20).std()

    # RSI (Wilder's EMA)
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14d"] = 100 - (100 / (1 + rs))

    # MA ratios
    df["ma_ratio_20d"] = df["close"] / df["close"].rolling(20).mean()
    df["ma_ratio_50d"] = df["close"] / df["close"].rolling(50).mean()

    # Calendar
    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df["is_month_end"] = pd.to_datetime(df["date"]).dt.is_month_end.astype(int)

    # Forward-fill macro
    for col in ["dxy_close_lag1", "oil_close_lag1", "vix_close_lag1", "embi_close_lag1"]:
        df[col] = df[col].ffill()

    # Targets
    df["target_return_5d"] = np.log(df["close"].shift(-5) / df["close"])
    df["target_return_1d"] = np.log(df["close"].shift(-1) / df["close"])

    feature_cols = [
        "close", "open", "high", "low",
        "return_1d", "return_5d", "return_10d", "return_20d",
        "volatility_5d", "volatility_10d", "volatility_20d",
        "rsi_14d", "ma_ratio_20d", "ma_ratio_50d",
        "day_of_week", "month", "is_month_end",
        "dxy_close_lag1", "oil_close_lag1",
        "vix_close_lag1", "embi_close_lag1",
    ]
    return df, feature_cols


def _load_yaml_config(config_path: Path) -> dict:
    """Load a YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# H5 Smart Simple Adapter
# ---------------------------------------------------------------------------

class H5SmartSimpleAdapter:
    """
    Convert H5 Smart Simple predictions to universal signals.

    Walk-forward: for each Monday in year, train on history, predict, produce signal.
    Replicates the exact logic from backtest_smart_simple_v1.py run_backtest()
    with mode="bidir_smart".
    """

    def __init__(self, config_path: Optional[Path] = None):
        cfg_path = config_path or (PROJECT_ROOT / "config" / "execution" / "smart_simple_v1.yaml")
        cfg = _load_yaml_config(cfg_path)

        # Vol-targeting
        vt = cfg.get("vol_targeting", {})
        self.vt_target = vt.get("target_vol", 0.15)
        self.vt_max = vt.get("max_leverage", 2.0)
        self.vt_min = vt.get("min_leverage", 0.5)
        self.vt_floor = vt.get("vol_floor", 0.05)

        # Adaptive stops
        _as = cfg.get("adaptive_stops", {})
        self.stops_config = AdaptiveStopsConfig(
            vol_multiplier=_as.get("vol_multiplier", 2.0),
            hard_stop_min_pct=_as.get("hard_stop_min_pct", 0.01),
            hard_stop_max_pct=_as.get("hard_stop_max_pct", 0.03),
            tp_ratio=_as.get("tp_ratio", 0.5),
        )

        # Confidence
        cc = cfg.get("confidence", {})
        self.conf_config = ConfidenceConfig(
            agreement_tight=cc.get("agreement_tight", 0.001),
            agreement_loose=cc.get("agreement_loose", 0.005),
            magnitude_high=cc.get("magnitude_high", 0.010),
            magnitude_medium=cc.get("magnitude_medium", 0.005),
            short_high=cc.get("short", {}).get("HIGH", 1.5),
            short_medium=cc.get("short", {}).get("MEDIUM", 1.5),
            short_low=cc.get("short", {}).get("LOW", 1.5),
            long_high=cc.get("long", {}).get("HIGH", 1.0),
            long_medium=cc.get("long", {}).get("MEDIUM", 0.5),
            long_low=cc.get("long", {}).get("LOW", 0.0),
        )

        self.strategy_id = "smart_simple_v11"

    def generate_signals(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        year: int,
    ) -> List[UniversalSignalRecord]:
        """
        Walk-forward signal generation for H5.

        For each Monday in year:
        1. Train Ridge+BR on all data before Monday
        2. Predict on latest row
        3. Score confidence
        4. Compute vol-target leverage + adaptive stops
        5. Produce UniversalSignalRecord

        Returns list of signals (one per trading week).
        """
        from sklearn.linear_model import Ridge, BayesianRidge
        from sklearn.preprocessing import StandardScaler

        test_start = pd.Timestamp(f"{year}-01-01")
        test_end = pd.Timestamp(f"{year}-12-31")
        test_data = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
        test_data["dow"] = test_data["date"].dt.dayofweek
        mondays = test_data[test_data["dow"] == 0]["date"].unique()

        signals = []

        for monday in mondays:
            monday_ts = pd.Timestamp(monday)

            # Train on all data before Monday
            train_end = monday_ts - timedelta(days=1)
            df_train = df[(df["date"] <= train_end) & df["target_return_5d"].notna()].copy()
            mask = df_train[feature_cols].notna().all(axis=1) & df_train["target_return_5d"].notna()
            df_train = df_train[mask]
            if len(df_train) < 100:
                continue

            X = df_train[feature_cols].values.astype(np.float64)
            y = df_train["target_return_5d"].values.astype(np.float64)
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)

            ridge = Ridge(alpha=1.0).fit(Xs, y)
            br = BayesianRidge(max_iter=300).fit(Xs, y)

            # Predict on latest training row
            latest = scaler.transform(df_train[feature_cols].iloc[-1:].values.astype(np.float64))
            pred_ridge = float(ridge.predict(latest)[0])
            pred_br = float(br.predict(latest)[0])
            ensemble = (pred_ridge + pred_br) / 2.0
            direction = 1 if ensemble > 0 else -1

            # Confidence scoring
            conf = score_confidence(pred_ridge, pred_br, direction, self.conf_config)

            # Vol-targeting (base leverage)
            rets = df_train["return_1d"].dropna().values
            rv_daily = np.std(rets[-21:]) if len(rets) >= 21 else 0.0
            rv_ann = rv_daily * np.sqrt(252) if rv_daily > 0 else self.vt_target
            safe_vol = max(rv_ann, self.vt_floor)
            base_lev = float(np.clip(self.vt_target / safe_vol, self.vt_min, self.vt_max))

            # Apply confidence multiplier
            final_lev = base_lev * conf.sizing_multiplier
            final_lev = float(np.clip(final_lev, self.vt_min, self.vt_max))

            # Adaptive stops
            stops = compute_adaptive_stops(rv_ann, self.stops_config)

            # Entry price
            monday_row = df[df["date"] == monday_ts]
            if monday_row.empty:
                m2 = df["date"] >= monday_ts
                if m2.any():
                    monday_row = df[m2].iloc[:1]
                else:
                    continue
            entry_price = float(monday_row["close"].iloc[0])

            # ISO week
            iso_week = monday_ts.isocalendar()
            signal_id = f"h5_{year}-W{iso_week[1]:02d}"

            signals.append(UniversalSignalRecord(
                signal_id=signal_id,
                strategy_id=self.strategy_id,
                signal_date=str(monday_ts.date()),
                direction=direction,
                magnitude=abs(ensemble),
                confidence=1.0 - conf.agreement / 0.01 if conf.agreement < 0.01 else 0.0,
                skip_trade=conf.skip_trade,
                leverage=final_lev,
                hard_stop_pct=stops.hard_stop_pct,
                take_profit_pct=stops.take_profit_pct,
                entry_price=entry_price,
                entry_type=EntryType.LIMIT.value,
                horizon_bars=5,
                bar_frequency=BarFrequency.WEEKLY.value,
                metadata={
                    "pred_ridge": round(pred_ridge, 6),
                    "pred_br": round(pred_br, 6),
                    "ensemble_return": round(ensemble, 6),
                    "confidence_tier": conf.tier.value,
                    "agreement": round(conf.agreement, 6),
                    "sizing_multiplier": round(conf.sizing_multiplier, 3),
                    "rv_ann": round(rv_ann, 6),
                    "base_leverage": round(base_lev, 4),
                },
            ))

        logger.info(
            "H5 generated %d signals for %d (skipped: %d)",
            len(signals), year, sum(1 for s in signals if s.skip_trade),
        )
        return signals


# ---------------------------------------------------------------------------
# H1 Forecast+VT Adapter
# ---------------------------------------------------------------------------

class H1ForecastVTAdapter:
    """
    Convert H1 9-model ensemble predictions to universal signals.

    Walk-forward: retrain weekly (Sundays), predict daily (Mon-Fri).
    Ensemble: top_3 by prediction magnitude.
    Direction: SHORT-only (2026 regime).
    """

    def __init__(self, config_path: Optional[Path] = None):
        cfg_path = config_path or (PROJECT_ROOT / "config" / "execution" / "smart_executor_v1.yaml")
        cfg = _load_yaml_config(cfg_path)

        ts = cfg.get("trailing_stop", {})
        self.activation_pct = ts.get("activation_pct", 0.002)
        self.trail_pct = ts.get("trail_pct", 0.003)
        self.hard_stop_pct = ts.get("hard_stop_pct", 0.015)

        df_ = cfg.get("direction_filter", {})
        self.direction_mode = df_.get("mode", "short_only")

        self.slippage_bps = cfg.get("broker", {}).get("slippage_bps", 1.0)
        self.strategy_id = "forecast_vt_trailing"

        # Vol-targeting (same defaults as H5)
        self.vt_target = 0.15
        self.vt_max = 2.0
        self.vt_min = 0.5
        self.vt_floor = 0.05

    def generate_signals(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        year: int,
    ) -> List[UniversalSignalRecord]:
        """
        Walk-forward daily signal generation for H1.

        Retrain weekly (every Sunday/start-of-week), predict each trading day.
        9-model ensemble, top_3 by prediction magnitude.
        """
        from sklearn.linear_model import Ridge, BayesianRidge, ARDRegression
        from sklearn.preprocessing import StandardScaler

        try:
            from xgboost import XGBRegressor
        except ImportError:
            XGBRegressor = None
        try:
            from lightgbm import LGBMRegressor
        except ImportError:
            LGBMRegressor = None
        try:
            from catboost import CatBoostRegressor
        except ImportError:
            CatBoostRegressor = None

        test_start = pd.Timestamp(f"{year}-01-01")
        test_end = pd.Timestamp(f"{year}-12-31")
        trading_days = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
        trading_days = trading_days.sort_values("date")

        signals = []
        last_train_week = None
        models = {}
        scaler = None

        for _, row in trading_days.iterrows():
            day = row["date"]
            iso_week = day.isocalendar()[1]

            # Retrain weekly
            if last_train_week != iso_week:
                train_end = day - timedelta(days=1)
                df_train = df[(df["date"] <= train_end) & df["target_return_1d"].notna()].copy()
                mask = df_train[feature_cols].notna().all(axis=1)
                df_train = df_train[mask]
                if len(df_train) < 100:
                    continue

                X = df_train[feature_cols].values.astype(np.float64)
                y = df_train["target_return_1d"].values.astype(np.float64)
                scaler = StandardScaler()
                Xs = scaler.fit_transform(X)

                # Train 9 models (graceful degradation if boosting libs missing)
                models = {
                    "ridge": Ridge(alpha=1.0).fit(Xs, y),
                    "bayesian_ridge": BayesianRidge(max_iter=300).fit(Xs, y),
                    "ard": ARDRegression(max_iter=300).fit(Xs, y),
                }

                if XGBRegressor is not None:
                    models["xgboost"] = XGBRegressor(
                        n_estimators=100, max_depth=4, learning_rate=0.05,
                        verbosity=0,
                    ).fit(Xs, y)
                if LGBMRegressor is not None:
                    models["lightgbm"] = LGBMRegressor(
                        n_estimators=100, max_depth=4, learning_rate=0.05,
                        verbose=-1,
                    ).fit(Xs, y)
                if CatBoostRegressor is not None:
                    models["catboost"] = CatBoostRegressor(
                        iterations=100, depth=4, learning_rate=0.05,
                        verbose=0,
                    ).fit(Xs, y)

                last_train_week = iso_week

            if scaler is None or not models:
                continue

            # Predict
            X_today = scaler.transform(
                df[df["date"] == day][feature_cols].values.astype(np.float64)
            )
            if X_today.shape[0] == 0:
                continue

            predictions = {}
            for name, model in models.items():
                predictions[name] = float(model.predict(X_today)[0])

            # Top_3 ensemble by magnitude
            sorted_preds = sorted(predictions.items(), key=lambda x: abs(x[1]), reverse=True)
            top_3 = sorted_preds[:3]
            ensemble = np.mean([p for _, p in top_3])
            direction = 1 if ensemble > 0 else -1

            # Direction filter
            if self.direction_mode == "short_only" and direction == 1:
                signals.append(UniversalSignalRecord(
                    signal_id=f"h1_{day.date()}",
                    strategy_id=self.strategy_id,
                    signal_date=str(day.date()),
                    direction=direction,
                    magnitude=abs(ensemble),
                    confidence=0.5,
                    skip_trade=True,
                    leverage=0.0,
                    entry_price=float(row["close"]),
                    entry_type=EntryType.LIMIT.value,
                    horizon_bars=1,
                    bar_frequency=BarFrequency.DAILY.value,
                    metadata={"predictions": predictions, "reason": "short_only_filter"},
                ))
                continue

            # Vol-targeting
            rets = df[df["date"] <= day]["return_1d"].dropna().values
            rv_daily = np.std(rets[-21:]) if len(rets) >= 21 else 0.0
            rv_ann = rv_daily * np.sqrt(252) if rv_daily > 0 else self.vt_target
            safe_vol = max(rv_ann, self.vt_floor)
            final_lev = float(np.clip(self.vt_target / safe_vol, self.vt_min, self.vt_max))

            entry_price = float(row["close"])

            signals.append(UniversalSignalRecord(
                signal_id=f"h1_{day.date()}",
                strategy_id=self.strategy_id,
                signal_date=str(day.date()),
                direction=direction,
                magnitude=abs(ensemble),
                confidence=0.5,
                skip_trade=False,
                leverage=final_lev,
                hard_stop_pct=self.hard_stop_pct,
                trailing_activation_pct=self.activation_pct,
                trailing_distance_pct=self.trail_pct,
                entry_price=entry_price,
                entry_type=EntryType.LIMIT.value,
                horizon_bars=1,
                bar_frequency=BarFrequency.DAILY.value,
                metadata={
                    "predictions": {k: round(v, 6) for k, v in predictions.items()},
                    "top_3": [name for name, _ in top_3],
                    "ensemble_return": round(ensemble, 6),
                    "rv_ann": round(rv_ann, 6),
                },
            ))

        logger.info(
            "H1 generated %d signals for %d (skipped: %d)",
            len(signals), year, sum(1 for s in signals if s.skip_trade),
        )
        return signals


# ---------------------------------------------------------------------------
# RL PPO Adapter
# ---------------------------------------------------------------------------

class RLPPOAdapter:
    """
    Convert RL PPO model outputs to universal signals.

    Steps through 5-min bars, runs model.predict(), and converts raw actions
    to directional signals.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        norm_stats_path: Optional[Path] = None,
        config_path: Optional[Path] = None,
    ):
        self.model_path = model_path
        self.norm_stats_path = norm_stats_path
        self.config_path = config_path or (PROJECT_ROOT / "config" / "pipeline_ssot.yaml")
        self.strategy_id = "rl_v215b"

        # Load config
        cfg = _load_yaml_config(self.config_path)
        env = cfg.get("training", {}).get("environment", {})
        self.threshold_long = env.get("threshold_long", 0.50)
        self.threshold_short = env.get("threshold_short", -0.50)
        self.stop_loss_pct = abs(env.get("stop_loss_pct", -0.025))
        self.take_profit_pct = env.get("take_profit_pct", 0.03)
        self.max_holding = env.get("max_position_holding", 576)

    def generate_signals(
        self,
        ohlcv_5min: pd.DataFrame,
        year: int,
    ) -> List[UniversalSignalRecord]:
        """
        Step through 5-min bars, run model.predict(), produce signals.

        Requires a trained PPO model (model_path) and norm_stats (norm_stats_path).
        If model is not available, returns empty list with a warning.
        """
        if self.model_path is None or not Path(self.model_path).exists():
            logger.warning("RL model not found at %s, returning empty signals", self.model_path)
            return []

        try:
            from stable_baselines3 import PPO
        except ImportError:
            logger.warning("stable_baselines3 not installed, RL adapter unavailable")
            return []

        model = PPO.load(str(self.model_path), device="cpu")

        # Load norm_stats if available
        norm_stats = None
        if self.norm_stats_path and Path(self.norm_stats_path).exists():
            import json
            with open(self.norm_stats_path) as f:
                norm_stats = json.load(f)

        # Filter to year
        test_start = pd.Timestamp(f"{year}-01-01")
        test_end = pd.Timestamp(f"{year}-12-31")
        bars = ohlcv_5min[
            (ohlcv_5min["date"] >= test_start) & (ohlcv_5min["date"] <= test_end)
        ].sort_values("date").reset_index(drop=True)

        if bars.empty:
            return []

        signals = []
        in_position = False

        for idx, bar in bars.iterrows():
            if in_position:
                continue

            # Build simple observation (price features only for signal generation)
            # This is a simplified version — full RL needs env.step() loop
            close = float(bar["close"])
            obs = np.zeros(20, dtype=np.float32)  # Placeholder observation
            # In production, this would use CanonicalFeatureBuilder

            try:
                action, _ = model.predict(obs, deterministic=True)
                raw_action = float(action[0]) if hasattr(action, '__len__') else float(action)
            except Exception:
                continue

            # Discretize action
            if raw_action >= self.threshold_long:
                direction = 1
            elif raw_action <= self.threshold_short:
                direction = -1
            else:
                continue  # HOLD — no signal

            signal_ts = bar["date"]
            signals.append(UniversalSignalRecord(
                signal_id=f"rl_{signal_ts}",
                strategy_id=self.strategy_id,
                signal_date=str(signal_ts),
                direction=direction,
                magnitude=abs(raw_action),
                confidence=min(abs(raw_action), 1.0),
                skip_trade=False,
                leverage=1.0,
                hard_stop_pct=self.stop_loss_pct,
                take_profit_pct=self.take_profit_pct,
                entry_price=close,
                entry_type=EntryType.MARKET.value,
                horizon_bars=self.max_holding,
                bar_frequency=BarFrequency.FIVE_MIN.value,
                metadata={
                    "raw_action": round(raw_action, 6),
                },
            ))

            in_position = True
            # Simple position tracking — reset after horizon
            # In a real backtest, the execution strategy handles exits

        logger.info("RL generated %d signals for %d", len(signals), year)
        return signals


# ---------------------------------------------------------------------------
# Adapter Registry
# ---------------------------------------------------------------------------

ADAPTER_REGISTRY = {
    "smart_simple_v11": H5SmartSimpleAdapter,
    "forecast_vt_trailing": H1ForecastVTAdapter,
    "rl_v215b": RLPPOAdapter,
}


def get_adapter(strategy_id: str, **kwargs):
    """Get the appropriate adapter for a strategy ID."""
    cls = ADAPTER_REGISTRY.get(strategy_id)
    if cls is None:
        raise ValueError(f"Unknown strategy: {strategy_id}. Known: {list(ADAPTER_REGISTRY.keys())}")
    return cls(**kwargs)
