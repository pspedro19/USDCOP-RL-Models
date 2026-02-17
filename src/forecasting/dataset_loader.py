"""
Forecasting Dataset Loader (SSOT)
=================================

Single shared data loading + feature building for all forecasting consumers.
DB-first with parquet fallback.

Replaces duplicated load_data() / load_full_dataset() / load_and_build_features()
across 4 files with ONE implementation.

Usage:
    from src.forecasting.ssot_config import ForecastingSSOTConfig
    from src.forecasting.dataset_loader import ForecastingDatasetLoader

    cfg = ForecastingSSOTConfig.load()
    loader = ForecastingDatasetLoader(cfg)
    df, feature_cols = loader.load_dataset(target_horizon=5)

Contract: CTR-FORECAST-DATA-LOADER-001
Version: 1.0.0
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.forecasting.ssot_config import ForecastingSSOTConfig

logger = logging.getLogger(__name__)


def _find_project_root() -> Path:
    """Walk up from this file to find project root (contains pyproject.toml)."""
    p = Path(__file__).resolve().parent
    while p != p.parent:
        if (p / "pyproject.toml").is_file():
            return p
        p = p.parent
    airflow_root = Path("/opt/airflow")
    if airflow_root.exists():
        return airflow_root
    return Path(__file__).resolve().parent.parent.parent


class ForecastingDatasetLoader:
    """DB-first data loader with parquet fallback. Builds 21 SSOT features."""

    def __init__(
        self,
        config: ForecastingSSOTConfig,
        db_url: Optional[str] = None,
        project_root: Optional[Path] = None,
    ):
        self.config = config
        self.db_url = db_url or os.environ.get("DATABASE_URL")
        self.project_root = project_root or _find_project_root()

    def load_dataset(
        self,
        target_horizon: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Load OHLCV + macro, build features, optionally compute target.

        Data source priority:
        1. PostgreSQL (db_url set and DB reachable)
        2. Parquet fallback (always works)

        Args:
            target_horizon: If set, compute target column for this horizon.
                           None = no target column (e.g., for inference).

        Returns:
            (df, feature_columns) where df has all feature columns built.
        """
        # Load OHLCV
        df_ohlcv = self._load_ohlcv()

        # Load macro
        df_macro = self._load_macro()

        # Merge macro -> OHLCV with T-1 anti-leakage
        df = self._merge_macro(df_ohlcv, df_macro)

        # Build 21 SSOT features
        df = self._build_features(df)

        # Compute target if requested
        if target_horizon is not None:
            col_name = f"target_return_{target_horizon}d"
            df[col_name] = np.log(df["close"].shift(-target_horizon) / df["close"])

        feature_cols = list(self.config.get_feature_columns())
        return df, feature_cols

    # ------------------------------------------------------------------
    # OHLCV loading
    # ------------------------------------------------------------------

    def _load_ohlcv(self) -> pd.DataFrame:
        """Load daily OHLCV. Try DB first, fall back to parquet."""
        df = self._load_ohlcv_from_db()
        if df is not None and len(df) > 0:
            return df
        return self._load_ohlcv_from_parquet()

    def _load_ohlcv_from_db(self) -> Optional[pd.DataFrame]:
        """Try loading OHLCV from PostgreSQL."""
        if not self.db_url:
            return None

        try:
            import psycopg2

            ohlcv_cfg = self.config.get_data_source("ohlcv")
            table = ohlcv_cfg["db_table"]
            cols = ohlcv_cfg["db_columns"]
            col_str = ", ".join(cols)

            conn = psycopg2.connect(self.db_url)
            try:
                cur = conn.cursor()
                cur.execute(f"SELECT {col_str} FROM {table} ORDER BY date ASC")
                rows = cur.fetchall()
            finally:
                conn.close()

            if not rows:
                logger.info("[DataLoader] DB OHLCV table empty, falling back to parquet")
                return None

            df = pd.DataFrame(rows, columns=cols)
            df["date"] = pd.to_datetime(df["date"])
            for c in ["open", "high", "low", "close"]:
                df[c] = df[c].astype(float)
            df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
            logger.info(f"[DataLoader] OHLCV from DB: {len(df)} rows, "
                        f"{df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()}")
            return df

        except Exception as e:
            logger.warning(f"[DataLoader] DB OHLCV failed ({e}), falling back to parquet")
            return None

    def _load_ohlcv_from_parquet(self) -> pd.DataFrame:
        """Load OHLCV from parquet seed file."""
        ohlcv_cfg = self.config.get_data_source("ohlcv")
        parquet_path = self.project_root / ohlcv_cfg["fallback_parquet"]

        if not parquet_path.exists():
            raise FileNotFoundError(f"[DataLoader] Daily OHLCV not found: {parquet_path}")

        time_col = ohlcv_cfg.get("parquet_time_column", "time")
        df = pd.read_parquet(parquet_path).reset_index()
        if time_col in df.columns and "date" not in df.columns:
            df.rename(columns={time_col: "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        df = df[["date", "open", "high", "low", "close"]].copy()
        df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

        logger.info(f"[DataLoader] OHLCV from parquet: {len(df)} rows, "
                    f"{df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()}")
        return df

    # ------------------------------------------------------------------
    # Macro loading
    # ------------------------------------------------------------------

    def _load_macro(self) -> pd.DataFrame:
        """Load macro data. Try DB first, fall back to parquet."""
        df = self._load_macro_from_db()
        if df is not None and len(df) > 0:
            return df
        return self._load_macro_from_parquet()

    def _load_macro_from_db(self) -> Optional[pd.DataFrame]:
        """Try loading macro from PostgreSQL."""
        if not self.db_url:
            return None

        try:
            import psycopg2

            macro_cfg = self.config.get_data_source("macro")
            table = macro_cfg["db_table"]
            date_col = macro_cfg["db_date_column"]
            col_mapping = self.config.get_macro_column_mapping()

            # Build SQL: select date + raw macro columns
            raw_cols = list(col_mapping.keys())
            col_str = ", ".join([date_col] + raw_cols)

            conn = psycopg2.connect(self.db_url)
            try:
                cur = conn.cursor()
                cur.execute(f"SELECT {col_str} FROM {table} ORDER BY {date_col} ASC")
                rows = cur.fetchall()
            finally:
                conn.close()

            if not rows:
                logger.info("[DataLoader] DB macro table empty, falling back to parquet")
                return None

            df = pd.DataFrame(rows, columns=[date_col] + raw_cols)
            df.rename(columns={date_col: "date"}, inplace=True)
            df["date"] = pd.to_datetime(df["date"])
            df.rename(columns=col_mapping, inplace=True)
            df = df.sort_values("date").reset_index(drop=True)

            logger.info(f"[DataLoader] Macro from DB: {len(df)} rows, "
                        f"{len(col_mapping)} variables")
            return df

        except Exception as e:
            logger.warning(f"[DataLoader] DB macro failed ({e}), falling back to parquet")
            return None

    def _load_macro_from_parquet(self) -> pd.DataFrame:
        """Load macro from parquet seed file."""
        macro_cfg = self.config.get_data_source("macro")
        parquet_path = self.project_root / macro_cfg["fallback_parquet"]

        if not parquet_path.exists():
            raise FileNotFoundError(f"[DataLoader] Macro data not found: {parquet_path}")

        col_mapping = self.config.get_macro_column_mapping()

        df = pd.read_parquet(parquet_path).reset_index()
        # First column is the date (may be named 'fecha', 'date', or index)
        df.rename(columns={df.columns[0]: "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()

        # Select only the raw columns we need + rename
        available_raw = [c for c in col_mapping.keys() if c in df.columns]
        if not available_raw:
            raise ValueError(
                f"[DataLoader] No macro columns found. Expected: {list(col_mapping.keys())}, "
                f"available: {list(df.columns[:20])}"
            )

        df_sub = df[["date"] + available_raw].copy()
        df_sub.rename(columns=col_mapping, inplace=True)
        df_sub = df_sub.sort_values("date").reset_index(drop=True)

        logger.info(f"[DataLoader] Macro from parquet: {len(df_sub)} rows, "
                    f"{len(available_raw)}/{len(col_mapping)} variables")
        return df_sub

    # ------------------------------------------------------------------
    # Macro merge (anti-leakage)
    # ------------------------------------------------------------------

    def _merge_macro(self, df_ohlcv: pd.DataFrame, df_macro: pd.DataFrame) -> pd.DataFrame:
        """
        Merge macro into OHLCV with anti-leakage:
        1. shift(1) on macro columns = T-1 (use yesterday's macro)
        2. merge_asof(direction='backward') = no forward fill from future
        """
        macro_feature_cols = list(self.config.get_macro_column_mapping().values())

        # T-1 shift for anti-leakage
        for col in macro_feature_cols:
            if col in df_macro.columns:
                df_macro[col] = df_macro[col].shift(1)

        df = pd.merge_asof(
            df_ohlcv.sort_values("date"),
            df_macro.sort_values("date"),
            on="date",
            direction="backward",
        )
        return df

    # ------------------------------------------------------------------
    # Feature building (THE single copy)
    # ------------------------------------------------------------------

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build 21 SSOT features from raw OHLCV + macro.

        Features:
            4 price:     close, open, high, low (already in df)
            4 returns:   return_1d, return_5d, return_10d, return_20d
            3 volatility: volatility_5d, volatility_10d, volatility_20d
            3 technical: rsi_14d (Wilder's EMA), ma_ratio_20d, ma_ratio_50d
            3 calendar:  day_of_week, month, is_month_end
            4 macro:     dxy_close_lag1, oil_close_lag1, vix_close_lag1, embi_close_lag1
        """
        df = df.copy()

        # Returns (4)
        df["return_1d"] = df["close"].pct_change(1)
        df["return_5d"] = df["close"].pct_change(5)
        df["return_10d"] = df["close"].pct_change(10)
        df["return_20d"] = df["close"].pct_change(20)

        # Volatility (3)
        df["volatility_5d"] = df["return_1d"].rolling(5).std()
        df["volatility_10d"] = df["return_1d"].rolling(10).std()
        df["volatility_20d"] = df["return_1d"].rolling(20).std()

        # RSI 14d — Wilder's EMA (alpha=1/period, NOT pandas default)
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1 / 14, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1 / 14, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi_14d"] = 100 - (100 / (1 + rs))

        # MA ratios (2)
        df["ma_ratio_20d"] = df["close"] / df["close"].rolling(20).mean()
        df["ma_ratio_50d"] = df["close"] / df["close"].rolling(50).mean()

        # Calendar (3)
        df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
        df["month"] = pd.to_datetime(df["date"]).dt.month
        df["is_month_end"] = pd.to_datetime(df["date"]).dt.is_month_end.astype(int)

        # Macro (4) — forward-fill gaps after merge
        macro_cols = list(self.config.get_macro_column_mapping().values())
        for col in macro_cols:
            if col in df.columns:
                df[col] = df[col].ffill()

        return df

    # ------------------------------------------------------------------
    # Convenience: load with DB extension (for DAGs)
    # ------------------------------------------------------------------

    def load_dataset_with_db_extension(
        self,
        target_horizon: Optional[int] = None,
        db_conn_func=None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Load from parquet, then extend with latest DB rows (if available).
        Used by Airflow DAGs that have DB access and want the freshest data.

        Args:
            target_horizon: Horizon for target column.
            db_conn_func: Callable returning a psycopg2 connection.
                         If None, uses self.db_url.
        """
        # Load base from parquet (always available)
        df_ohlcv = self._load_ohlcv_from_parquet()

        # Try extending with DB rows newer than parquet
        if db_conn_func is not None or self.db_url:
            try:
                last_parquet_date = df_ohlcv["date"].iloc[-1]

                if db_conn_func:
                    conn = db_conn_func()
                else:
                    import psycopg2
                    conn = psycopg2.connect(self.db_url)

                try:
                    ohlcv_cfg = self.config.get_data_source("ohlcv")
                    table = ohlcv_cfg["db_table"]
                    cur = conn.cursor()
                    cur.execute(f"""
                        SELECT date, open, high, low, close
                        FROM {table}
                        WHERE date > %s
                        ORDER BY date ASC
                    """, (last_parquet_date.date(),))
                    db_rows = cur.fetchall()
                finally:
                    conn.close()

                if db_rows:
                    df_db = pd.DataFrame(
                        db_rows, columns=["date", "open", "high", "low", "close"]
                    )
                    df_db["date"] = pd.to_datetime(df_db["date"])
                    for c in ["open", "high", "low", "close"]:
                        df_db[c] = df_db[c].astype(float)
                    df_ohlcv = pd.concat([df_ohlcv, df_db], ignore_index=True)
                    df_ohlcv = (
                        df_ohlcv.drop_duplicates(subset=["date"])
                        .sort_values("date")
                        .reset_index(drop=True)
                    )
                    logger.info(f"[DataLoader] Extended with {len(db_rows)} DB rows -> "
                                f"{len(df_ohlcv)} total")

            except Exception as e:
                logger.warning(f"[DataLoader] DB extension failed ({e}), using parquet only")

        # Load macro (parquet)
        df_macro = self._load_macro_from_parquet()

        # Merge + features
        df = self._merge_macro(df_ohlcv, df_macro)
        df = self._build_features(df)

        if target_horizon is not None:
            col_name = f"target_return_{target_horizon}d"
            df[col_name] = np.log(df["close"].shift(-target_horizon) / df["close"])

        feature_cols = list(self.config.get_feature_columns())
        return df, feature_cols
