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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.forecasting.ssot_config import ForecastingSSOTConfig
from src.identity.canonical import semantic_hash

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SourceProvenance:
    kind: str
    storage_uri: str
    semantic_hash: str
    row_count: int


@dataclass(frozen=True, slots=True)
class DatasetProvenance:
    """Content identity of the exact frame offered to forecasting consumers."""

    snapshot_semantic_hash: str
    snapshot_columns: tuple[str, ...]
    row_count: int
    min_event_time: str
    max_event_time: str
    ohlcv: SourceProvenance
    macro: SourceProvenance


def _canonical_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    # Dataset dates are deliberately represented as ISO wall dates.  They are
    # daily economic keys, not timezone-naive instants.
    if hasattr(value, "isoformat") and type(value).__module__ == "datetime":
        return value.isoformat()
    return value


def _frame_payload(frame: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    rows = [
        [_canonical_scalar(value) for value in row]
        for row in frame.loc[:, columns].itertuples(index=False, name=None)
    ]
    return {"columns": columns, "rows": rows}


def _frame_semantic_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    return semantic_hash(_frame_payload(frame, columns))


def rebind_dataset_provenance(
    provenance: DatasetProvenance,
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> DatasetProvenance:
    """Bind existing source provenance to a downstream frame actually consumed."""
    columns = [
        column
        for column in ["date", "open", "high", "low", "close", *feature_columns]
        if column in frame.columns
    ]
    return DatasetProvenance(
        snapshot_semantic_hash=_frame_semantic_hash(frame, columns),
        snapshot_columns=tuple(columns),
        row_count=len(frame),
        min_event_time=pd.Timestamp(frame["date"].min()).isoformat(),
        max_event_time=pd.Timestamp(frame["date"].max()).isoformat(),
        ohlcv=provenance.ohlcv,
        macro=provenance.macro,
    )


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
        db_url: str | None = None,
        project_root: Path | None = None,
    ):
        self.config = config
        self.db_url = db_url or os.environ.get("DATABASE_URL")
        self.project_root = project_root or _find_project_root()
        self._ohlcv_provenance: SourceProvenance | None = None
        self._macro_provenance: SourceProvenance | None = None
        self._last_provenance: DatasetProvenance | None = None

    @property
    def provenance(self) -> DatasetProvenance:
        if self._last_provenance is None:
            raise RuntimeError("dataset provenance is unavailable before load_dataset()")
        return self._last_provenance

    def load_dataset(
        self,
        target_horizon: int | None = None,
    ) -> tuple[pd.DataFrame, list[str]]:
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
        if self._ohlcv_provenance is None or self._macro_provenance is None:
            raise RuntimeError("dataset sources completed without provenance")
        snapshot_columns = [
            column
            for column in ["date", "open", "high", "low", "close", *feature_cols]
            if column in df.columns
        ]
        self._last_provenance = DatasetProvenance(
            snapshot_semantic_hash=_frame_semantic_hash(df, snapshot_columns),
            snapshot_columns=tuple(snapshot_columns),
            row_count=len(df),
            min_event_time=pd.Timestamp(df["date"].min()).isoformat(),
            max_event_time=pd.Timestamp(df["date"].max()).isoformat(),
            ohlcv=self._ohlcv_provenance,
            macro=self._macro_provenance,
        )
        return df, feature_cols

    # ------------------------------------------------------------------
    # OHLCV loading
    # ------------------------------------------------------------------

    def _load_ohlcv(self) -> pd.DataFrame:
        """Load daily OHLCV. Try DB first, fall back to parquet."""
        df = self._load_ohlcv_from_db()
        if df is not None and len(df) > 0:
            table = self.config.get_data_source("ohlcv")["db_table"]
            self._ohlcv_provenance = SourceProvenance(
                kind="postgresql",
                storage_uri=f"db://{table}",
                semantic_hash=_frame_semantic_hash(df, list(df.columns)),
                row_count=len(df),
            )
            return df
        df = self._load_ohlcv_from_parquet()
        fallback_path = Path(self.config.get_data_source("ohlcv")["fallback_parquet"])
        self._ohlcv_provenance = SourceProvenance(
            kind="parquet",
            storage_uri=f"repo://{fallback_path.as_posix()}",
            semantic_hash=_frame_semantic_hash(df, list(df.columns)),
            row_count=len(df),
        )
        return df

    def _load_ohlcv_from_db(self) -> pd.DataFrame | None:
        """Try loading OHLCV from PostgreSQL."""
        if not self.db_url:
            return None

        try:
            import psycopg2

            ohlcv_cfg = self.config.get_data_source("ohlcv")
            # Per-asset opt-out: some assets (e.g. BTC in the mixed-symbol
            # asset_daily_ohlcv table) must always use the parquet seed.
            if ohlcv_cfg.get("disable_db"):
                return None
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
            table = self.config.get_data_source("macro")["db_table"]
            self._macro_provenance = SourceProvenance(
                kind="postgresql",
                storage_uri=f"db://{table}",
                semantic_hash=_frame_semantic_hash(df, list(df.columns)),
                row_count=len(df),
            )
            return df
        df = self._load_macro_from_parquet()
        fallback_path = Path(self.config.get_data_source("macro")["fallback_parquet"])
        self._macro_provenance = SourceProvenance(
            kind="parquet",
            storage_uri=f"repo://{fallback_path.as_posix()}",
            semantic_hash=_frame_semantic_hash(df, list(df.columns)),
            row_count=len(df),
        )
        return df

    def _load_macro_from_db(self) -> pd.DataFrame | None:
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
        target_horizon: int | None = None,
        db_conn_func=None,
    ) -> tuple[pd.DataFrame, list[str]]:
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

        # Load macro (DB first, parquet fallback)
        df_macro = self._load_macro()

        # Merge + features
        df = self._merge_macro(df_ohlcv, df_macro)
        df = self._build_features(df)

        if target_horizon is not None:
            col_name = f"target_return_{target_horizon}d"
            df[col_name] = np.log(df["close"].shift(-target_horizon) / df["close"])

        feature_cols = list(self.config.get_feature_columns())
        return df, feature_cols
