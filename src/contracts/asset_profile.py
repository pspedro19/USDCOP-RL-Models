"""AssetProfile — asset-generic SSOT that parameterizes everything hardcoded to USD/COP.

Contract: CTR-ASSET-PROFILE-001  (see .claude/rules/sdd-multi-asset-onboarding.md §2,
.claude/SPECSGOLD/specs/SPEC-12-scalable-registry-integration.md Contract A).

Onboarding a new tradeable asset (Gold, BTC) MUST NOT string-replace "USDCOP". Instead every
consumer reads an ``AssetProfile`` loaded from ``config/assets/<asset_id>.yaml``. USD/COP itself
is ``config/assets/usdcop.yaml`` so nothing is a special case.

Loaded as a standalone leaf module (yaml only, no ML stack) so it is cheap to import in DAGs,
ingestion scripts, and fast contract tests.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Repo root = three parents up from src/contracts/asset_profile.py
_REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = _REPO_ROOT / "config" / "assets"

_VALID_ASSET_CLASSES = {"fx", "crypto", "commodity", "equity_index"}
_VALID_SESSION_MODES = {"exchange_hours", "metals", "24x7"}


@dataclass(frozen=True)
class SessionSpec:
    """Trading session — replaces the COT 08:00-12:55 / bars_per_day=78 hardcodes."""

    mode: str  # exchange_hours | metals | 24x7
    timezone: str  # canonical storage/session tz for this asset
    days: tuple[int, ...]  # 0=Mon .. 6=Sun; crypto = 0..6
    open: str | None = None  # "HH:MM" in session tz (None for 24x7)
    close: str | None = None
    bars_per_day: int | None = None  # 5-min bars/day; None => measure from data (test A3)
    bars_per_year: int | None = None  # for Sharpe annualization
    trading_days_per_year: int = 250
    weekend_flat: bool = True  # BTC=False (holds over weekend)
    forced_close: str | None = "friday"  # friday | none

    @property
    def is_24x7(self) -> bool:
        return self.mode == "24x7" or set(self.days) >= {5, 6}


@dataclass(frozen=True)
class DataSourceSpec:
    provider: str  # twelvedata | dukascopy | investing | ...
    provider_symbol: str
    interval: str = "5min"
    needs_tz_convert: bool = True  # provider tz -> profile.session.timezone
    seed_file: str | None = None  # relative to repo root
    daily_provider: str | None = None  # optional deep-history daily source
    daily_seed_file: str | None = None


@dataclass(frozen=True)
class MacroDriver:
    """One macro driver, mapped from a provider series to a T-1 lagged feature."""

    series_id: str
    feature: str
    sign: str = "+"  # "+" | "-" expected direction on the asset
    role: str = ""


@dataclass(frozen=True)
class RegimeGateSpec:
    """Hurst-based regime gate. Logic is reusable; thresholds are PER-ASSET (re-fit, not copied)."""

    hurst_lookback: int = 60
    hurst_trending: float | None = None  # None => must be re-fit (test D1); never copy COP 0.52
    hurst_mean_rev: float | None = None
    hysteresis_dwell_days: int = 0


@dataclass(frozen=True)
class AssetProfile:
    asset_id: str
    symbol: str
    chart_symbol: str
    display_name: str
    asset_class: str
    quote_ccy: str
    base_ccy: str
    price_range: tuple[float, float]
    session: SessionSpec
    data_source: DataSourceSpec
    macro_drivers: tuple[MacroDriver, ...] = ()
    cross_asset_leaders: tuple[str, ...] = ()
    regime_gate: RegimeGateSpec = field(default_factory=RegimeGateSpec)
    decimals: int = 2
    tick_size: float = 0.01
    strategy_id: str | None = None
    base_strategy: str | None = None
    pipeline_type: str = "ml_forecasting"
    timeframe: str = "weekly"
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    # ---- validation --------------------------------------------------------
    def validate(self) -> list[str]:
        """Return a list of problems (empty => valid). Backs onboarding test A1."""
        errs: list[str] = []
        if not self.asset_id or "/" in self.asset_id:
            errs.append("asset_id must be a non-empty slug (no '/')")
        if "/" not in self.symbol:
            errs.append(f"symbol should look like 'XAU/USD', got {self.symbol!r}")
        if self.chart_symbol != self.symbol.replace("/", ""):
            errs.append(
                f"chart_symbol {self.chart_symbol!r} must equal symbol without '/' "
                f"({self.symbol.replace('/', '')!r})"
            )
        if self.asset_class not in _VALID_ASSET_CLASSES:
            errs.append(f"asset_class {self.asset_class!r} not in {_VALID_ASSET_CLASSES}")
        lo, hi = self.price_range
        if not (lo < hi):
            errs.append(f"price_range must be (lo<hi), got {self.price_range}")
        if self.session.mode not in _VALID_SESSION_MODES:
            errs.append(f"session.mode {self.session.mode!r} not in {_VALID_SESSION_MODES}")
        if not self.session.days:
            errs.append("session.days must be non-empty")
        return errs

    def require_valid(self) -> "AssetProfile":
        errs = self.validate()
        if errs:
            raise ValueError(f"Invalid AssetProfile '{self.asset_id}':\n  - " + "\n  - ".join(errs))
        return self

    # ---- convenience -------------------------------------------------------
    @property
    def safe_name(self) -> str:
        """Filesystem/seed-safe name, e.g. 'xauusd' for symbol 'XAU/USD'."""
        return self.symbol.replace("/", "").lower()

    def in_price_range(self, price: float) -> bool:
        lo, hi = self.price_range
        return lo <= price <= hi


# --------------------------------------------------------------------------- loader
def _mk_session(d: dict) -> SessionSpec:
    return SessionSpec(
        mode=d.get("mode", "exchange_hours"),
        timezone=d.get("timezone", "UTC"),
        days=tuple(d.get("days", [0, 1, 2, 3, 4])),
        open=d.get("open"),
        close=d.get("close"),
        bars_per_day=d.get("bars_per_day"),
        bars_per_year=d.get("bars_per_year"),
        trading_days_per_year=d.get("trading_days_per_year", 250),
        weekend_flat=d.get("weekend_flat", True),
        forced_close=d.get("forced_close", "friday"),
    )


def _mk_data_source(d: dict) -> DataSourceSpec:
    return DataSourceSpec(
        provider=d.get("provider", "twelvedata"),
        provider_symbol=d.get("provider_symbol", ""),
        interval=d.get("interval", "5min"),
        needs_tz_convert=d.get("needs_tz_convert", True),
        seed_file=d.get("seed_file"),
        daily_provider=d.get("daily_provider"),
        daily_seed_file=d.get("daily_seed_file"),
    )


def profile_from_dict(data: dict[str, Any]) -> AssetProfile:
    symbol = data["symbol"]
    session = _mk_session(data.get("session", {}))
    ds = _mk_data_source(data.get("data_source", {}))
    drivers = tuple(
        MacroDriver(
            series_id=m.get("series_id") or m.get("db_col", ""),
            feature=m.get("feature", ""),
            sign=m.get("sign", "+"),
            role=m.get("role", ""),
        )
        for m in data.get("macro_drivers", [])
    )
    rg_raw = data.get("regime_gate", {})
    regime = RegimeGateSpec(
        hurst_lookback=rg_raw.get("hurst_lookback", 60),
        hurst_trending=rg_raw.get("hurst_trending"),
        hurst_mean_rev=rg_raw.get("hurst_mean_rev"),
        hysteresis_dwell_days=rg_raw.get("hysteresis_dwell_days", 0),
    )
    pr = data.get("price_range", [0.0, 0.0])
    return AssetProfile(
        asset_id=data["asset_id"],
        symbol=symbol,
        chart_symbol=data.get("chart_symbol", symbol.replace("/", "")),
        display_name=data.get("display_name", symbol),
        asset_class=data.get("asset_class", "fx"),
        quote_ccy=data.get("quote_ccy", symbol.split("/")[-1]),
        base_ccy=data.get("base_ccy", symbol.split("/")[0]),
        price_range=(float(pr[0]), float(pr[1])),
        session=session,
        data_source=ds,
        macro_drivers=drivers,
        cross_asset_leaders=tuple(data.get("cross_asset_leaders", [])),
        regime_gate=regime,
        decimals=data.get("decimals", 2),
        tick_size=data.get("tick_size", 0.01),
        strategy_id=data.get("strategy_id"),
        base_strategy=data.get("base_strategy"),
        pipeline_type=data.get("pipeline_type", "ml_forecasting"),
        timeframe=data.get("timeframe", "weekly"),
        raw=data,
    )


def load_asset_profile(asset_id: str, *, assets_dir: Path | None = None) -> AssetProfile:
    """Load and validate ``config/assets/<asset_id>.yaml``."""
    base = assets_dir or ASSETS_DIR
    path = base / f"{asset_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"AssetProfile not found: {path}. Available: "
            f"{[p.stem for p in base.glob('*.yaml')] if base.exists() else '(no assets dir)'}"
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return profile_from_dict(data).require_valid()


def list_assets(*, assets_dir: Path | None = None) -> list[str]:
    base = assets_dir or ASSETS_DIR
    if not base.exists():
        return []
    return sorted(p.stem for p in base.glob("*.yaml"))


def profile_to_dict(p: AssetProfile) -> dict[str, Any]:
    """Serialize back (drops the raw mirror)."""
    d = dataclasses.asdict(p)
    d.pop("raw", None)
    return d
