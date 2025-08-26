"""
src/markets/usdcop/config.py

Configuración específica del mercado USDCOP:
- Parámetros del símbolo (digits, point, pip, lot size, timezone)
- Costos de trading (spread, slippage, comisiones, holding)
- Timeframes soportados y por defecto
- Perfiles de spread por hora (heurístico) y utilidades de coste dinámico
- Paths de datos y umbrales de calidad (en línea con mt5_config.yaml)
- Defaults del entorno RL y feature engineering
- Carga/merge con 'configs/mt5_config.yaml' (perfiles + ${ENV})

Sin dependencias pesadas; opcionalmente usa PyYAML si el YAML existe.

Uso rápido:
    from src.markets.usdcop.config import load_usdcop_config
    mkt = load_usdcop_config()  # devuelve USDCOPConfig
    pip_size = mkt.symbol.pip_size()
    pip_val_cop = mkt.symbol.pip_value_per_lot_cop(price=4090.25)

CLI:
    python -m src.markets.usdcop.config show
"""

from __future__ import annotations

import os
import math
import json
import logging
from dataclasses import dataclass, asdict, field, replace
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s:%(lineno)d | %(message)s")
    _h.setFormatter(_fmt)
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# Timeframes
# -----------------------------------------------------------------------------
TF_MIN = {"M1": 1, "M2": 2, "M3": 3, "M5": 5, "M10": 10, "M15": 15, "M30": 30, "H1": 60, "H2": 120, "H4": 240, "D1": 1440}
DEFAULT_TIMEFRAME = "M5"

# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------
@dataclass
class SymbolConfig:
    name: str = "USDCOP"
    digits: int = 2
    point: float = 0.01                 # 1 "point" en precio
    pip_in_points: int = 1              # 1 pip = pip_in_points * point (para 2 dígitos: 1)
    lot_size_units: int = 100_000       # estándar FX: 1 lote = 100k de la divisa base (USD)
    timezone: str = "America/Bogota"
    trading_days: Tuple[int, ...] = (0, 1, 2, 3, 4)  # L-V (Mon=0)

    def pip_size(self) -> float:
        return float(self.point) * float(self.pip_in_points)

    def pip_value_per_lot_cop(self, price: float) -> float:
        """
        Valor de 1 pip por lote en COP (divisa cotizada). Para cotización directa (USDCOP),
        pip_value_quote ≈ lot_size_units * pip_size.
        """
        return float(self.lot_size_units) * self.pip_size()

    def pip_value_per_lot_usd(self, price: float) -> float:
        """
        Valor de 1 pip por lote en USD. Para cotización directa (USDCOP),
        pip_value_USD ≈ (lot_size * pip_size) / price.
        """
        return (self.lot_size_units * self.pip_size()) / max(price, 1e-12)


@dataclass
class CostConfig:
    # Costos "promedio" base (pueden ajustarse dinámicamente via spread_profile)
    default_spread_points: float = 10.0
    slippage_points: float = 2.0
    commission_bp_per_side: float = 0.0
    holding_cost_bp_per_bar: float = 0.0

    def turnover_cost_fraction(self, *, price: float, point: float, spread_points: float | None = None) -> float:
        """
        Coste fraccional aplicado al cambiar de posición (long→flat, flat→short, etc.).
          spread_cost  ≈ spread_points * point / price
          slip_cost    ≈ 2 * slippage_points * point / price
          comm_cost    ≈ 2 * commission_bp_per_side * 1e-4
        """
        sp = float(spread_points if spread_points is not None else self.default_spread_points)
        spread_cost = (sp * point) / max(price, 1e-12)
        slip_cost = (2.0 * self.slippage_points * point) / max(price, 1e-12)
        comm_cost = 2.0 * self.commission_bp_per_side * 1e-4
        return float(spread_cost + slip_cost + comm_cost)


@dataclass
class QualityConfig:
    max_na_ratio: float = 0.01
    max_duplicate_ratio: float = 0.0
    max_gap_factor: float = 1.5
    min_rows: int = 100
    min_daily_coverage: float = 0.95
    exclude_weekends: bool = True
    max_staleness_seconds: int = 1800


@dataclass
class PathsConfig:
    data_dir: str = "data"
    bronze_dir: str = "data/bronze/USDCOP"
    silver_dir: str = "data/silver/USDCOP"
    gold_dir: str = "data/gold/USDCOP"
    reports_dir: str = "data/reports/usdcop"
    logs_dir: str = "logs"


@dataclass
class AcquisitionConfig:
    timeframe: str = DEFAULT_TIMEFRAME
    bars: int = 5000
    to_timeframe: Optional[str] = None
    days_per_chunk: int = 7
    retry_chunk_failures: int = 2


@dataclass
class FeatureDefaults:
    htf_list: Tuple[str, ...] = ("M15", "H1")
    dropna: bool = True
    clip_outliers: Optional[float] = None  # z-score clip window 200 si se setea


@dataclass
class EnvRewardDefaults:
    reward_scale: float = 1.0
    turnover_penalty: float = 0.0
    holding_penalty_per_bar: float = 0.0
    dd_penalty_weight: float = 0.0
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.03
    leverage: float = 1.0


@dataclass
class SpreadProfile:
    """
    Multiplicadores heurísticos por hora local (America/Bogota) sobre el spread base.
    Ejemplo típico EM: más estrecho en solape Londres-NY (~08:00–12:00 local), más amplio fuera.
    """
    hourly_multipliers: Dict[int, float] = field(default_factory=lambda: {
        # 00-23 horas locales
        0: 1.6, 1: 1.6, 2: 1.6, 3: 1.6,
        4: 1.3, 5: 1.3, 6: 1.2,
        7: 1.1, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0,  # London–NY overlap aprox 08–12 local
        12: 1.1, 13: 1.1, 14: 1.2, 15: 1.2,
        16: 1.3, 17: 1.3, 18: 1.4, 19: 1.5,
        20: 1.5, 21: 1.6, 22: 1.6, 23: 1.6,
    })
    weekend_multiplier: float = 10.0  # por si se genera calendario 24x7 (normalmente no operamos finde)

    def spread_points_at(self, *, base_spread_points: float, dt_utc: datetime, tz: str) -> float:
        import pandas as pd
        ts = pd.Timestamp(dt_utc).tz_convert(tz)
        hour = int(ts.hour)
        mul = float(self.hourly_multipliers.get(hour, 1.3))
        if ts.weekday() >= 5:
            mul = max(mul, self.weekend_multiplier)
        return float(base_spread_points * mul)


@dataclass
class USDCOPConfig:
    symbol: SymbolConfig = field(default_factory=SymbolConfig)
    timeframes_supported: Tuple[str, ...] = tuple(TF_MIN.keys())
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    features: FeatureDefaults = field(default_factory=FeatureDefaults)
    env_reward: EnvRewardDefaults = field(default_factory=EnvRewardDefaults)
    spread_profile: SpreadProfile = field(default_factory=SpreadProfile)
    profile_name: str = "dev"  # perfil activo (solo informativo)

    # ------ Helpers ------
    def pip_config(self) -> Dict[str, Any]:
        """Devuelve dict compatible con PipConfig (metrics.py)."""
        return {
            "point": float(self.symbol.point),
            "pip_in_points": int(self.symbol.pip_in_points),
            "pip_size_price": None,
        }

    def expected_spread_points(self, dt_utc: Optional[datetime] = None) -> float:
        """Spread esperado (en 'points') para un timestamp UTC dado (o ahora si None)."""
        if dt_utc is None:
            dt_utc = datetime.now(timezone.utc)
        base = float(self.costs.default_spread_points)
        return self.spread_profile.spread_points_at(
            base_spread_points=base, dt_utc=dt_utc, tz=self.symbol.timezone
        )

    def turnover_cost_fraction(self, price: float, dt_utc: Optional[datetime] = None) -> float:
        """Coste fraccional por cambio de posición, usando spread dinámico por hora."""
        sp = self.expected_spread_points(dt_utc=dt_utc)
        return self.costs.turnover_cost_fraction(price=price, point=self.symbol.point, spread_points=sp)

    def validate(self) -> None:
        # sanity mínimo
        if self.acquisition.timeframe.upper() not in TF_MIN:
            raise ValueError(f"Timeframe no soportado: {self.acquisition.timeframe}")
        if self.symbol.digits <= 0 or self.symbol.point <= 0:
            raise ValueError("digits/point inválidos")
        if self.costs.default_spread_points <= 0:
            logger.warning("[USDCOP] default_spread_points <= 0; revise costos.")
        # paths
        for p in (self.paths.data_dir, self.paths.bronze_dir, self.paths.silver_dir, self.paths.gold_dir, self.paths.reports_dir, self.paths.logs_dir):
            # No creamos aquí; solo validamos strings
            if not isinstance(p, str) or not p:
                raise ValueError("Ruta inválida en paths.*")

# -----------------------------------------------------------------------------
# YAML loader & env expansion (compatible con scripts/run_system.py)
# -----------------------------------------------------------------------------
_ENV_PAT = r"\$\{([^}:]+)(:-([^}]*))?\}"

def _expand_env(s: str) -> str:
    import re
    def repl(m: "re.Match") -> str:
        var = m.group(1)
        default = m.group(3) if m.group(3) is not None else ""
        return os.getenv(var, default)
    return re.sub(_ENV_PAT, repl, s)

def _deep_expand(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _deep_expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_expand(v) for v in obj]
    if isinstance(obj, str):
        return _expand_env(obj)
    return obj

def _merge_dicts(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base or {})
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dicts(out[k], v)
        else:
            out[k] = v
    return out

def _load_yaml_profile(path: str) -> Dict[str, Any]:
    if not yaml or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data = _deep_expand(data)
    active = os.getenv("APP_PROFILE", data.get("active_profile", "dev"))
    defaults = data.get("defaults", {})
    profiles = data.get("profiles", {})
    prof = profiles.get(active, {})
    merged = _merge_dicts(defaults, prof)
    merged["active_profile"] = active
    return merged

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def build_default_usdcop_config() -> USDCOPConfig:
    """
    Crea una configuración por defecto coherente con buenas prácticas y
    lo que se sugiere en configs/mt5_config.yaml (si no existe el YAML).
    """
    cfg = USDCOPConfig()
    # Env overrides (opcionales):
    # spread/slippage desde ENV si están
    sp_env = os.getenv("USDCOP_SPREAD_POINTS")
    if sp_env:
        try:
            cfg.costs.default_spread_points = float(sp_env)
        except Exception:
            pass
    sl_env = os.getenv("USDCOP_SLIPPAGE_POINTS")
    if sl_env:
        try:
            cfg.costs.slippage_points = float(sl_env)
        except Exception:
            pass
    tf_env = os.getenv("USDCOP_TIMEFRAME")
    if tf_env and tf_env.upper() in TF_MIN:
        cfg.acquisition.timeframe = tf_env.upper()
    return cfg

def load_usdcop_config(yaml_path: str = "configs/mt5_config.yaml") -> USDCOPConfig:
    """
    Carga configuración efectiva:
    1) Defaults de este módulo
    2) Merge con perfiles del YAML (si existe), mapeando claves relevantes:
       - paths.*, acquisition.*, quality.*, trading_costs.*, mt5.symbols.USDCOP.*
       - health/fallback (parcialmente para status/razones en dashboard)
    """
    cfg = build_default_usdcop_config()
    y = _load_yaml_profile(yaml_path)
    if not y:
        cfg.validate()
        return cfg

    # Perfil activo (informativo)
    cfg.profile_name = y.get("active_profile", cfg.profile_name)

    # Mapear paths
    p = y.get("paths", {})
    if p:
        cfg.paths = replace(cfg.paths,
            data_dir=p.get("data_dir", cfg.paths.data_dir),
            bronze_dir=p.get("bronze_dir", cfg.paths.bronze_dir),
            silver_dir=p.get("silver_dir", cfg.paths.silver_dir),
            gold_dir=p.get("gold_dir", cfg.paths.gold_dir),
            reports_dir=p.get("reports_dir", cfg.paths.reports_dir),
            logs_dir=p.get("logs_dir", cfg.paths.logs_dir),
        )

    # Adquisición
    aq = y.get("acquisition", {})
    if aq:
        tf = aq.get("timeframe", cfg.acquisition.timeframe).upper()
        cfg.acquisition = replace(cfg.acquisition,
            timeframe=tf if tf in TF_MIN else cfg.acquisition.timeframe,
            bars=aq.get("bars", cfg.acquisition.bars),
            to_timeframe=aq.get("to_timeframe", cfg.acquisition.to_timeframe),
            days_per_chunk=aq.get("days_per_chunk", cfg.acquisition.days_per_chunk),
            retry_chunk_failures=aq.get("retry_chunk_failures", cfg.acquisition.retry_chunk_failures),
        )

    # Calidad
    q = y.get("quality", {})
    if q:
        cfg.quality = replace(cfg.quality,
            max_na_ratio=q.get("max_na_ratio", cfg.quality.max_na_ratio),
            max_duplicate_ratio=q.get("max_duplicate_ratio", cfg.quality.max_duplicate_ratio),
            max_gap_factor=q.get("max_gap_factor", cfg.quality.max_gap_factor),
            min_rows=q.get("min_rows", cfg.quality.min_rows),
            min_daily_coverage=q.get("min_daily_coverage", cfg.quality.min_daily_coverage),
            exclude_weekends=q.get("exclude_weekends", cfg.quality.exclude_weekends),
            max_staleness_seconds=q.get("max_staleness_seconds", cfg.quality.max_staleness_seconds),
        )

    # Costos (trading_costs.*)
    tc = y.get("trading_costs", {})
    if tc:
        cfg.costs = replace(cfg.costs,
            commission_bp_per_side=tc.get("commission_bp_per_side", cfg.costs.commission_bp_per_side),
            slippage_points=tc.get("slippage_points", cfg.costs.slippage_points),
            holding_cost_bp_per_bar=tc.get("holding_cost_bp_per_bar", cfg.costs.holding_cost_bp_per_bar),
        )

    # Símbolo en YAML: mt5.symbols.USDCOP
    mt5 = y.get("mt5", {})
    sym = (mt5.get("symbols", {}) or {}).get("USDCOP", {})
    if sym:
        cfg.symbol = replace(cfg.symbol,
            name=sym.get("name", cfg.symbol.name),
            digits=int(sym.get("digits", cfg.symbol.digits)),
            point=float(sym.get("point", cfg.symbol.point)),
            pip_in_points=int(sym.get("pip_in_points", cfg.symbol.pip_in_points)),
        )
        # spread base se puede configurar desde YAML
        cfg.costs.default_spread_points = float(sym.get("default_spread_points", cfg.costs.default_spread_points))

    # Preferencias horario (health/fallback) — usado para degradación si SIM en dashboard
    # (aquí no cambiamos lógica interna; el HealthMonitor lo leerá del YAML directo)
    cfg.validate()
    return cfg

# -----------------------------------------------------------------------------
# USDCOP Constants
# -----------------------------------------------------------------------------

class USDCOPConstants:
    """Single source of truth for USDCOP market constants"""
    POINT_SIZE = 0.01           # 1 point = 0.01 COP
    PIP_SIZE = 0.01             # 1 pip = 1 point for 2-digit pricing
    LOT_SIZE_UNITS = 100_000    # Standard FX lot
    DIGITS = 2                  # Price decimal places
    TIMEZONE = "America/Bogota"
    TRADING_DAYS = (0, 1, 2, 3, 4)  # Monday-Friday
    
    @classmethod
    def get_point_value(cls) -> float:
        """Get the point value for USDCOP"""
        return cls.POINT_SIZE
    
    @classmethod
    def get_pip_value(cls) -> float:
        """Get the pip value for USDCOP"""
        return cls.PIP_SIZE
    
    @classmethod
    def get_lot_size(cls) -> int:
        """Get the standard lot size for USDCOP"""
        return cls.LOT_SIZE_UNITS

# -----------------------------------------------------------------------------
# Pretty print / CLI
# -----------------------------------------------------------------------------
def to_dict(cfg: USDCOPConfig) -> Dict[str, Any]:
    out = asdict(cfg)
    return out

def _build_parser():
    import argparse
    p = argparse.ArgumentParser(description="USDCOP market configuration")
    sub = p.add_subparsers(dest="cmd", required=True)
    show = sub.add_parser("show", help="Show effective configuration as JSON")
    show.add_argument("--yaml", default="configs/mt5_config.yaml", help="mt5_config.yaml path (optional)")
    show.add_argument("--compact", action="store_true")
    return p

def main():
    parser = _build_parser()
    args = parser.parse_args()
    cfg = load_usdcop_config(args.yaml) if args.yaml else build_default_usdcop_config()
    d = to_dict(cfg)
    if args.compact:
        print(json.dumps(d, separators=(",", ":"), ensure_ascii=False))
    else:
        print(json.dumps(d, indent=2, ensure_ascii=False))

if __name__ == "__main__":  # pragma: no cover
    main()