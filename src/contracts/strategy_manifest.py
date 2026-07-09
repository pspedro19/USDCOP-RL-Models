"""Strategy Bundle Manifest & Dynamic Registry contract.

Implements CTR-STRAT-REGISTRY-001 (see .claude/rules/sdd-strategy-lifecycle-registry.md).

This module is the SSOT for how a strategy describes itself to the frontend so the UI can be
built dynamically (multi-strategy, multi-asset, multi-version, multi-year) with zero hardcoded
strategy_id / symbol / year.

Two artifacts:
  - StrategyBundleManifest  -> public/data/strategies/<strategy_id>/manifest.json
  - RegistryIndex           -> public/data/registry.json  (dynamic discovery index)

Design (SOLID):
  - Dataclasses = the contract (pure data, no I/O).
  - LegacyBundleAdapter = single responsibility: legacy production/*.json -> manifest (shim).
  - RegistryBuilder = single responsibility: discover manifests -> registry index.

DRY: JSON safety is delegated to strategy_schema.safe_json_dump (never re-implemented here).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# NOTE (intentional decoupling): the registry/manifest tool is a JSON-only leaf. It must run
# WITHOUT the ML/forecasting stack (which `src.contracts.__init__` eager-imports). So JSON safety
# is inlined here rather than imported from strategy_schema — it mirrors
# strategy_schema.safe_json_dump exactly (Infinity/NaN -> null). Keep the two in lockstep.


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively replace Infinity/NaN floats with None so JSON.parse() never breaks."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def safe_json_dump(data: Any, fp: Any, **kwargs: Any) -> None:
    """JSON dump that never emits Infinity/NaN. Mirrors strategy_schema.safe_json_dump."""
    kwargs.setdefault("indent", 2)
    json.dump(_sanitize_for_json(data), fp, allow_nan=False, **kwargs)


SCHEMA_VERSION = "1.0.0"

# Asset display metadata. USD/COP is the only tradeable asset today; new assets (XAU, BTC)
# arrive via config/assets/<asset_id>.yaml (AssetProfile) — see sdd-multi-asset-onboarding.md.
# chart_symbol is what TradingChartWithSignals must render (no slash).
_DEFAULT_ASSETS: dict[str, dict[str, str]] = {
    "usdcop": {"symbol": "USD/COP", "chart_symbol": "USDCOP", "display_name": "USD/COP", "asset_class": "fx"},
    "xauusd": {"symbol": "XAU/USD", "chart_symbol": "XAUUSD", "display_name": "Gold", "asset_class": "commodity"},
    "btcusd": {"symbol": "BTC/USD", "chart_symbol": "BTCUSD", "display_name": "Bitcoin", "asset_class": "crypto"},
    "btcusdt": {"symbol": "BTC/USDT", "chart_symbol": "BTCUSDT", "display_name": "Bitcoin", "asset_class": "crypto"},
}


def chart_symbol_for(symbol: str) -> str:
    """Derive the UI chart symbol from a provider symbol (e.g. 'USD/COP' -> 'USDCOP')."""
    return symbol.replace("/", "").replace(" ", "").upper()


# ---------------------------------------------------------------------------
# Contract dataclasses
# ---------------------------------------------------------------------------
@dataclass
class BacktestEntry:
    """One immutable backtest, keyed by (model_version, year). NEVER overwritten (spec §5)."""

    model_version: str
    year: int
    immutable_id: str
    summary: str  # path relative to public/data/
    trades: str
    signals: str | None = None  # present iff replayable
    replayable: bool = False
    gates: dict[str, Any] = field(default_factory=dict)  # {passed, of, recommendation}
    headline: dict[str, Any] = field(default_factory=dict)  # {return_pct, sharpe, p_value}


@dataclass
class ModelVersionEntry:
    version: str
    active: bool = False
    trained_at: str | None = None
    train_window: str | None = None
    feature_hash: str | None = None
    norm_stats_hash: str | None = None
    artifact_uri: str | None = None


@dataclass
class StrategyBundleManifest:
    """Self-describing strategy bundle — the only thing the UI needs to render a strategy."""

    strategy_id: str
    asset_id: str
    symbol: str
    chart_symbol: str
    display_name: str
    pipeline_type: str  # ml_forecasting | rl | rule_based | hybrid
    timeframe: str  # weekly | daily | intraday_5m
    status: str  # experimental | paper | production | archived
    schema_version: str = SCHEMA_VERSION
    capabilities: dict[str, bool] = field(default_factory=lambda: {"replay": False, "live": False, "approval": True})
    produced_by: dict[str, Any] = field(default_factory=dict)
    backtests: list[BacktestEntry] = field(default_factory=list)
    production: dict[str, Any] | None = None
    approval: dict[str, Any] | None = None
    model_versions: list[ModelVersionEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _sanitize_for_json(asdict(self))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "StrategyBundleManifest":
        backtests = [BacktestEntry(**b) for b in raw.get("backtests", [])]
        model_versions = [ModelVersionEntry(**m) for m in raw.get("model_versions", [])]
        known = set(cls.__dataclass_fields__)
        base = {k: v for k, v in raw.items() if k in known and k not in ("backtests", "model_versions")}
        return cls(**base, backtests=backtests, model_versions=model_versions)

    @property
    def backtest_years(self) -> list[int]:
        return sorted({b.year for b in self.backtests})

    @property
    def versions(self) -> list[str]:
        return sorted({b.model_version for b in self.backtests})

    @property
    def active_version(self) -> str | None:
        for mv in self.model_versions:
            if mv.active:
                return mv.version
        return self.model_versions[-1].version if self.model_versions else None

    @property
    def active_backtest(self) -> "BacktestEntry | None":
        """The backtest entry for the active version (prefers a replayable one), else the last."""
        av = self.active_version
        matches = [b for b in self.backtests if b.model_version == av]
        for b in matches:
            if b.replayable:
                return b
        return matches[0] if matches else (self.backtests[-1] if self.backtests else None)


@dataclass
class RegistryStrategyEntry:
    strategy_id: str
    asset_id: str
    status: str
    display_name: str
    pipeline_type: str
    timeframe: str
    manifest: str  # path relative to public/data/
    backtest_years: list[int] = field(default_factory=list)
    has_production: bool = False
    has_replay: bool = False
    # Active-version headline metrics — let the frontend strategy selector show real numbers
    # without an N+1 manifest fetch. All optional (default None) → additive, backward-compatible.
    active_version: str | None = None
    return_pct: float | None = None
    sharpe: float | None = None
    p_value: float | None = None


@dataclass
class RegistryIndex:
    """The dynamic index the frontend fetches to build all selectors (spec §4)."""

    generated_at: str
    assets: list[dict[str, str]]
    strategies: list[RegistryStrategyEntry]
    default: dict[str, str]
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return _sanitize_for_json(asdict(self))


# ---------------------------------------------------------------------------
# LegacyBundleAdapter — synthesize a manifest from the current flat layout (migration shim)
# ---------------------------------------------------------------------------
class LegacyBundleAdapter:
    """Builds a StrategyBundleManifest from the legacy public/data/production/*.json layout.

    This is the non-destructive migration shim (spec §11 step 1): it references the EXISTING
    legacy files so the current USDCOP dashboard keeps working while gaining a dynamic registry.
    Paths in the manifest are relative to `public/data/`.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.prod_dir = data_dir / "production"

    def _load_json(self, rel: str) -> dict[str, Any] | None:
        p = self.data_dir / rel
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def _discover_backtest_years(self, strategy_id: str) -> list[int]:
        years: set[int] = set()
        for f in self.prod_dir.glob("summary_*.json"):
            stem = f.stem  # summary_2025
            tail = stem.rsplit("_", 1)[-1]
            if tail.isdigit():
                years.add(int(tail))
        return sorted(years)

    def build(self, entry: dict[str, Any]) -> StrategyBundleManifest:
        strategy_id = entry["strategy_id"]
        asset_id = entry.get("asset_id", "usdcop")
        asset = _DEFAULT_ASSETS.get(asset_id, _DEFAULT_ASSETS["usdcop"])
        symbol = entry.get("symbol", asset["symbol"])

        approval = self._load_json("production/approval_state.json") or {}
        model_version = entry.get("version") or "1.0.0"

        backtests: list[BacktestEntry] = []
        for year in self._discover_backtest_years(strategy_id):
            summary = self._load_json(f"production/summary_{year}.json") or {}
            stats = (summary.get("strategies") or {}).get(strategy_id, {})
            trades_rel = f"production/trades/{strategy_id}_{year}.json"
            has_trades = (self.data_dir / trades_rel).exists()
            gates = approval.get("gates", []) if approval.get("backtest_year") == year else []
            backtests.append(
                BacktestEntry(
                    model_version=model_version,
                    year=year,
                    immutable_id=f"{strategy_id}__{model_version}__{year}",
                    summary=f"production/summary_{year}.json",
                    trades=trades_rel if has_trades else "",
                    signals=None,  # legacy has no stored signals parquet yet -> replay disabled
                    replayable=False,
                    gates={
                        "passed": sum(1 for g in gates if g.get("passed")),
                        "of": len(gates),
                        "recommendation": approval.get("backtest_recommendation", "REVIEW"),
                    },
                    headline={
                        "return_pct": stats.get("total_return_pct"),
                        "sharpe": stats.get("sharpe"),
                        "p_value": (summary.get("statistical_tests") or {}).get("p_value"),
                    },
                )
            )

        production = None
        if (self.data_dir / "production/summary.json").exists():
            prod_summary = self._load_json("production/summary.json") or {}
            production = {
                "model_version": model_version,
                "year": prod_summary.get("year"),
                "summary": "production/summary.json",
                "trades": f"production/trades/{strategy_id}.json",
            }

        # Lifecycle status is distinct from approval status. The legacy strategies.json `status`
        # field carries the APPROVAL state ("APPROVED"/"PENDING_APPROVAL"), not lifecycle. Derive
        # lifecycle: an approved strategy with production output is "production", else "paper".
        approval_status = approval.get("status", "PENDING_APPROVAL")
        lifecycle_status = "production" if (production is not None and approval_status == "APPROVED") else "paper"

        return StrategyBundleManifest(
            strategy_id=strategy_id,
            asset_id=asset_id,
            symbol=symbol,
            chart_symbol=asset["chart_symbol"],
            display_name=entry.get("strategy_name", strategy_id),
            pipeline_type=entry.get("pipeline", entry.get("pipeline_type", "ml_forecasting")),
            timeframe=entry.get("timeframe", "weekly"),
            status=lifecycle_status,
            capabilities={"replay": any(b.replayable for b in backtests), "live": production is not None, "approval": True},
            produced_by={"source": "legacy_shim", "adapter": "LegacyBundleAdapter"},
            backtests=backtests,
            production=production,
            approval={"file": "production/approval_state.json", "status": approval.get("status", "PENDING_APPROVAL")},
            model_versions=[ModelVersionEntry(version=model_version, active=True)],
        )


# ---------------------------------------------------------------------------
# RegistryBuilder — discover manifests -> registry index
# ---------------------------------------------------------------------------
class RegistryBuilder:
    """Scans the dashboard data dir and produces a dynamic registry index.

    Discovery order (both supported, so new + legacy coexist during migration):
      1. New layout:    public/data/strategies/<sid>/manifest.json
      2. Legacy layout: public/data/production/strategies.json (synthesized via LegacyBundleAdapter)
    """

    def __init__(self, data_dir: Path, generated_at: str):
        self.data_dir = data_dir
        self.generated_at = generated_at
        self.strategies_dir = data_dir / "strategies"
        self.legacy = LegacyBundleAdapter(data_dir)

    def _load_manifest(self, path: Path) -> StrategyBundleManifest | None:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        backtests = [BacktestEntry(**b) for b in raw.get("backtests", [])]
        model_versions = [ModelVersionEntry(**m) for m in raw.get("model_versions", [])]
        known = {f.name for f in StrategyBundleManifest.__dataclass_fields__.values()}
        base = {k: v for k, v in raw.items() if k in known and k not in ("backtests", "model_versions")}
        return StrategyBundleManifest(**base, backtests=backtests, model_versions=model_versions)

    def _synthesize_legacy_manifests(self) -> list[StrategyBundleManifest]:
        legacy_index = self.legacy._load_json("production/strategies.json") or {}
        manifests: list[StrategyBundleManifest] = []
        for entry in legacy_index.get("strategies", []):
            sid = entry.get("strategy_id")
            if not sid:
                continue
            # Do not clobber a hand-authored/new-layout manifest.
            if (self.strategies_dir / sid / "manifest.json").exists():
                continue
            manifests.append(self.legacy.build(entry))
        return manifests

    def build(self, *, write_manifests: bool = True) -> RegistryIndex:
        manifests: dict[str, StrategyBundleManifest] = {}

        # 1. New-layout manifests (authoritative).
        if self.strategies_dir.exists():
            for mf in self.strategies_dir.glob("*/manifest.json"):
                m = self._load_manifest(mf)
                if m:
                    manifests[m.strategy_id] = m

        # 2. Legacy shim (only for strategies without a new-layout manifest).
        for m in self._synthesize_legacy_manifests():
            manifests.setdefault(m.strategy_id, m)
            if write_manifests:
                out = self.strategies_dir / m.strategy_id / "manifest.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                with out.open("w", encoding="utf-8") as f:
                    safe_json_dump(m.to_dict(), f, indent=2)

        # Assemble assets referenced by the discovered strategies.
        asset_ids = {m.asset_id for m in manifests.values()} or {"usdcop"}
        assets = []
        for aid in sorted(asset_ids):
            meta = _DEFAULT_ASSETS.get(aid, {"symbol": aid.upper(), "chart_symbol": aid.upper(), "display_name": aid.upper(), "asset_class": "unknown"})
            assets.append({"asset_id": aid, **meta})

        def _entry(m: StrategyBundleManifest) -> RegistryStrategyEntry:
            ab = m.active_backtest
            h = ab.headline if ab else {}
            return RegistryStrategyEntry(
                strategy_id=m.strategy_id,
                asset_id=m.asset_id,
                status=m.status,
                display_name=m.display_name,
                pipeline_type=m.pipeline_type,
                timeframe=m.timeframe,
                manifest=f"strategies/{m.strategy_id}/manifest.json",
                backtest_years=m.backtest_years,
                has_production=m.production is not None,
                has_replay=m.capabilities.get("replay", False),
                active_version=m.active_version,
                return_pct=h.get("return_pct"),
                sharpe=h.get("sharpe"),
                p_value=h.get("p_value"),
            )

        strategies = [_entry(m) for m in sorted(manifests.values(), key=lambda x: x.strategy_id)]

        # Default: first production strategy, else first discovered.
        default_m = next((m for m in manifests.values() if m.status == "production"), None) or next(iter(manifests.values()), None)
        default = (
            {"asset_id": default_m.asset_id, "strategy_id": default_m.strategy_id}
            if default_m
            else {"asset_id": "usdcop", "strategy_id": "smart_simple_v11"}
        )

        return RegistryIndex(generated_at=self.generated_at, assets=assets, strategies=strategies, default=default)

    def write(self, index: RegistryIndex) -> Path:
        out = self.data_dir / "registry.json"
        with out.open("w", encoding="utf-8") as f:
            safe_json_dump(index.to_dict(), f, indent=2)
        return out


# ---------------------------------------------------------------------------
# BundlePublisher — publish a training run's backtest as an immutable, versioned bundle
# ---------------------------------------------------------------------------
class BundlePublisher:
    """Publishes one training run's backtest as an IMMUTABLE, versioned bundle and upserts the
    per-strategy manifest, then refreshes the registry index.

    Implements the immutability keystone (CTR-STRAT-REGISTRY-001 §5): a backtest is content-
    addressed by (strategy_id, model_version, year) and is NEVER overwritten. Training a new
    model version (e.g. changed hyperparameters/features -> bumped config version) publishes a
    NEW immutable entry that coexists with prior versions -> the frontend can list every version
    and replay each independently.

    ADDITIVE / non-breaking: this writes ONLY under strategies/<sid>/... and registry.json.
    It never touches the legacy production/*.json files the current frontend consumes.
    """

    def __init__(self, data_dir: Path, generated_at: str):
        self.data_dir = Path(data_dir)
        self.generated_at = generated_at

    def _bundle_dir(self, sid: str) -> Path:
        return self.data_dir / "strategies" / sid

    def _manifest_path(self, sid: str) -> Path:
        return self._bundle_dir(sid) / "manifest.json"

    @staticmethod
    def _write_immutable(path: Path, payload: Any) -> bool:
        """Write JSON only if the target does not already exist. Returns True if written,
        False if it already existed (immutability preserved)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return False
        with path.open("w", encoding="utf-8") as f:
            safe_json_dump(payload, f, indent=2)
        return True

    def _load_manifest(self, sid: str) -> StrategyBundleManifest | None:
        p = self._manifest_path(sid)
        if not p.exists():
            return None
        try:
            return StrategyBundleManifest.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError, TypeError):
            return None

    def publish(
        self,
        *,
        strategy_id: str,
        asset_id: str,
        symbol: str,
        display_name: str,
        pipeline_type: str,
        timeframe: str,
        version: str,
        year: int,
        summary: dict[str, Any],
        trades: dict[str, Any],
        gates: dict[str, Any] | None = None,
        headline: dict[str, Any] | None = None,
        signals: Any | None = None,
        status: str = "experimental",
        trained_at: str | None = None,
        refresh_registry: bool = True,
        phase: str = "backtest",
    ) -> dict[str, Any]:
        """Publish (strategy_id, version, year). Returns a summary dict.

        phase="backtest" (default): immutable, content-addressed write under
        backtests/<version>/ — a frozen OOS result never changes.
        phase="production" (audit A4-01): the LIVE year is MUTABLE by nature (it
        grows weekly) — write/overwrite strategies/<sid>/production/*.json and set
        the manifest's `production` pointer; never route the live year through the
        immutable path (the first weekly write would freeze it forever).
        """
        sid = strategy_id
        chart = (_DEFAULT_ASSETS.get(asset_id) or {}).get("chart_symbol") or chart_symbol_for(symbol)

        if phase == "production":
            return self._publish_production(
                sid=sid, asset_id=asset_id, symbol=symbol, chart=chart,
                display_name=display_name, pipeline_type=pipeline_type,
                timeframe=timeframe, version=version, year=year,
                summary=summary, trades=trades, signals=signals,
                status=status, trained_at=trained_at, refresh_registry=refresh_registry,
            )

        vdir = self._bundle_dir(sid) / "backtests" / version

        summary_rel = f"strategies/{sid}/backtests/{version}/summary_{year}.json"
        trades_rel = f"strategies/{sid}/backtests/{version}/trades_{year}.json"
        wrote_summary = self._write_immutable(vdir / f"summary_{year}.json", summary)
        wrote_trades = self._write_immutable(vdir / f"trades_{year}.json", trades)

        signals_rel: str | None = None
        if signals is not None:
            signals_rel = f"strategies/{sid}/backtests/{version}/signals_{year}.json"
            self._write_immutable(vdir / f"signals_{year}.json", signals)

        replayable = (vdir / f"trades_{year}.json").exists()
        entry = BacktestEntry(
            model_version=version,
            year=year,
            immutable_id=f"{sid}__{version}__{year}",
            summary=summary_rel,
            trades=trades_rel,
            signals=signals_rel,
            replayable=replayable,
            gates=gates or {},
            headline=headline or {},
        )

        # Load existing manifest (keeps prior versions) or create a new one.
        m = self._load_manifest(sid) or StrategyBundleManifest(
            strategy_id=sid,
            asset_id=asset_id,
            symbol=symbol,
            chart_symbol=chart,
            display_name=display_name,
            pipeline_type=pipeline_type,
            timeframe=timeframe,
            status=status,
        )
        # Upsert the (version, year) backtest entry — coexistence across versions.
        m.backtests = [b for b in m.backtests if not (b.model_version == version and b.year == year)]
        m.backtests.append(entry)
        # Upsert the model version; the just-published one becomes the active version.
        m.model_versions = [mv for mv in m.model_versions if mv.version != version]
        for mv in m.model_versions:
            mv.active = False
        m.model_versions.append(ModelVersionEntry(version=version, active=True, trained_at=trained_at))
        m.capabilities["replay"] = any(b.replayable for b in m.backtests)
        m.produced_by = {"source": "BundlePublisher", "version": version, "year": year, "generated_at": self.generated_at}

        with self._manifest_path(sid).open("w", encoding="utf-8") as f:
            safe_json_dump(m.to_dict(), f, indent=2)

        if refresh_registry:
            RegistryBuilder(self.data_dir, generated_at=self.generated_at).write(
                RegistryBuilder(self.data_dir, generated_at=self.generated_at).build(write_manifests=False)
            )

        return {
            "strategy_id": sid,
            "version": version,
            "year": year,
            "wrote_new_files": bool(wrote_summary and wrote_trades),
            "immutable_hit": not (wrote_summary and wrote_trades),
            "versions": m.versions,
            "manifest": str(self._manifest_path(sid)),
        }

    def _publish_production(
        self,
        *,
        sid: str,
        asset_id: str,
        symbol: str,
        chart: str,
        display_name: str,
        pipeline_type: str,
        timeframe: str,
        version: str,
        year: int,
        summary: dict[str, Any],
        trades: dict[str, Any],
        signals: Any | None,
        status: str,
        trained_at: str | None,
        refresh_registry: bool,
    ) -> dict[str, Any]:
        """MUTABLE production publish (audit A4-01): overwrite the live-year files under
        strategies/<sid>/production/ and set the manifest `production` pointer. The live
        year grows weekly — freezing it in the immutable backtests/ path silently dropped
        every weekly update after the first."""
        pdir = self._bundle_dir(sid) / "production"
        pdir.mkdir(parents=True, exist_ok=True)

        summary_rel = f"strategies/{sid}/production/summary.json"
        trades_rel = f"strategies/{sid}/production/trades.json"
        with (pdir / "summary.json").open("w", encoding="utf-8") as f:
            safe_json_dump(summary, f, indent=2)
        with (pdir / "trades.json").open("w", encoding="utf-8") as f:
            safe_json_dump(trades, f, indent=2)
        signals_rel: str | None = None
        if signals is not None:
            signals_rel = f"strategies/{sid}/production/signals.json"
            with (pdir / "signals.json").open("w", encoding="utf-8") as f:
                safe_json_dump(signals, f, indent=2)

        m = self._load_manifest(sid) or StrategyBundleManifest(
            strategy_id=sid,
            asset_id=asset_id,
            symbol=symbol,
            chart_symbol=chart,
            display_name=display_name,
            pipeline_type=pipeline_type,
            timeframe=timeframe,
            status=status,
        )
        m.production = {
            "model_version": version,
            "year": year,
            "summary": summary_rel,
            "trades": trades_rel,
            **({"signals": signals_rel} if signals_rel else {}),
            "updated_at": self.generated_at,
        }
        # Upsert the model version; the deployed one becomes the active version.
        m.model_versions = [mv for mv in m.model_versions if mv.version != version]
        for mv in m.model_versions:
            mv.active = False
        m.model_versions.append(ModelVersionEntry(version=version, active=True, trained_at=trained_at))
        m.produced_by = {"source": "BundlePublisher", "version": version, "year": year,
                         "phase": "production", "generated_at": self.generated_at}

        with self._manifest_path(sid).open("w", encoding="utf-8") as f:
            safe_json_dump(m.to_dict(), f, indent=2)

        if refresh_registry:
            RegistryBuilder(self.data_dir, generated_at=self.generated_at).write(
                RegistryBuilder(self.data_dir, generated_at=self.generated_at).build(write_manifests=False)
            )

        return {
            "strategy_id": sid,
            "version": version,
            "year": year,
            "phase": "production",
            "wrote_new_files": True,
            "immutable_hit": False,
            "versions": m.versions,
            "manifest": str(self._manifest_path(sid)),
        }
