#!/usr/bin/env python
"""Backfill replay facts and quarantined legacy observations for every strategy.

Archived, withdrawn, experimental and champion entries are intentionally
treated alike.  The catalog is the population; status is a dimension, never a
filter.  The command plans by default and writes only with ``--apply``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "usdcop-trading-dashboard" / "public" / "data"


@dataclass(frozen=True)
class MetricFact:
    strategy_id: str
    asset_id: str
    status: str
    version: str
    year: int
    name: str
    value: float
    source: str

    @property
    def run_id(self) -> str:
        return f"catalog-backfill:{self.strategy_id}:{self.version}:{self.year}"

    @property
    def event_id(self) -> str:
        key = f"{self.run_id}:{self.name}:{self.source}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


@dataclass(frozen=True)
class TradeFact:
    strategy_id: str
    asset_id: str
    version: str
    year: int
    trade_id: str
    entry_time: datetime
    exit_time: datetime
    side: str
    entry_price: float
    exit_price: float
    pnl: float
    equity_at_entry: float
    leverage: float
    currency: str
    source: str

    @property
    def fill_set_id(self) -> str:
        return f"legacy:{self.strategy_id}:{self.version}:{self.year}:{self.trade_id}"

    @property
    def derivation_id(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True, default=str).encode()
        return "sha256:" + hashlib.sha256(payload).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _public_path(reference: Any, *, field: str) -> Path:
    """Resolve a registry reference without allowing it to escape ``PUBLIC``."""
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError(f"{field} must be a non-empty relative path")
    public_root = PUBLIC.resolve()
    candidate = (public_root / reference).resolve()
    try:
        candidate.relative_to(public_root)
    except ValueError as exc:
        raise ValueError(
            f"{field} must resolve under the configured PUBLIC root"
        ) from exc
    return candidate


def _source_uri(path: Path) -> str:
    """Keep production URIs unchanged and isolated-test URIs machine-independent."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        try:
            relative = resolved.relative_to(PUBLIC.resolve())
        except ValueError as exc:  # defensive: callers must use _public_path first
            raise ValueError(
                "source path must be under the repository or configured PUBLIC root"
            ) from exc
        return f"public-data:///{relative.as_posix()}"


def _metric_values(
    summary: dict[str, Any], strategy_id: str
) -> Iterable[tuple[str, float]]:
    candidate = (summary.get("strategies") or {}).get(strategy_id, {})
    headline = summary.get("headline") or {}
    oos = (summary.get("oos") or {}).get("metrics", {})
    if "n_trades" in candidate:
        n_trades = candidate["n_trades"]
    elif "n_long" in candidate or "n_short" in candidate:
        n_trades = (candidate.get("n_long") or 0) + (candidate.get("n_short") or 0)
    else:
        n_trades = headline.get("n_trades")
    aliases = {
        "sharpe": candidate.get("sharpe", oos.get("sharpe", headline.get("sharpe"))),
        "sortino": candidate.get("sortino", oos.get("sortino")),
        "calmar": candidate.get("calmar", oos.get("calmar")),
        "max_drawdown_pct": candidate.get(
            "max_dd_pct", oos.get("max_dd", headline.get("max_dd_pct"))
        ),
        "total_return_pct": candidate.get(
            "total_return_pct", oos.get("total_return_pct", headline.get("total_return_pct"))
        ),
        "n_trades": n_trades,
        "p_value": (summary.get("statistical_tests") or {}).get("p_value"),
        "dsr": ((summary.get("statistical_tests") or {}).get("deflated_sharpe") or {}).get("dsr"),
    }
    for name, value in aliases.items():
        if isinstance(value, (int, float)) and value == value and abs(float(value)) != float("inf"):
            yield name, float(value)


def _parse_time(value: Any) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("trade timestamp is missing")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def inventory(
    registry_path: Path,
) -> tuple[list[MetricFact], list[TradeFact], list[str], list[str]]:
    registry = _json(registry_path)
    facts: list[MetricFact] = []
    trades: list[TradeFact] = []
    missing: list[str] = []
    population = [str(strategy["strategy_id"]) for strategy in registry.get("strategies", [])]
    for strategy in registry.get("strategies", []):
        strategy_id = str(strategy["strategy_id"])
        manifest_path = _public_path(
            strategy.get("manifest"), field=f"{strategy_id}.manifest"
        )
        if not manifest_path.is_file():
            missing.append(f"{strategy_id}: missing {_source_uri(manifest_path)}")
            continue
        manifest = _json(manifest_path)
        for backtest in manifest.get("backtests", []):
            summary_path = _public_path(
                backtest.get("summary"), field=f"{strategy_id}.backtest.summary"
            )
            if not summary_path.is_file():
                missing.append(f"{strategy_id}: missing {_source_uri(summary_path)}")
                continue
            summary = _json(summary_path)
            version = str(backtest.get("model_version") or strategy.get("active_version") or "unknown")
            year = int(backtest["year"])
            for metric_name, value in _metric_values(summary, strategy_id):
                facts.append(
                    MetricFact(
                        strategy_id=strategy_id,
                        asset_id=str(strategy["asset_id"]),
                        status=str(strategy.get("status", "unknown")),
                        version=version,
                        year=year,
                        name=metric_name,
                        value=value,
                        source=_source_uri(summary_path),
                    )
                )
            # Baselines are part of the anti-survivorship population too.  They
            # remain namespaced under the owning strategy to avoid pretending
            # they are independently governed sleeves.
            for baseline_id in sorted((summary.get("strategies") or {})):
                if baseline_id == strategy_id:
                    continue
                for metric_name, value in _metric_values(summary, baseline_id):
                    facts.append(
                        MetricFact(
                            strategy_id=f"{strategy_id}::baseline::{baseline_id}",
                            asset_id=str(strategy["asset_id"]),
                            status="baseline",
                            version=version,
                            year=year,
                            name=metric_name,
                            value=value,
                            source=_source_uri(summary_path),
                        )
                    )
            trades_ref = backtest.get("trades")
            if not trades_ref:
                continue
            trades_path = _public_path(
                trades_ref, field=f"{strategy_id}.backtest.trades"
            )
            if not trades_path.is_file():
                missing.append(f"{strategy_id}: missing {_source_uri(trades_path)}")
                continue
            trades_doc = _json(trades_path)
            for raw_trade in trades_doc.get("trades", []):
                try:
                    trades.append(
                        TradeFact(
                            strategy_id=strategy_id,
                            asset_id=str(strategy["asset_id"]),
                            version=version,
                            year=year,
                            trade_id=str(raw_trade["trade_id"]),
                            entry_time=_parse_time(raw_trade["timestamp"]),
                            exit_time=_parse_time(raw_trade["exit_timestamp"]),
                            side=str(raw_trade["side"]).upper(),
                            entry_price=float(raw_trade["entry_price"]),
                            exit_price=float(raw_trade["exit_price"]),
                            pnl=float(raw_trade["pnl_usd"]),
                            equity_at_entry=float(raw_trade["equity_at_entry"]),
                            leverage=float(raw_trade.get("leverage", 1.0)),
                            currency="USD",
                            source=_source_uri(trades_path),
                        )
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    missing.append(
                        f"{strategy_id}: invalid trade {raw_trade.get('trade_id')}: {exc}"
                    )
    return facts, trades, missing, population


def _instrument_ids(cursor: Any, asset_ids: Iterable[str]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for asset_id in sorted(set(asset_ids)):
        cursor.execute(
            """
            SELECT instrument_id::text
            FROM reference.instrument
            WHERE asset_id = %s AND active
            ORDER BY canonical_symbol
            """,
            (asset_id,),
        )
        rows = cursor.fetchall()
        if len(rows) != 1:
            raise RuntimeError(
                f"{asset_id}: expected exactly one active canonical instrument, got {len(rows)}"
            )
        resolved[asset_id] = rows[0][0]
    return resolved


def _write(
    connection: Any, facts: Iterable[MetricFact], trades: Iterable[TradeFact]
) -> dict[str, int]:
    metric_count = 0
    pnl_count = 0
    position_count = 0
    facts = list(facts)
    trades = list(trades)
    with connection.cursor() as cursor:
        instruments = _instrument_ids(cursor, [trade.asset_id for trade in trades])
        for fact in facts:
            event_time = datetime(fact.year, 12, 31, tzinfo=timezone.utc)
            entity_id = f"{fact.strategy_id}:{fact.version}:{fact.year}"
            cursor.execute(
                """
                INSERT INTO control.legacy_metric_observation (
                    observation_id, observed_at, entity_type, entity_id,
                    strategy_id, asset_id, run_id, environment,
                    legacy_metric_name, observed_value, source_uri, details
                ) VALUES (
                    %s, %s, 'strategy_backtest', %s,
                    %s, %s, %s, 'backtest', %s, %s, %s, %s::jsonb
                )
                ON CONFLICT (observation_id) DO NOTHING
                """,
                (
                    fact.event_id,
                    event_time,
                    entity_id,
                    fact.strategy_id,
                    fact.asset_id,
                    fact.run_id,
                    fact.name,
                    fact.value,
                    fact.source,
                    json.dumps(
                        {
                            "registry_status": fact.status,
                            "year": fact.year,
                            "version": fact.version,
                            "anti_survivorship_population": True,
                            "reason": "legacy value is not a governed MetricEngine event",
                        }
                    ),
                ),
            )
            metric_count += cursor.rowcount
        pnl_groups: dict[tuple[str, str, datetime, str, int], list[TradeFact]] = {}
        position_events: dict[
            tuple[str, str, datetime], list[tuple[float, float, TradeFact]]
        ] = {}
        for trade in trades:
            pnl_groups.setdefault(
                (
                    trade.strategy_id,
                    trade.asset_id,
                    trade.exit_time,
                    trade.version,
                    trade.year,
                ),
                [],
            ).append(trade)
            signed_qty = (
                trade.equity_at_entry
                * trade.leverage
                / trade.entry_price
                * (1.0 if trade.side == "LONG" else -1.0)
            )
            position_events.setdefault(
                (trade.strategy_id, trade.asset_id, trade.entry_time), []
            ).append((signed_qty, trade.entry_price, trade))
            position_events.setdefault(
                (trade.strategy_id, trade.asset_id, trade.exit_time), []
            ).append((-signed_qty, trade.exit_price, trade))

        for (strategy_id, asset_id, exit_time, version, year), grouped in sorted(
            pnl_groups.items(), key=lambda item: item[0]
        ):
            instrument_id = instruments[asset_id]
            fill_ids = sorted(item.fill_set_id for item in grouped)
            fill_set_id = "legacy-set:sha256:" + hashlib.sha256(
                "\n".join(fill_ids).encode()
            ).hexdigest()
            derivation = "sha256:" + hashlib.sha256(
                json.dumps(
                    [item.__dict__ for item in grouped],
                    sort_keys=True,
                    default=str,
                ).encode()
            ).hexdigest()
            amount = sum(item.pnl for item in grouped)
            nav_amount = max(abs(item.equity_at_entry) for item in grouped)
            if nav_amount <= 0:
                raise RuntimeError(
                    f"non-positive legacy NAV for {strategy_id}/{asset_id}/{exit_time}"
                )
            # Legacy bundles do not carry sufficient attribution to classify
            # beta/timing/carry.  Preserve the honest accounting identity by
            # assigning the amount to residual, never by fabricating timing alpha.
            for component in ("gross_pnl", "pnl_residual"):
                cursor.execute(
                    """
                    INSERT INTO fact.pnl (
                        as_of, strategy_id, sleeve_id, instrument_id, environment,
                        pnl_component, amount,
                        nav_amount,
                        currency, attribution_model_version, benchmark_id,
                        source_fill_set_id, reconciliation_status, run_id, derivation_id
                    ) VALUES (
                        %s, %s, %s, %s, 'backtest', %s, %s, %s, %s,
                        'legacy_residual_v1', 'UNATTRIBUTED', %s, 'PENDING', %s, %s
                    )
                    ON CONFLICT DO NOTHING
                    """,
                    (
                        exit_time,
                        strategy_id,
                        strategy_id,
                        instrument_id,
                        component,
                        amount,
                        nav_amount,
                        grouped[0].currency,
                        fill_set_id,
                        f"catalog-backfill:{strategy_id}:{version}:{year}",
                        derivation,
                    ),
                )
                pnl_count += cursor.rowcount
        running_qty: dict[tuple[str, str], float] = {}
        for (strategy_id, asset_id, at), events in sorted(
            position_events.items(), key=lambda item: item[0]
        ):
            key = (strategy_id, asset_id)
            qty = running_qty.get(key, 0.0) + sum(delta for delta, _, _ in events)
            running_qty[key] = qty
            events = sorted(events, key=lambda item: item[2].fill_set_id)
            price = events[-1][1]
            grouped = [event[2] for event in events]
            fill_ids = sorted(item.fill_set_id for item in grouped)
            fill_set_id = "legacy-set:sha256:" + hashlib.sha256(
                "\n".join(fill_ids).encode()
            ).hexdigest()
            derivation = "sha256:" + hashlib.sha256(
                json.dumps(
                    [item.__dict__ for item in grouped],
                    sort_keys=True,
                    default=str,
                ).encode()
            ).hexdigest()
            cursor.execute(
                """
                INSERT INTO fact.position (
                    as_of, strategy_id, sleeve_id, instrument_id, environment,
                    qty, average_cost,
                    market_price, market_value, currency, source_fill_set_id,
                    reconciliation_status, run_id, derivation_id
                ) VALUES (
                    %s, %s, %s, %s, 'backtest', %s, NULL, %s, %s, %s,
                    %s, 'PENDING', %s, %s
                )
                ON CONFLICT DO NOTHING
                """,
                (
                    at,
                    strategy_id,
                    strategy_id,
                    instruments[asset_id],
                    qty,
                    price,
                    qty * price,
                    grouped[0].currency,
                    fill_set_id,
                    f"catalog-backfill:{strategy_id}:{grouped[0].version}:{grouped[0].year}",
                    derivation,
                ),
            )
            position_count += cursor.rowcount
    connection.commit()
    return {
        "metric_events": metric_count,
        "pnl_facts": pnl_count,
        "position_facts": position_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=PUBLIC / "registry.json")
    parser.add_argument("--dsn", default=os.getenv("DATABASE_URL"))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    facts, trades, missing, population = inventory(args.registry)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "population_policy": "all_registry_statuses_including_archived",
        "strategy_count": len(set(population)),
        "metric_event_count": len(facts),
        "trade_count": len(trades),
        "missing": missing,
        "inventory_hash": hashlib.sha256(
            json.dumps(
                {
                    "metrics": [fact.__dict__ for fact in facts],
                    "trades": [trade.__dict__ for trade in trades],
                    "population": population,
                },
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest(),
        "applied": False,
    }
    if args.apply:
        if not args.dsn:
            raise SystemExit("--dsn or DATABASE_URL is required with --apply")
        import psycopg

        with psycopg.connect(args.dsn) as connection:
            report["inserted"] = _write(connection, facts, trades)
        report["applied"] = True
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
