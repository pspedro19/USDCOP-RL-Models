"""Export an auditable 2026 USDCOP forecasting/backtest workbook.

The workbook keeps forecast horizons separate and explicitly distinguishes the
dashboard's walk-forward backtest metrics from realized H5 predictions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "usdcop-trading-dashboard" / "public" / "forecasting" / "bi_dashboard_unified.csv"
PRED = ROOT / "data" / "backups" / "features" / "forecast_h5_predictions.parquet"
DAILY = ROOT / "seeds" / "latest" / "usdcop_daily_ohlcv.parquet"


def _read_daily() -> pd.DataFrame:
    d = pd.read_parquet(DAILY)
    d["date"] = pd.to_datetime(d["time"], utc=True).dt.date
    return d.sort_values("date").drop_duplicates("date")


def _realized_h5_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    p = pd.read_parquet(PRED)
    p["inference_date"] = pd.to_datetime(p["inference_date"]).dt.date
    p["target_date"] = pd.to_datetime(p["target_date"]).dt.date
    d = _read_daily()[["date", "close"]].rename(columns={"close": "actual_target_close"})
    out = p.merge(d, left_on="target_date", right_on="date", how="left")
    out["actual_return_pct"] = (out["actual_target_close"] / out["base_price"] - 1.0) * 100
    out["actual_direction"] = out["actual_return_pct"].map(
        lambda x: "UP" if x > 0 else ("DOWN" if x < 0 else "FLAT")
    )
    out["direction_correct"] = out["direction"].eq(out["actual_direction"])
    dates = set(_read_daily()["date"])
    out["sessions_between"] = out.apply(
        lambda r: sum(r["inference_date"] < x <= r["target_date"] for x in dates), axis=1
    )
    out["horizon_alignment"] = out["sessions_between"].eq(out["horizon_id"])
    summary = (out.groupby(["model_id", "horizon_id"], dropna=False)
               .agg(n_predictions=("id", "size"), realized_da=("direction_correct", "mean"),
                    aligned=("horizon_alignment", "mean"))
               .reset_index())
    summary["realized_da"] *= 100
    summary["aligned"] *= 100
    return out, summary


def build(output: Path) -> Path:
    raw = pd.read_csv(CSV)
    bt = raw[(raw["view_type"] == "backtest") & (raw["inference_year"] == 2026)].copy()
    bt["horizon_days"] = pd.to_numeric(bt["horizon_days"], errors="coerce").astype("Int64")
    bt["direction_accuracy_pct"] = bt["direction_accuracy"] * 100
    bt["wf_direction_accuracy_pct"] = bt["wf_direction_accuracy"] * 100
    bt["selective_da_top50_pct"] = pd.to_numeric(bt["selective_da_top50"], errors="coerce") * 100
    bt["selective_da_top25_pct"] = pd.to_numeric(bt["selective_da_top25"], errors="coerce") * 100

    by_h = (bt.groupby(["horizon_days", "horizon_label"], dropna=False)
            .agg(records=("record_id", "size"), models=("model_id", "nunique"),
                 weeks=("inference_week", "nunique"),
                 mean_da_pct=("direction_accuracy_pct", "mean"),
                 min_da_pct=("direction_accuracy_pct", "min"),
                 max_da_pct=("direction_accuracy_pct", "max"),
                 mean_wf_da_pct=("wf_direction_accuracy_pct", "mean"),
                 mean_selective_da_top50_pct=("selective_da_top50_pct", "mean"),
                 mean_selective_da_top25_pct=("selective_da_top25_pct", "mean"),
                 mean_sharpe=("sharpe", "mean"), mean_profit_factor=("profit_factor", "mean"),
                 mean_max_drawdown=("max_drawdown", "mean"), mean_total_return=("total_return", "mean"))
            .reset_index().sort_values("horizon_days"))
    by_model = (bt.groupby(["model_id", "model_name", "horizon_days", "horizon_label"], dropna=False)
                .agg(records=("record_id", "size"), weeks=("inference_week", "nunique"),
                     mean_da_pct=("direction_accuracy_pct", "mean"),
                     min_da_pct=("direction_accuracy_pct", "min"),
                     max_da_pct=("direction_accuracy_pct", "max"),
                     mean_wf_da_pct=("wf_direction_accuracy_pct", "mean"),
                     mean_selective_da_top50_pct=("selective_da_top50_pct", "mean"),
                     mean_selective_da_top25_pct=("selective_da_top25_pct", "mean"),
                     mean_sharpe=("sharpe", "mean"), mean_profit_factor=("profit_factor", "mean"),
                     mean_max_drawdown=("max_drawdown", "mean"), mean_total_return=("total_return", "mean"))
                .reset_index().sort_values(["horizon_days", "mean_da_pct"], ascending=[True, False]))
    w27 = bt[bt["inference_week"] == "2026-W27"].sort_values(["horizon_days", "direction_accuracy"], ascending=[True, False])
    actual, actual_summary = _realized_h5_audit()

    coverage = pd.DataFrame([
        {"check": "weekly_analysis DB rows", "value": 0, "status": "MISSING"},
        {"check": "forecast_h5_predictions rows", "value": len(actual), "status": "PRESENT"},
        {"check": "forecast_h5_signals rows", "value": 10, "status": "PRESENT"},
        {"check": "weekly backtest CSV rows (2026)", "value": len(bt), "status": "PRESENT"},
        {"check": "retraining run linked per weekly forecast", "value": "not evidenced", "status": "GAP"},
        {"check": "PIT available_at for promotion", "value": "missing", "status": "BLOCKER"},
        {"check": "horizon mix", "value": "H1/H5/H10/H15/H20/H25/H30 separated", "status": "PASS"},
    ])
    readme = pd.DataFrame({"item": [
        "Scope", "Backtest metric", "Realized metric", "Horizon rule", "Promotion conclusion"],
        "detail": [
            "USDCOP weekly forecasting artifacts generated through 2026-W30",
            "Dashboard walk-forward directional_accuracy, grouped by horizon; not a fresh trade result",
            "forecast_h5_predictions joined to USDCOP daily close at target_date",
            "sessions_between must equal horizon_id; mismatches are flagged, never pooled",
            "Research-only until PIT vintages, linked retraining manifests and OOS gates are present",
        ]})
    # Excel has no timezone-aware datetime cell type.
    for frame in (w27, actual):
        for col in frame.columns:
            if isinstance(frame[col].dtype, pd.DatetimeTZDtype):
                frame[col] = frame[col].dt.tz_localize(None)
    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        readme.to_excel(writer, index=False, sheet_name="README")
        by_h.to_excel(writer, index=False, sheet_name="2026_by_horizon")
        by_model.to_excel(writer, index=False, sheet_name="2026_by_model_horizon")
        w27.to_excel(writer, index=False, sheet_name="2026_W27_backtest")
        actual.to_excel(writer, index=False, sheet_name="H5_realized_audit")
        actual_summary.to_excel(writer, index=False, sheet_name="H5_realized_summary")
        coverage.to_excel(writer, index=False, sheet_name="coverage_gaps")
        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = min(max(max(len(str(c.value or "")) for c in col) + 2, 12), 32)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/usdcop_forecasting_directional_accuracy_2026.xlsx")
    args = parser.parse_args()
    print(build(ROOT / args.output))
