"""C044: fixed-parameter 2026 representation diagnostic, not a strategy experiment.

The archived holdout is explicitly observed. No claim that inspecting covariates
preserves an unseen test: future design changes require a later evaluation period.
C043 code and evidence stay untouched; their bytes are bound through an externally
verified C043 report. No network, training, current-loader bypass or trading.
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import platform
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnostics.audit_hmm_representation import (  # noqa: E402
    MASK,
    MODEL,
    PORTABLE,
    SEED,
    SOURCE_BINDINGS,
    filtered,
    observations,
)
from src.research.historical_hmm_audit import (  # noqa: E402
    HistoricalArchive,
    _digest,
    _json,
    compare_stored,
    historical_model,
    recover_mask,
    representation_delta,
    safe_path,
    sha256,
)

SUCCESS = "DIAGNOSTIC_REPRODUCED_NOT_SCIENTIFIC_CLOSURE"


def load_reference(path: Path, expected_sha256: str):
    raw = safe_path(path).read_bytes()
    if sha256(raw) != _digest(expected_sha256):
        raise ValueError("selection reference hash mismatch")
    ref = _json(raw)
    if (ref.get("contract") != "THESIS-HISTORICAL-HMM-DIAGNOSTIC-1"
            or ref.get("status") != SUCCESS
            or ref["cohort"]["partition"] != "selection"
            or ref["cohort"]["n_sessions"] != 226
            or ref["baseline_parity"]["passed"] is not True):
        raise ValueError("reference must be the successful C043 selection diagnostic")
    return ref


def bind_reference_code(ref: dict, repo: Path):
    keys = {"runner_sha256": "scripts/diagnostics/audit_hmm_representation.py",
            "helper_sha256": "src/research/historical_hmm_audit.py"}
    for field, key in keys.items():
        if sha256(safe_path(repo / key).read_bytes()) != _digest(ref[field]):
            raise ValueError("C043 source binding mismatch: " + key)


def cohort_specs(blob):
    specs = [s for s in blob["holdout"] if pd.Timestamp(s.date).year == 2026]
    dates = pd.DatetimeIndex([s.date for s in specs])
    if len(dates) != 150 or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("C044 requires all 150 ordered unique archived 2026 sessions")
    return specs


def audit_2026(snapshot, expected_snapshot_sha, reference_path, expected_reference_sha, *, repo=ROOT):
    ref = load_reference(reference_path, expected_reference_sha)
    if ref["snapshot_sha256"] != expected_snapshot_sha:
        raise ValueError("reference/snapshot identity mismatch")
    bind_reference_code(ref, repo)
    archive = HistoricalArchive(snapshot, expected_snapshot_sha)
    archive.bind_current_code(repo, SOURCE_BINDINGS)
    # Local trusted project pickle: hash the SAME bytes before deserializing them.
    blob = pickle.loads(archive.read(PORTABLE))
    model, model_report = historical_model(archive.json(MODEL), blob,
                                           model_sha256=archive.entries[MODEL]["sha256"])
    if model.k != 5:
        raise ValueError("C044 requires the same archived K5, not another model")
    for key in ("dataset_identity", "parameter_sha256", "model_artifact_sha256"):
        if model_report[key] != ref["model"][key]:
            raise ValueError("reference model mismatch: " + key)
    specs = cohort_specs(blob)
    dates = pd.DatetimeIndex([s.date for s in specs])
    contexts = [np.asarray(s.context) for s in specs]
    if any(x.shape != (7,) or x.dtype != np.float32 for x in contexts):
        raise ValueError("archived context must have seven float32 coordinates")
    stored = np.vstack([x[-4:] for x in contexts])
    prices = pd.read_parquet(io.BytesIO(archive.read(SEED)))
    if "symbol" in prices:
        prices = prices[prices["symbol"].astype(str).str.upper().str.replace(
            "/", "", regex=False).eq("USDCOP")].copy()
    times = pd.to_datetime(prices["time"])
    valid, train = recover_mask(archive.json(MASK), times.dt.date.unique())
    if not set(dates.date).issubset(valid):
        raise ValueError("2026 cohort not contained in the archived valid mask")
    prices = prices[times.dt.date <= dates[-1].date()].copy()
    cohort_prices = prices[pd.to_datetime(prices["time"]).dt.date.isin(set(dates.date))]
    if len(cohort_prices) != 150 * 60:
        raise ValueError("C044 cohort must contain 9000 bars")
    ohlc = cohort_prices[["open", "high", "low", "close"]].astype(float)
    original_obs = observations(archive, prices, valid)
    baseline, sources, numeric = filtered(model, original_obs)
    parity = compare_stored(baseline, dates, stored)
    parity["rows"] = [{"date": str(d.date()),
                       "observation_date": None if pd.isna(sources.get(d)) else str(sources[d].date()),
                       "max_abs_error": None if d not in baseline.index or
                       not np.isfinite(baseline.loc[d].to_numpy()[:4]).all() else
                       float(np.max(np.abs(baseline.loc[d].to_numpy()[:4] - x)))}
                      for d, x in zip(dates, stored, strict=True)]
    report = {
        "contract": "THESIS-HISTORICAL-HMM-2026-DIAGNOSTIC-1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scope": "retrospective_diagnostic_not_confirmatory",
        "snapshot_sha256": expected_snapshot_sha,
        "selection_reference_sha256": expected_reference_sha,
        "selection_reference_payload_sha256": sha256(json.dumps(
            ref, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()),
        "reference_code_sha256": {"runner": ref["runner_sha256"], "helper": ref["helper_sha256"]},
        "runtime": {"python": platform.python_version(), **{x: version(x) for x in
                                                            ("numpy", "pandas", "scipy", "matplotlib")}},
        "cohort": {"source_partition": "holdout", "year": 2026, "n_sessions": len(dates),
                   "n_bars": len(cohort_prices), "first": str(dates[0].date()),
                   "last": str(dates[-1].date()),
                   "dates_sha256": sha256("\n".join(str(d.date()) for d in dates).encode()),
                   "flat_ohlc_fraction": float(ohlc.eq(ohlc["close"], axis=0).all(axis=1).mean()),
                   "archived_valid_count": len(valid), "archived_train_valid_count": len(train),
                   "not_compared_cohorts": ["holdout_2024", "holdout_2025", "forward"]},
        "model": model_report, "transform_contract": ref["transform_contract"],
        "baseline_parity": parity, "baseline_numerics": numeric,
        "counterfactual": None, "status": "BASELINE_PARITY_FAILED",
        "holdout_inputs_read": True, "market_returns_derived": True,
        "claim_preserves_unseen_holdout": False, "strategy_returns_evaluated": False,
        "network_used": False, "secrets_read": False, "training_performed": False,
        "current_model_eligible": False,
        "does_not_prove": ["historical_point_in_time_availability", "vendor_bias_causality",
                           "cause_of_PPO_losses", "tradable_alpha", "future_K_or_schema_approval",
                           "independent_replication_across_market_periods"],
        "future_design_rule": "changes_motivated_here_require_a_later_evaluation_period",
    }
    if parity["passed"] and not numeric["fallback_rows"]:
        flattened = prices.copy()
        for col in ("open", "high", "low"):
            flattened[col] = flattened["close"]
        other, other_sources, other_numeric = filtered(model, observations(archive, flattened, valid))
        same = baseline.index.equals(other.index) and sources.equals(other_sources)
        report["history_alignment"] = {
            "passed": same,
            "original_only_dates": [str(x.date()) for x in baseline.index.difference(other.index)],
            "flattened_only_dates": [str(x.date()) for x in other.index.difference(baseline.index)]}
        report["counterfactual_numerics"] = other_numeric
        if same and not other_numeric["fallback_rows"]:
            report["counterfactual"] = representation_delta(baseline, other, dates, baseline_passed=True)
            report["counterfactual"].update({
                "same_fitted_parameters": True,
                "intervention": "replace_each_bar_OHL_by_its_unchanged_close_in_entire_prefix"})
            report["status"] = SUCCESS
        else:
            report["status"] = "COUNTERFACTUAL_HISTORY_OR_NUMERICS_BLOCKED"
    elif parity["passed"]:
        report["status"] = "BASELINE_NUMERICAL_FALLBACK_BLOCKED"
    archive.bind_current_code(repo, SOURCE_BINDINGS)
    bind_reference_code(ref, repo)
    for key in list(archive.used):
        archive.read(key)
    report["inputs_sha256"] = dict(sorted(archive.used.items()))
    report["runner_sha256"] = sha256(Path(__file__).read_bytes())
    return report


def ecdf(values):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
            or (values < 0).any() or (values > 1).any()):
        raise ValueError("TV distribution must be finite and in [0, 1]")
    return np.sort(values), np.arange(1, len(values) + 1) / len(values)


def figure(report, reference, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    rows = report["counterfactual"]["rows"]
    dates = pd.to_datetime([x["date"] for x in rows])
    tv = np.array([x["total_variation"] for x in rows])
    changed = np.array([x["original_state_coordinate"] != x["flattened_state_coordinate"] for x in rows])
    prior = [x["total_variation"] for x in reference["counterfactual"]["rows"]]
    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "C044-fixed-cohort", "font.family": "DejaVu Sans"}):
        fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.7))
        fig.subplots_adjust(left=0.07, right=0.98, top=0.79, bottom=0.30, wspace=0.26)
        fig.suptitle("Sensibilidad del HMM a la representación OHLC", x=0.07, y=0.98,
                     ha="left", fontsize=18, weight="bold")
        fig.text(0.07, 0.90, "Diagnóstico retrospectivo · K=5 fijo · cierres conservados · paridad previa aprobada",
                 fontsize=11, color="#374151")
        zoom = axes[0].inset_axes([0.35, 0.33, 0.60, 0.43])
        for values, label, color, style in ((prior, "Selección 2023 · n=226", "#1D4ED8", "--"),
                                             (tv, "Hold-out histórico 2026 · n=150", "#9A3412", "-")):
            x, y = ecdf(values)
            axes[0].step(np.r_[0, x, 1], np.r_[0, y, 1], where="post", color=color,
                         linestyle=style, linewidth=2, label=label)
            zoom.step(np.r_[0, x, 1], np.r_[0, y, 1], where="post", color=color,
                      linestyle=style, linewidth=1.6)
        zoom.set(xlim=(0, 0.05), ylim=(0.8, 1.005), xticks=[0, 0.025, 0.05],
                 yticks=[0.8, 0.9, 1.0])
        zoom.set_title("Ampliación: TV [0; 0,05] · fracción [0,8; 1]", fontsize=9)
        zoom.tick_params(labelsize=8)
        zoom.grid(axis="y", color="#D1D5DB", linewidth=0.6)
        axes[0].set(title="A · Distribución acumulada", xlabel="Distancia de variación total (TV)",
                    ylabel="Fracción de sesiones", xlim=(-0.01, 1.01), ylim=(0, 1.03))
        axes[0].legend(frameon=False, fontsize=10, loc="lower right")
        axes[1].plot(dates, tv, color="#9A3412", linewidth=1.4, label="TV por sesión")
        axes[1].scatter(dates[changed], tv[changed], color="#111827", marker="s", s=22,
                        zorder=3, label="Cambia estado dominante")
        axes[1].set(title="B · Sesiones de 2026", ylabel="Distancia de variación total (TV)",
                    ylim=(0, 1.03))
        axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        axes[1].legend(frameon=False, fontsize=9, loc="upper left")
        for axis in axes:
            axis.grid(axis="y", color="#D1D5DB", linewidth=0.6)
            axis.set_axisbelow(True)
        flat_old = reference["cohort"]["flat_ohlc_fraction_selection"] * 100
        flat_new = report["cohort"]["flat_ohlc_fraction"] * 100
        fig.text(0.07, 0.07, f"OHLC ya planas: 2023 {flat_old:.2f}% de 13.560 barras; "
                 f"2026 {flat_new:.2f}% de 9.000 barras.\n"
                 "TV = ½ Σ |p original - p aplanado|. No es retorno ni atribución causal al proveedor.",
                 fontsize=10, linespacing=1.6)
        fig.savefig(output / "sensitivity.png", dpi=180, metadata={"Software": "C044"})
        fig.savefig(output / "sensitivity.svg", metadata={"Date": None, "Creator": "C044"})
        plt.close(fig)


def verified_rows(report, *, year: int, expected_count: int):
    """Do not turn an inconsistent 'PASS' label into a thesis figure."""
    parity = report["baseline_parity"]
    if (parity["passed"] is not True or parity["n_expected"] != expected_count
            or parity["n_finite_sessions"] != expected_count
            or parity["max_abs_error"] is None or not 0 <= parity["max_abs_error"] <= 1e-6
            or report["history_alignment"]["passed"] is not True
            or report["model"]["k"] != 5
            or any(report[key]["fallback_rows"] for key in ("baseline_numerics", "counterfactual_numerics"))):
        raise ValueError("figure evidence gates are inconsistent")
    result = report["counterfactual"]
    rows = result["rows"]
    dates = pd.to_datetime([row["date"] for row in rows])
    if (len(rows) != expected_count or not dates.is_unique or not dates.is_monotonic_increasing
            or not (dates.year == year).all()):
        raise ValueError("figure row cohort mismatch")
    arrays = [np.asarray([row[key] for row in rows], dtype=float) for key in ("original", "flattened")]
    for arr in arrays:
        if (arr.shape != (expected_count, 5) or not np.isfinite(arr).all()
                or (arr < 0).any() or (arr > 1).any()
                or not np.allclose(arr.sum(axis=1), 1, atol=1e-12, rtol=0)):
            raise ValueError("figure posterior is not a five-state probability simplex")
    checked = representation_delta(pd.DataFrame(arrays[0], index=dates),
                                   pd.DataFrame(arrays[1], index=dates), dates, baseline_passed=True)
    for key in ("n_sessions", "changed_argmax", "total_variation_mean", "total_variation_median",
                "total_variation_p95", "total_variation_max"):
        if not np.isclose(result[key], checked[key], atol=1e-12, rtol=0):
            raise ValueError("figure summary mismatch: " + key)
    for row, expected in zip(rows, checked["rows"], strict=True):
        if (row["original_state_coordinate"] != expected["original_state_coordinate"]
                or row["flattened_state_coordinate"] != expected["flattened_state_coordinate"]
                or not np.isclose(row["total_variation"], expected["total_variation"], atol=1e-12, rtol=0)):
            raise ValueError("figure row probability/state/TV mismatch")
    return rows


def write_bundle(report, reference, output):
    output = safe_path(output)
    if not output.is_relative_to(ROOT):
        raise ValueError("output must stay inside the workspace")
    successful = report["status"] == SUCCESS and report["baseline_parity"]["passed"] is True
    if successful:
        if (sha256(json.dumps(reference, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
                != report["selection_reference_payload_sha256"]):
            raise ValueError("figure reference payload changed after verification")
        verified_rows(report, year=2026, expected_count=150)
        verified_rows(reference, year=2023, expected_count=226)
    output.mkdir(parents=True, exist_ok=False)
    (output / "diagnostic.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                            encoding="utf-8")
    keys = ["diagnostic.json"]
    if successful:
        rows = report["counterfactual"]["rows"]
        pd.DataFrame([{k: row[k] for k in ("date", "total_variation", "original_state_coordinate",
                                          "flattened_state_coordinate")} for row in rows]).to_csv(
                                              output / "posterior_rows.csv", index=False)
        figure(report, reference, output)
        text = ("Figura C044, datos retrospectivos reales archivados.\n"
                "Panel A: distribuciones acumuladas completas de TV; no se emparejan fechas entre años.\n"
                "Recuadro: amplía TV [0, 0.05], fracción [0.8, 1.005]; mismas filas, sin filtrar ni renormalizar.\n"
                "Panel B: TV por sesión2026; cuadrados señalan cambio de estado de mayor probabilidad.\n"
                "Cada TV compara original y OHL=C sobre la misma historia y los mismos parámetros.\n"
                "Paridad: cuatro coordenadas float32 archivadas, no posterior K5 previamente almacenado.\n"
                "La comparación entre años no identifica causalidad del proveedor ni pérdida económica.\n"
                f"Sesiones2026: {len(rows)}; cambios: {report['counterfactual']['changed_argmax']}.\n")
        (output / "sensitivity.txt").write_text(text, encoding="utf-8")
        keys.extend(["posterior_rows.csv", "sensitivity.png", "sensitivity.svg", "sensitivity.txt"])
    manifest = {"contract": "THESIS-HMM-2026-FIGURE-BUNDLE-1",
                "selection_reference_sha256": report["selection_reference_sha256"],
                "runner_sha256": report["runner_sha256"],
                "files": {key: sha256((output / key).read_bytes()) for key in keys}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--expected-snapshot-sha", required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--expected-selection-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if safe_path(args.output).exists():
        parser.error("output directory must be NEW")
    report = audit_2026(args.snapshot, args.expected_snapshot_sha,
                        args.selection_report, args.expected_selection_sha)
    reference = load_reference(args.selection_report, args.expected_selection_sha)
    write_bundle(report, reference, args.output)
    print(json.dumps({"status": report["status"], "cohort": report["cohort"],
                      "parity_max_error": report["baseline_parity"]["max_abs_error"],
                      "counterfactual": {k: v for k, v in (report["counterfactual"] or {}).items()
                                         if k != "rows"}}))
    return 0 if report["status"] == SUCCESS else 2


if __name__ == "__main__":
    raise SystemExit(main())
