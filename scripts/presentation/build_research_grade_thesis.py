"""Reproduce the retrospective thesis from a verified, preserved local snapshot.

No training, API calls, source refetching, policy selection, or hold-out evaluation.
Outputs are new and exclusive. Missing historical provenance is not fabricated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import pickle
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.reporting_v2 import (  # noqa: E402
    bootstrap_indices,
    capital_path,
    hierarchical_sharpe,
    holm,
    paired_difference,
    path_counts,
    summarize,
)
from src.research.retrospective_regime import recover_regime_categories  # noqa: E402

SEEDS = (42, 123, 456, 789, 1337)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Snapshot:
    def __init__(self, manifest):
        self.path = Path(manifest).resolve()
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if raw.get("contract") != "THESIS-EVIDENCE-SNAPSHOT-1":
            raise ValueError("not a preserved thesis snapshot")
        self.files = {r["path"]: r for r in raw["files"]}
        if len(self.files) != len(raw["files"]):
            raise ValueError("duplicate snapshot path")
        self.used = {}

    def read(self, key):
        r = self.files[key]
        if r["object"] != "objects/" + r["sha256"] or len(r["sha256"]) != 64:
            raise ValueError("snapshot object is not content-addressed")
        p = (self.path.parent / r["object"]).resolve()
        p.relative_to(self.path.parent)
        data = p.read_bytes()
        if hashlib.sha256(data).hexdigest() != r["sha256"]:
            raise ValueError("snapshot content hash mismatch: " + key)
        self.used[key] = r["sha256"]
        return data

    def json(self, key):
        return json.loads(self.read(key))


def independent_score(spec, weights, commission, slippage):
    """Independent scalar replay of the declared cost formula, including final close."""
    c, w = np.asarray(spec.close, float), np.asarray(weights, float)
    if c.shape != (60,) or w.shape != (59,) or (c <= 0).any():
        raise ValueError("invalid session shape or prices")
    if not np.isfinite([*c, *w]).all() or (abs(w) > 1).any():
        raise ValueError("nonfinite price or invalid exposure")
    lr = np.diff(np.log(c), prepend=np.log(c[0]))
    vol = np.array([np.std(lr[max(0, i - 11) : i + 1], ddof=1) if i else 0.0 for i in range(60)])
    dw = np.diff(np.r_[0.0, w, 0.0])
    cost = float(
        np.sum(abs(dw) * ((float(spec.spread_pips) / 2 + commission) / c + slippage * vol))
    )
    gross = float(np.sum(w * (c[1:] / c[:-1] - 1)))
    return gross, cost


def write_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)


def build(manifest, output, *, replications=10_000):
    import pandas as pd
    import yaml

    snapshot, out = Snapshot(manifest), Path(output).resolve()
    out.relative_to(ROOT)
    if out.exists():
        raise FileExistsError("refusing to overwrite a report; use a new versioned output")
    # Trusted local objects captured by freeze_thesis_evidence; only selection is evaluated.
    portable = pickle.loads(snapshot.read("data/thesis/research_data_portable_v2.pkl"))
    specs = portable["selection"]
    regime_ids, regime_evidence = recover_regime_categories(
        [s.context[-4:] for s in specs], declared_k=portable.get("regime_meta", {}).get("k")
    )
    regime_evidence["portable_sha256"] = snapshot.used["data/thesis/research_data_portable_v2.pkl"]
    regime_evidence["portable_declared_hmm_sha256"] = portable.get("regime_artifact_sha256")
    hmm_paths = [
        path
        for path, entry in snapshot.files.items()
        if entry["sha256"] == regime_evidence["portable_declared_hmm_sha256"]
    ]
    if not hmm_paths:
        raise ValueError("historical HMM declared by portable is absent from the snapshot")
    archived_hmm = snapshot.json(sorted(hmm_paths)[0])
    if (
        archived_hmm["k"] != regime_evidence["declared_k"]
        or not portable.get("identity")
        or archived_hmm.get("dataset_identity") != portable["identity"]
    ):
        raise ValueError("archived portable and HMM metadata do not identify the same dataset")
    regime_evidence["archived_hmm_metadata_link_verified"] = True
    regime_evidence["dataset_identity"] = portable["identity"]
    dates = [s.date.isoformat() for s in specs]
    if len(set(dates)) != len(dates) or dates != sorted(dates):
        raise ValueError("selection dates not unique and sorted")
    regime_evidence["session_dates"] = dates
    costs = yaml.safe_load(snapshot.read("config/research/cost_contract.yaml"))

    def score(s, w):
        return independent_score(s, w, costs["commission_per_side"], costs["slippage_coef"])

    weight_blob = snapshot.json("outputs/thesis-repair/ppo_v2_weights_selection.json")
    if weight_blob["block"] != "selection" or set(weight_blob["median"]) != set(dates):
        raise ValueError("PPO weights do not cover the frozen selection")
    median = np.asarray([weight_blob["median"][d] for d in dates])
    arms, metrics, series, tests, matrices, reproduction = {}, {}, {}, {}, {}, {}

    def add(name, g, c, weights=None, min_trades=None, note=""):
        g, c = np.asarray(g), np.asarray(c)
        if g.shape != (len(dates),) or c.shape != g.shape:
            raise ValueError("arm not aligned to full selection: " + name)
        perday = [path_counts(w) for w in weights] if weights is not None else None
        daily_count = (
            np.array([x["round_trips"] for x in perday]) if perday else (c > 0).astype(int)
        )
        count = int(daily_count.sum()) if min_trades is None else min_trades
        arms[name] = {
            "gross": g,
            "cost": c,
            "net": g - c,
            "weights": weights,
            "daily_count": daily_count,
        }
        metrics[name] = {
            **summarize(
                g,
                c,
                min_round_trips=count,
                count_is_lower_bound=weights is None and min_trades is None,
            ),
            "note": note,
        }
        if perday:
            metrics[name].update(
                mean_changes_decisions_only=float(
                    np.mean([x["changes_decisions_only"] for x in perday])
                ),
                turnover_sum=float(sum(x["turnover"] for x in perday)),
            )
        series[name] = [
            {
                "date": d,
                "gross_return": float(g[i]),
                "cost_return": float(c[i]),
                "net_return": float(g[i] - c[i]),
                "round_trips_or_lower_bound": int(daily_count[i]),
            }
            for i, d in enumerate(dates)
        ]

    for name, level in [("always_flat", 0), ("B1_session", 1), ("NULL_A_short", -1)]:
        w = np.full((len(specs), 59), level, dtype=float)
        z = np.array([score(s, wi) for s, wi in zip(specs, w, strict=False)])
        add(name, z[:, 0], z[:, 1], w)
    close = np.array([s.close[-1] for s in specs])
    add(
        "B1_passive_uncosted",
        np.r_[0.0, close[1:] / close[:-1] - 1],
        np.zeros(len(dates)),
        min_trades=1,
        note="Uncosted close-to-close reference, not an intraday policy; ratios suppressed.",
    )
    for config in ["ppo_regime", "ppo_backbone"]:
        gross, fee = [], []
        for seed in SEEDS:
            blob = snapshot.json(
                f"outputs/thesis-repair/ppo_v2_recipe_flat/{config}_seed{seed}.json"
            )
            b = blob["selection"]
            if b["dates"] != dates or blob.get("recipe_probe") != "flat_init_no_turn":
                raise ValueError("wrong PPO dates or recipe")
            g, c, r = map(
                np.asarray, (b["daily_gross_returns"], b["daily_costs"], b["daily_returns"])
            )
            if not np.allclose(g - c, r, rtol=0, atol=1e-12):
                raise ValueError("PPO daily accounting mismatch")
            w = None
            if config == "ppo_regime":
                w = np.array([weight_blob["per_seed"][str(seed)][d] for d in dates])
                z = np.array([score(s, wi) for s, wi in zip(specs, w, strict=False)])
                delta = float(np.max(abs(z[:, 0] - z[:, 1] - r)))
                reproduction[f"{config}_{seed}"] = delta
                if delta > 1e-12:
                    raise ValueError("independent PPO replay differs")
            add(
                f"{config}_seed{seed}",
                g,
                c,
                w,
                note="Historical model provenance incomplete; saved return series preserved.",
            )
            gross.append(g)
            fee.append(c)
        matrices[config] = np.asarray(gross) - np.asarray(fee)
        add(
            config + "_mean5",
            np.mean(gross, axis=0),
            np.mean(fee, axis=0),
            note="Equal-weight daily-return portfolio; NOT a median-exposure policy.",
        )
    z = np.array([score(s, w) for s, w in zip(specs, median, strict=False)])
    add(
        "ppo_median_weights",
        z[:, 0],
        z[:, 1],
        median,
        note="Exact PPO component of both sign-veto hybrids.",
    )
    ledger_quality, veto = {}, {}
    for provider in ["deepseek", "azure"]:
        rows = [
            json.loads(x)
            for x in snapshot.read(
                f"data/thesis/llm/decisions_{provider}_selection_diagnostic.jsonl"
            )
            .decode("utf-8")
            .splitlines()
            if x
        ]
        by = {(r["session_date"], r["bar"]): r for r in rows}
        ids = [r["decision_id"] for r in rows]
        expected = {(d, i) for d in dates for i in range(59)}
        if len(by) != len(rows) or set(by) != expected or len(set(ids)) != len(ids):
            raise ValueError("duplicate, missing or unexpected LLM decisions")
        if any(r.get("valid_json") is not True or r.get("unavailable") for r in rows):
            raise ValueError("legacy LLM cohort has invalid or unavailable responses")
        w = np.array([[float(by[d, i]["weight"]) for i in range(59)] for d in dates])
        for j, d in enumerate(dates):
            for i in range(59):
                if abs(float(by[d, i]["previous_weight"]) - (w[j, i - 1] if i else 0)) > 1e-12:
                    raise ValueError("inconsistent recursive LLM state")
        z = np.array([score(s, wi) for s, wi in zip(specs, w, strict=False)])
        add(
            provider,
            z[:, 0],
            z[:, 1],
            w,
            note="Historical numerical prompt; no news, unclear scaling; not a general LLM evaluation.",
        )
        stored = snapshot.json(f"outputs/thesis-repair/settlement_{provider}_selection_v2.json")
        sr = {r["session_date"]: r["daily_return"] for r in stored["sessions"]}
        reproduction[provider] = max(abs(z[i, 0] - z[i, 1] - sr[d]) for i, d in enumerate(dates))
        if reproduction[provider] > 1e-12:
            raise ValueError("independent LLM replay differs")
        keep = (np.sign(w) == np.sign(median)) & (w != 0) & (median != 0)
        wh = np.where(keep, median, 0.0)
        zh = np.array([score(s, wi) for s, wi in zip(specs, wh, strict=False)])
        add("hybrid_" + provider, zh[:, 0], zh[:, 1], wh)
        market_returns = np.array([s.close[1:] / s.close[:-1] - 1 for s in specs])
        endorsed = np.sum(median * market_returns * keep, axis=1)
        rejected = np.sum(median * market_returns * (~keep), axis=1)
        idx = bootstrap_indices(len(specs), replications)
        veto[provider] = {
            "endorsed_gross_sum_pct": float(endorsed.sum() * 100),
            "rejected_gross_sum_pct": float(rejected.sum() * 100),
            "endorsed_ci95_sum_pct": list(
                map(float, np.percentile(endorsed[idx].sum(axis=1) * 100, [2.5, 97.5]))
            ),
            "endorsed_bars": int(keep.sum()),
            "endorsed_long_bars": int(np.sum(keep & (median > 0))),
            "endorsed_short_bars": int(np.sum(keep & (median < 0))),
            "interpretation": "Retrospective association, not independent replication or a causal signal claim.",
        }
        ledger_quality[provider] = {
            "unique_decisions": len(ids),
            "complete_sessions": len(dates),
            "invalid": 0,
            "unavailable": 0,
            "strict_original_provenance": False,
            "missing_original_fields": [
                "dataset_sha256",
                "cutoff_utc",
                "served_model",
                "response_id",
            ],
        }
    sup = snapshot.json("outputs/thesis-repair/supervised_v2_selection.json")
    if sup["dates"] != dates:
        raise ValueError("supervised dates mismatch")
    if not np.allclose(
        np.asarray(sup["daily_gross_returns"]) - sup["daily_costs"],
        sup["daily_returns"],
        atol=1e-12,
        rtol=0,
    ):
        raise ValueError("supervised net does not reconcile with gross minus costs")
    add(
        "LogReg",
        sup["daily_gross_returns"],
        sup["daily_costs"],
        note="Historical diagnostic supervised variant; trial reconciliation pending.",
    )
    headline = [
        "always_flat",
        "NULL_A_short",
        "B1_passive_uncosted",
        "B1_session",
        "ppo_backbone_mean5",
        "ppo_regime_mean5",
        "ppo_median_weights",
        "LogReg",
        "hybrid_deepseek",
        "hybrid_azure",
        "deepseek",
        "azure",
    ]
    for name in headline:
        if name in ["always_flat", "B1_passive_uncosted"]:
            continue
        tests[name + "_vs_flat"] = paired_difference(
            arms[name]["net"],
            arms["always_flat"]["net"],
            min_trades_a=metrics[name]["round_trips_minimum"],
            min_trades_b=0,
            flat_reference=True,
            replications=replications,
        )
    adjusted = holm(
        {k: v["p_centered_stationary"] for k, v in tests.items() if "p_centered_stationary" in v}
    )
    for k, value in adjusted.items():
        tests[k]["p_holm_retrospective_family"] = value
    h2 = hierarchical_sharpe(
        matrices["ppo_regime"], matrices["ppo_backbone"], replications=replications
    )
    idx = bootstrap_indices(len(dates), replications)
    gross_ci = list(
        map(
            float,
            np.percentile(arms["ppo_median_weights"]["gross"][idx].sum(axis=1) * 100, [2.5, 97.5]),
        )
    )
    ds_mask = arms["hybrid_deepseek"]["weights"] != 0
    az_mask = arms["hybrid_azure"]["weights"] != 0
    veto["joint"] = {
        "intersection_bars": int(np.sum(ds_mask & az_mask)),
        "union_bars": int(np.sum(ds_mask | az_mask)),
        "weight_correlation": float(
            np.corrcoef(arms["deepseek"]["weights"].ravel(), arms["azure"]["weights"].ravel())[0, 1]
        ),
    }
    report = {
        "scope": "retrospective_diagnostic",
        "confirmatory": False,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dates": [dates[0], dates[-1]],
        "n_sessions": len(dates),
        "cost_contract": costs,
        "headline_order": headline,
        "metrics": metrics,
        "paired_primary_mean_return_tests": tests,
        "h2_hierarchical": h2,
        "ppo_median_gross_sum_ci95_pct": gross_ci,
        "veto": veto,
        "ledger_quality": ledger_quality,
        "independent_replay_max_errors": reproduction,
        "dsr": {
            "status": "PENDING_TRIAL_RECONCILIATION",
            "legacy_registry_count": 115,
            "note": "Do not certify the inherited count as updated.",
        },
        "pbo": {
            "status": "OMITTED",
            "reason": "seed matrix is not the historical selection process",
        },
        "unresolved": [
            "macro_reference_certification",
            "historical_publication_and_vintages",
            "current_sanity_evidence",
            "original_model_request_provenance",
            "trial_reconciliation",
            "real_bid_ask_and_fills",
            "future_replication",
        ],
    }
    out.mkdir(parents=True)
    write_json(out / "regime_provenance.json", regime_evidence)
    write_json(out / "results.json", report)
    write_json(out / "daily_series.json", series)
    with (out / "metrics.csv").open("x", encoding="utf-8", newline="") as f:
        keys = [
            "arm",
            "return_compounded_pct",
            "sharpe",
            "max_drawdown_pct",
            "gross_sum_pct",
            "cost_sum_pct",
            "gross_compounded_pct",
            "cost_account_units",
            "round_trips_minimum",
        ]
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        for name in headline:
            writer.writerow({"arm": name, **{k: metrics[name][k] for k in keys[1:]}})
    seed_frame = pd.read_parquet(io.BytesIO(snapshot.read("seeds/latest/usdcop_m5_ohlcv.parquet")))
    quality = {}
    if {"open", "high", "low", "close"} <= set(seed_frame):
        flat = seed_frame[["open", "high", "low", "close"]].nunique(axis=1) == 1
        date_col = next((k for k in ["time", "timestamp", "datetime"] if k in seed_frame), None)
        times = pd.to_datetime(seed_frame[date_col]) if date_col else pd.Series(seed_frame.index)
        years = times.dt.year.to_numpy()
        quality["flat_ohlc_fraction_by_year"] = {
            str(y): float(flat.to_numpy()[years == y].mean()) for y in np.unique(years)
        }
    quality["scope"] = "Structural description, not quote executability or source certification"
    write_json(out / "data_quality.json", quality)
    make_figures(out, dates, arms, metrics, matrices, regime_ids, regime_evidence, quality)
    text = [
        "# Resultados retrospectivos corregidos",
        "",
        f"Selección {dates[0]} a {dates[-1]}: {len(dates)} sesiones. Costos supuestos, no fills observados.",
        "",
        "| Brazo | Neto compuesto % | Sharpe | MaxDD % |",
        "|---|---:|---:|---:|",
    ]
    for name in headline:
        m = metrics[name]
        sr = "N/A" if m["sharpe"] is None else f"{m['sharpe']:.3f}"
        text.append(
            f"| {name} | {m['return_compounded_pct']:.2f} | {sr} | {m['max_drawdown_pct']:.2f} |"
        )
    text += [
        "",
        "## Conclusiones permitidas",
        "",
        "Las políticas evaluadas perdieron neto bajo el contrato supuesto. Las liquidaciones se reproducen; esto no demuestra imposibilidad de rentabilidad.",
        "El acuerdo de los LLM empeoró la política PPO de pesos medianos. Es asociación retrospectiva, condicionada por dirección y exposición; los proveedores no son replicaciones independientes.",
        f"El IC jerárquico de ΔSharpe régimen/backbone es {h2['portfolio']['ci95']}; incorpora semillas y sesiones y no establece mejora.",
        f"El bruto sumado de PPO mediana tiene IC95 {gross_ci}; bruto positivo no demuestra alfa.",
        "Los prompts históricos no tenían noticias ni explicaban las escalas, posición y costos. No se generaliza a LLM financieros ni NLP.",
        f"La figura 05 usa K={regime_evidence['declared_k']} y método {regime_evidence['method']}. Las categorías y sus fechas están en regime_provenance.json; la recuperación residual, cuando aplica, es sólo descriptiva. No cambia políticas ni PnL ni prueba pérdida de información en PPO.",
        "",
        "## Pendientes que no certifica este informe",
        "",
        *["- " + x for x in report["unresolved"]],
        "",
        "Cada figura y tabla deriva de daily_series.json, results.json y del snapshot. Manifest.json registra los hashes. No se evaluó hold-out ni se llamó a proveedores.",
    ]
    (out / "results.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    artifacts = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    evidence = {
        "contract": "THESIS-RETROSPECTIVE-BUNDLE-1",
        "scope": "retrospective_diagnostic",
        "snapshot_sha256": sha(snapshot.path),
        "snapshot_manifest": str(snapshot.path),
        "inputs_sha256": snapshot.used,
        "generator_sha256": sha(__file__),
        "reporting_code_sha256": sha(ROOT / "src/research/reporting_v2.py"),
        "regime_reporting_code_sha256": sha(ROOT / "src/research/retrospective_regime.py"),
        "artifacts_sha256": artifacts,
        "replications": replications,
        "retrospective_results_reproduced": True,
        "scientific_closure_ready": False,
        "confirmatory_evidence_ready": False,
    }
    write_json(out / "manifest.json", evidence)
    return {
        "output": str(out),
        "n_sessions": len(dates),
        "manifest_sha256": sha(out / "manifest.json"),
        "figures": len(list(out.glob("*.png"))),
        "scientific_closure_ready": False,
    }


def make_figures(out, dates, arms, metrics, matrices, regimes, regime_evidence, quality):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    plt.rcParams.update(
        {
            "font.size": 10,
            "figure.dpi": 130,
            "savefig.dpi": 200,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    ts = pd.to_datetime(dates) + pd.Timedelta(hours=12, minutes=55)
    timeline = pd.DatetimeIndex([pd.Timestamp(dates[0]) + pd.Timedelta(hours=8), *ts])
    selected = [
        "always_flat",
        "ppo_median_weights",
        "deepseek",
        "azure",
        "hybrid_deepseek",
        "hybrid_azure",
    ]
    colors = ["#555555", "#0072B2", "#D55E00", "#CC79A7", "#009E73", "#806300"]
    styles = ["--", "-", ":", "-.", "--", "-"]
    footer = f"Retrospectivo · {dates[0]}–{dates[-1]} · n={len(dates)} · costos supuestos · manifiesto y datos adjuntos"  # noqa: RUF001

    def save(fig, name, *, bottom=0.05):
        fig.text(0.01, 0.01, footer, fontsize=8)
        fig.tight_layout(rect=(0, bottom, 1, 1))
        fig.savefig(out / (name + ".png"))
        fig.savefig(out / (name + ".svg"))
        plt.close(fig)

    for name, title, isdd in [
        ("01_capital", "Capital neto — inicio 100.000 unidades de cuenta", False),
        ("02_drawdown", "Drawdown incluyendo el capital inicial", True),
    ]:
        fig, ax = plt.subplots(figsize=(11, 5))
        for arm, color, style in zip(selected, colors, styles, strict=False):
            eq = capital_path(arms[arm]["net"])
            y = (eq / np.maximum.accumulate(eq) - 1) * 100 if isdd else eq
            ax.plot(timeline, y, label=arm, color=color, linestyle=style, linewidth=1.6)
        ax.set(
            title=title, ylabel="Drawdown %" if isdd else "Unidades de cuenta", xlabel="Sesión COT"
        )
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.2)
        save(fig, name)
    fig, ax = plt.subplots(figsize=(11, 5))
    rolling_rows = []
    for arm, color, style in zip(selected[1:], colors[1:], styles[1:], strict=False):
        r = pd.Series(arms[arm]["net"])
        sd = r.rolling(60).std()
        sufficient = pd.Series(arms[arm]["daily_count"]).rolling(60).sum() >= 20
        sr = (r.rolling(60).mean() / sd * np.sqrt(221)).where(sufficient & (sd > 0))
        ax.plot(ts, sr, label=arm, color=color, linestyle=style)
        rolling_rows += [
            {"date": d, "arm": arm, "sharpe_60": None if pd.isna(v) else float(v)}
            for d, v in zip(dates, sr, strict=False)
        ]
    write_json(out / "rolling_sharpe_data.json", rolling_rows)
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.set(title="Sharpe móvil: 60 sesiones, descriptivo", ylabel="Sharpe anualizado √221")
    ax.legend(fontsize=8, ncol=2)
    save(fig, "03_sharpe_movil")
    fig, ax = plt.subplots(figsize=(9, 5))
    stress = {}
    for arm, color, style in zip(selected[1:], colors[1:], styles[1:], strict=False):
        values = [
            float((capital_path(arms[arm]["gross"] - k * arms[arm]["cost"])[-1] / 100000 - 1) * 100)
            for k in [0, 1, 2, 3]
        ]
        stress[arm] = dict(zip(["zero_cost", "x1", "x2", "x3"], values, strict=False))
        ax.plot([0, 1, 2, 3], values, marker="o", color=color, linestyle=style, label=arm)
    ax.set(
        title="Stress sobre exposiciones congeladas",
        xlabel="Multiplicador del costo supuesto (0 = referencia bruta)",
        ylabel="Retorno compuesto %",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    save(fig, "04_costos")
    write_json(out / "cost_stress.json", stress)
    k_states = regime_evidence["declared_k"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Acciones por régimen: clasificación retrospectiva corregida", fontsize=12)
    action_rows = []
    for ax, arm in zip(axes, ["hybrid_deepseek", "hybrid_azure"], strict=False):
        bottoms = np.zeros(k_states)
        for sign, color, hatch, label in [
            (-1, "#D55E00", "//", "SHORT"),
            (0, "#999999", "..", "FLAT"),
            (1, "#0072B2", "", "LONG"),
        ]:
            counts = []
            for k in range(k_states):
                w = arms[arm]["weights"][regimes == k]
                pct = float(np.mean(np.sign(w) == sign) * 100) if w.size else 0.0
                counts.append(pct)
                action_rows.append(
                    {
                        "arm": arm,
                        "regime_id": k,
                        "action": label,
                        "count": int(np.sum(np.sign(w) == sign)),
                        "total": int(w.size),
                        "pct": pct,
                    }
                )
            ax.bar(range(k_states), counts, bottom=bottoms, color=color, hatch=hatch, label=label)
            for k, pct in enumerate(counts):
                if pct >= 8:
                    ax.text(
                        k,
                        bottoms[k] + pct / 2,
                        f"{pct:.1f}%",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=9,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1},
                    )
            bottoms += counts
        ax.set(
            title=arm,
            xticks=range(k_states),
            xticklabels=[
                f"R{k}\nn={regime_evidence['session_counts'][k]}" for k in range(k_states)
            ],
            xlabel="ID HMM; n = sesiones, porcentajes = decisiones",
            ylabel="% decisiones",
            ylim=(0, 116),
        )
        ax.legend(fontsize=8, loc="upper center", ncol=3)
    coordinate_note = (
        "p4 = 1 - suma(p0..3); sin renormalizar."
        if regime_evidence["declared_k"] == 5
        else "Posteriores completos almacenados; sin reconstrucción residual."
    )
    fig.text(
        0.02,
        0.065,
        coordinate_note + " Sin refit ni cambios de operaciones/PnL. Ver regime_provenance.json.",
        fontsize=9,
    )
    save(fig, "05_acciones_regimen", bottom=0.13)
    write_json(out / "actions_regime.json", action_rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, config in enumerate(["ppo_regime", "ppo_backbone"]):
        vals = [metrics[config + "_seed" + str(s)]["sharpe"] for s in SEEDS]
        x = i + np.linspace(-0.12, 0.12, 5)
        ax.scatter(x, vals, color=colors[i + 1], s=40)
        for xi, value, seed in zip(x, vals, SEEDS, strict=False):
            if value is not None:
                ax.annotate(
                    str(seed), (xi, value), xytext=(3, 3), textcoords="offset points", fontsize=8
                )
    ax.set(
        xticks=[0, 1],
        xticklabels=["PPO régimen", "PPO backbone"],
        title="Cinco semillas por configuración; ninguna seleccionada",
        ylabel="Sharpe por semilla √221",
    )
    ax.axhline(0, color="gray", linestyle="--")
    save(fig, "06_semillas")
    fig, ax = plt.subplots(figsize=(10, 5))
    names = selected[1:]
    x = np.arange(len(names))
    ax.bar(
        x - 0.18,
        [metrics[a]["gross_sum_pct"] for a in names],
        0.36,
        label="Bruto: suma diaria",
        color="#0072B2",
    )
    ax.bar(
        x + 0.18,
        [-metrics[a]["cost_sum_pct"] for a in names],
        0.36,
        label="Costo: suma diaria negativa",
        color="#D55E00",
        hatch="//",
    )
    ax.set(
        xticks=x,
        xticklabels=names,
        ylabel="Suma de porcentajes diarios (no capital inicial)",
        title="Descomposición aritmética, distinta del retorno compuesto",
    )
    ax.tick_params(axis="x", labelrotation=15)
    ax.legend()
    save(fig, "07_bruto_costos")
    if quality.get("flat_ohlc_fraction_by_year"):
        fig, ax = plt.subplots(figsize=(9, 4))
        v = quality["flat_ohlc_fraction_by_year"]
        ax.bar(list(v), np.array(list(v.values())) * 100, color="#0072B2", hatch="//")
        ax.set(
            title="Representación de la fuente: O=H=L=C por año",
            ylabel="% barras planas",
            xlabel="Año; no demuestra bid/ask ni ejecutabilidad",
        )
        save(fig, "08_calidad_ohlc")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--replications", type=int, default=10_000)
    args = p.parse_args()
    print(json.dumps(build(args.snapshot, args.output, replications=args.replications)))


if __name__ == "__main__":
    main()
