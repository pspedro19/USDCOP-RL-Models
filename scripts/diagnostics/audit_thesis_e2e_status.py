"""Evidence-aware E2E audit. Local diagnostics only; no secrets or provider calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def safe_evidence_path(path: Path) -> Path:
    """Reject sensitive paths before any read, including paths embedded in manifests."""
    resolved = path.resolve()
    for candidate in (path, resolved):
        if (
            any(p.lower() == "secrets" or p.lower().startswith(".env") for p in candidate.parts)
            or candidate.suffix.lower() in {".pem", ".key"}
            or candidate.name.lower().startswith(("credentials", "service-account"))
        ):
            raise ValueError("sensitive evidence path refused")
    return resolved


def sha256(path: Path) -> str:
    path = safe_evidence_path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict | None:
    try:
        path = safe_evidence_path(path)
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def _prereg_signed(path: Path) -> bool:
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    return bool(re.search(r"^operator_signature:\s*SIGNED\s*$", text, re.MULTILINE))


def _context_artifact(path: Path) -> dict:
    if not path.is_file():
        return {"status": "NOT_GENERATED", "artifact": str(path)}
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        hashes = {row.get("dataset_sha256") for row in rows}
        if len(hashes) != 1:
            raise ValueError("mixed context identities")
        return {
            "status": "DIAGNOSTIC_ONLY",
            "contexts": len(rows),
            "dataset_sha256": next(iter(hashes)),
            "artifact": str(path),
        }
    except (OSError, ValueError):
        return {"status": "INVALID", "artifact": str(path)}


def _ledger_status(llm_dir: Path, provider: str, dates: list[str] | None = None) -> dict:
    # Explicit cohort path: never silently select the largest, newest or best ledger.
    path = llm_dir / f"decisions_{provider}_selection_diagnostic.jsonl"
    if not path.is_file():
        return {"status": "NOT_EXECUTED", "artifact": str(path)}
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        # Parallel historical calls append out of order. Validate logical order;
        # physical file order is not the position state or proof of live sealing.
        rows.sort(key=lambda row: (row["session_date"], row["bar"]))
        expected = {(d, b) for d in (dates or []) for b in range(59)}
        observed, ids, bad, provenance_missing = set(), set(), 0, set()
        previous, last_bar = {}, {}
        for row in rows:
            key = (row["session_date"], row["bar"])
            if key in observed or row["decision_id"] in ids:
                raise ValueError("duplicate decision/session-bar")
            observed.add(key)
            ids.add(row["decision_id"])
            day, bar = key
            if (
                row.get("valid_json") is not True
                or row.get("unavailable") is not False
                or row.get("dataset_block") != "selection"
            ):
                bad += 1
            w = row.get("weight")
            if not isinstance(w, int | float) or not -1 <= w <= 1:
                raise ValueError("invalid position")
            if row.get("previous_weight") != previous.get(day, 0.0):
                raise ValueError("nonrecursive previous position")
            if type(bar) is not int or bar != last_bar.get(day, -1) + 1:
                raise ValueError("out-of-order session bars")
            previous[day] = w
            last_bar[day] = bar
            for field in (
                "dataset_sha256",
                "cutoff_utc",
                "request_sha256",
                "response_id",
                "response_model",
                "api_version",
            ):
                if not row.get(field):
                    provenance_missing.add(field)
        complete = bool(expected) and observed == expected and not bad
        return {
            "status": ("RETROSPECTIVE_COMPLETE" if complete else "INCOMPLETE_OR_INVALID"),
            "artifact": str(path),
            "sha256": sha256(path),
            "decisions_sealed": len(rows),
            "decisions_expected": len(expected),
            "invalid_or_unavailable": bad,
            "missing_provenance_fields": sorted(provenance_missing),
            "strict_provenance_verified": complete and not provenance_missing,
            "confirmatory": False,
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"status": "INVALID", "artifact": str(path), "reason": str(exc)}


def verify_bundle(directory: Path | None) -> dict:
    if directory is None:
        return {"status": "NOT_SUPPLIED", "reason": "a directory of PNGs is not evidence"}
    from scripts.presentation.build_research_grade_thesis import Snapshot

    try:
        directory = directory.resolve()
        manifest = load_json(directory / "manifest.json")
        if not manifest or manifest.get("contract") != "THESIS-RETROSPECTIVE-BUNDLE-1":
            raise ValueError("missing versioned bundle manifest")
        snapshot_path = Path(manifest["snapshot_manifest"])
        if sha256(snapshot_path) != manifest["snapshot_sha256"]:
            raise ValueError("snapshot manifest hash changed")
        snapshot = Snapshot(snapshot_path)
        for name, digest in manifest["inputs_sha256"].items():
            if hashlib.sha256(snapshot.read(name)).hexdigest() != digest:
                raise ValueError("input hash differs from preserved evidence")
        artifacts = manifest["artifacts_sha256"]
        required = {
            "results.json",
            "daily_series.json",
            "metrics.csv",
            "results.md",
            "rolling_sharpe_data.json",
            "cost_stress.json",
            "actions_regime.json",
        }
        required |= {
            name + ext
            for name in (
                "01_capital",
                "02_drawdown",
                "03_sharpe_movil",
                "04_costos",
                "05_acciones_regimen",
                "06_semillas",
                "07_bruto_costos",
                "08_calidad_ohlc",
            )
            for ext in (".png", ".svg")
        }
        if not required <= artifacts.keys():
            raise ValueError("incomplete figure/table bundle")
        for name, digest in artifacts.items():
            path = (directory / name).resolve()
            if (
                not path.is_relative_to(directory)
                or path.suffix not in {".json", ".csv", ".md", ".png", ".svg"}
                or path.name.startswith(".env")
            ):
                raise ValueError("unsafe artifact path")
            if sha256(path) != digest:
                raise ValueError("figure/table content hash changed: " + name)
        result = load_json(directory / "results.json")
        if result.get("scope") != "retrospective_diagnostic":
            raise ValueError("bundle scope is not retrospective")
        return {
            "status": "REPRODUCED",
            "artifact_dir": str(directory),
            "manifest_sha256": sha256(directory / "manifest.json"),
            "figures": 8,
            "confirmatory": False,
            "unresolved": result.get("unresolved", []),
        }
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return {"status": "INVALID", "reason": str(exc), "artifact_dir": str(directory)}


def audit(
    root: Path = ROOT,
    *,
    bundle: Path | None = None,
    macro_report: Path | None = None,
    sanity_report: Path | None = None,
    vintage_capture: Path | None = None,
    expected_vintage_sha: str | None = None,
    expected_bundle_sha: str | None = None,
) -> dict:
    from src.research.sanity_gate import require_macro_identity, require_sanity_pass

    vintage_options = (vintage_capture, expected_vintage_sha, expected_bundle_sha)
    use_vintages = any(value is not None for value in vintage_options)
    if use_vintages and (bundle is None or any(value is None for value in vintage_options)):
        raise ValueError(
            "vintage capture, both expected hashes and bundle must be supplied together"
        )
    root = safe_evidence_path(root)
    repair = root / "outputs/thesis-repair"
    macro_path = macro_report or repair / "macro_identity_research_v2_latest.json"
    sanity_path = sanity_report or repair / "sanity_research_grade_20260912/protocol.json"
    for path in (macro_path, sanity_path):
        safe_evidence_path(path)
    controls = {}
    for name, path, check in (
        (
            "macro_numerical_identity",
            macro_path,
            lambda p: require_macro_identity(
                p,
                availability=root / "config/research/macro_availability.yaml",
                clean=root / "data/pipeline/04_cleaning/output/MACRO_RESEARCH_v2.parquet",
            ),
        ),
        ("current_synthetic_sanity", sanity_path, require_sanity_pass),
    ):
        try:
            check(path)
            controls[name] = {"status": "PASS", "artifact": str(path), "sha256": sha256(path)}
        except RuntimeError as exc:
            controls[name] = {"status": "BLOCKED", "artifact": str(path), "reason": str(exc)}
    portable = root / "data/thesis/research_data_portable_v2.pkl"
    dates = []
    try:
        with portable.open("rb") as handle:
            blob = pickle.load(handle)  # trusted local artifact; no held-out evaluation
        dates = [s.date.isoformat() for s in blob["selection"]]
        from src.research.dataset import dataset_identity

        current = root == ROOT and blob.get("identity") == dataset_identity()
        controls["portable_v2"] = {
            "status": "PASS" if current else "HISTORICAL_IDENTITY_NOT_CURRENT",
            "selection_sessions": len(dates),
            "sha256": sha256(portable),
            "historical_use_requires_expected_sha256": True,
        }
    except (OSError, ValueError, KeyError, pickle.PickleError, AttributeError) as exc:
        controls["portable_v2"] = {"status": "BLOCKED", "reason": str(exc)}

    directory = repair / "ppo_v2_recipe_flat"
    runs = []
    for config in ("ppo_regime", "ppo_backbone"):
        for seed in (42, 123, 456, 789, 1337):
            path = directory / f"{config}_seed{seed}.json"
            row = load_json(path)
            if row and row.get("config") == config and row.get("seed") == seed:
                runs.append(
                    {
                        "config": config,
                        "seed": seed,
                        "timesteps": row.get("timesteps"),
                        "sha256": sha256(path),
                    }
                )
    controls["ppo_10_diagnostic_runs"] = {
        "status": "RETROSPECTIVE_COMPLETE"
        if len(runs) == 10 and all(int(r["timesteps"] or 0) >= 300_000 for r in runs)
        else "INCOMPLETE",
        "count": len(runs),
        "runs": runs,
        "artifact_dir": str(directory),
        "confirmatory": False,
    }
    controls["ppo_10_confirmatory_runs"] = {
        "status": "NOT_EXECUTED",
        "reason": "ten retrospective files are not future replication",
    }
    llm_dir = root / "data/thesis/llm"
    controls["deepseek_ledger"] = _ledger_status(llm_dir, "deepseek", dates)
    controls["azure_ledger"] = _ledger_status(llm_dir, "azure", dates)
    controls["llm_diagnostic_contexts"] = _context_artifact(
        repair / "llm_selection_contexts_v2.jsonl"
    )
    controls["figures_v2"] = verify_bundle(bundle)
    controls["hybrid"] = {
        "status": "REPRODUCED"
        if controls["figures_v2"]["status"] == "REPRODUCED"
        else "UNVERIFIED",
        "scope": "retrospective_diagnostic",
    }
    cross = load_json(repair / "usdcop_daily_cross_source_v2.json") or {}
    controls["usdcop_daily_twelvedata_vs_investing"] = {
        "status": "HISTORICAL_DIAGNOSTIC_REPORTED" if cross else "MISSING",
        **{
            k: cross.get(k)
            for k in ("common_rows", "median_abs_diff_pct", "p95_abs_diff_pct", "max_abs_diff_pct")
        },
        "certifies_intraday_execution": False,
    }
    controls["macro_publication_and_vintages"] = {
        "status": "PENDING",
        "reason": "T-1 calendar shift and numerical identity do not prove historical availability",
    }
    controls["macro_vintage_diagnostic"] = {"status": "NOT_PROVIDED"}
    if use_vintages:
        from src.research.vintage_evidence_gate import verify_vintage_capture

        try:
            controls["macro_vintage_diagnostic"] = verify_vintage_capture(
                capture=vintage_capture,
                expected_capture_sha=expected_vintage_sha,
                bundle=bundle,
                expected_bundle_sha=expected_bundle_sha,
            )
            controls["macro_vintage_diagnostic"]["e2e_runner_sha256"] = sha256(Path(__file__))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            controls["macro_vintage_diagnostic"] = {
                "status": "INVALID",
                "reason": str(exc),
                "opening_availability_verified": False,
                "scientific_closure_ready": False,
            }
    controls["executable_costs"] = {"status": "ASSUMED_NOT_MEASURED"}
    controls["trial_ledger_reconciliation"] = {"status": "OPERATOR_RECONCILIATION_REQUIRED"}
    controls["preregistration"] = {
        "status": "HISTORICAL_SIGNATURE_PRESENT"
        if _prereg_signed(root / ".claude/specs/planes/06-PRE-REGISTRATION-v3.md")
        else "SIGNATURE_NOT_FOUND",
        "licenses_new_experiment": False,
    }
    engineering = all(
        controls[k]["status"] == "PASS"
        for k in ("macro_numerical_identity", "current_synthetic_sanity", "portable_v2")
    )
    return {
        "contract": "CTR-RESEARCH-THESIS-E2E-STATUS-002",
        "network_called": False,
        "secrets_read": False,
        "controls": controls,
        "engineering_ready": engineering,
        "retrospective_results_reproduced": controls["figures_v2"]["status"] == "REPRODUCED",
        "scientific_closure_ready": False,
        "confirmatory_evidence_ready": False,
        "confirmatory_ready": False,
        "interpretation": "File presence, a signature, and old PASS flags are not scientific certification.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--macro-report", type=Path)
    parser.add_argument("--sanity-report", type=Path)
    parser.add_argument("--vintage-capture", type=Path)
    parser.add_argument("--expected-vintage-sha")
    parser.add_argument("--expected-bundle-sha")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing audit")
    try:
        safe_evidence_path(args.output)
        report = audit(
            bundle=args.bundle,
            macro_report=args.macro_report,
            sanity_report=args.sanity_report,
            vintage_capture=args.vintage_capture,
            expected_vintage_sha=args.expected_vintage_sha,
            expected_bundle_sha=args.expected_bundle_sha,
        )
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "engineering_ready": report["engineering_ready"],
                "retrospective_results_reproduced": report["retrospective_results_reproduced"],
                "scientific_closure_ready": False,
            }
        )
    )
    return 0  # report generated; inspect explicit gates, not this exit code, for readiness


if __name__ == "__main__":
    raise SystemExit(main())
