"""Fail-closed gates: replayable evidence, not hand-written PASS flags."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evidence_path(entry: dict, directory: Path, suffix: str) -> Path:
    path = Path(entry["path"]).resolve()
    if not path.is_relative_to(directory.resolve()) or path.suffix != suffix:
        raise ValueError("sanity artifact must remain inside its evidence directory")
    if _sha256(path) != entry.get("sha256"):
        raise ValueError("sanity artifact hash changed")
    return path


def _passes(fixture: str, stats: dict) -> bool:
    fields = ("mean_net", "mean_gross", "mean_cost", "mean_abs_exposure", "oracle_mean_net")
    if not all(np.isfinite(stats.get(field, np.nan)) for field in fields):
        raise ValueError("nonfinite or missing sanity statistics")
    if stats["mean_cost"] < 0 or not 0 <= stats["mean_abs_exposure"] <= 1:
        raise ValueError("sanity costs/exposure are outside their physical bounds")
    daily = np.asarray(stats.get("daily_returns", []), dtype=float)
    if (daily.shape != (100,) or not np.isfinite(daily).all()
            or stats.get("n_eval_sessions") != 100
            or not np.isclose(daily.mean(), stats["mean_net"], atol=1e-12, rtol=0)
            or not np.isclose(stats["mean_gross"] - stats["mean_cost"],
                              stats["mean_net"], atol=1e-12, rtol=0)):
        raise ValueError("sanity daily evidence does not reproduce its metrics")
    if fixture in ("S1", "S4"):
        return stats["mean_abs_exposure"] < .1 and stats["mean_net"] > -.005
    if fixture == "S2":
        return (stats["mean_cost"] == 0 and stats["oracle_mean_net"] > 0
                and stats["mean_net"] >= .7 * stats["oracle_mean_net"])
    return stats["mean_net"] > 0


def _verify_checkpoint(path: Path, row: dict, manifest: dict) -> None:
    """Check SB3's plain JSON metadata; never execute serialized checkpoint objects."""
    with zipfile.ZipFile(path) as archive:
        if not {"data", "policy.pth"} <= set(archive.namelist()):
            raise ValueError("checkpoint lacks SB3 metadata or policy weights")
        if archive.getinfo("data").file_size > 4 * 1024 * 1024:
            raise ValueError("oversized SB3 checkpoint metadata")
        metadata = json.loads(archive.read("data"))
    if (type(metadata.get("seed")) is not int or type(metadata.get("num_timesteps")) is not int
            or metadata["seed"] != row["seed"]
            or metadata["num_timesteps"] != row["timesteps_effective"]):
        raise ValueError("checkpoint seed/effective steps contradict the run result")
    effective = manifest["effective_recipe"]
    for key, expected in effective["ppo_kwargs"].items():
        actual = metadata.get(key)
        if key == "clip_range" and isinstance(actual, dict):
            if actual.get("value_schedule") != f"ConstantSchedule(val={expected})":
                raise ValueError("checkpoint clip schedule differs from the frozen recipe")
            continue
        if (not isinstance(actual, (int, float, bool)) or not np.isfinite(actual)
                or actual != expected):
            raise ValueError(f"checkpoint {key} differs from the frozen effective recipe")
    policy = metadata.get("policy_kwargs", {})
    if (policy.get("net_arch") != effective["policy_kwargs"]["net_arch"]
            or policy.get("activation_fn") != "<class 'torch.nn.modules.activation.Tanh'>"):
        raise ValueError("checkpoint policy architecture differs from the frozen recipe")


def require_sanity_pass(report: Path, *, fixtures: tuple[str, ...] = ("S1", "S2", "S3", "S4")) -> dict:
    """Require all current controls, five unique seeds, unseen sessions and hashes."""
    from src.research.ppo_recipe import sessions_sha256
    from src.research.synthetic_sessions import (
        SANITY_SEEDS, UNSEEN_SEED_OFFSET, oracle_result,
        sanity_session_splits, validate_sanity_manifest,
    )
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
        manifest = validate_sanity_manifest(payload)
        if (set(fixtures) != {"S1", "S2", "S3", "S4"}
                or payload.get("protocol") != "S1-S4"
                or payload.get("synthetic_only") is not True
                or payload.get("market_evidence") is not False
                or payload.get("market_trials_charged") != 0
                or manifest["timesteps_requested"] < 100_000):
            raise ValueError("incomplete/current protocol required; smoke is not sanity evidence")
        reports = payload.get("fixtures", {})
        if set(reports) != set(fixtures):
            raise ValueError("all four fixtures are mandatory")
        evidence = payload.get("input_evidence", [])
        if len(evidence) != 4:
            raise ValueError("missing independently hashed fixture reports")
        external = {}
        for entry in evidence:
            path = _evidence_path(entry, report.parent, ".json")
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("fixture") in external:
                raise ValueError("duplicate external fixture")
            external[value.get("fixture")] = value
        if external != reports:
            raise ValueError("aggregate differs from original fixture reports")
        for fixture, subreport in reports.items():
            validate_sanity_manifest(subreport)
            if subreport.get("fingerprint") != payload["fingerprint"]:
                raise ValueError("mixed versions of controls")
            rows = subreport.get("rows", [])
            if len(rows) != 5 or {row.get("seed") for row in rows} != set(SANITY_SEEDS):
                raise ValueError("five unique frozen seeds required")
            passing = []
            for row in rows:
                seed = row["seed"]
                steps = row.get("timesteps_effective", 0)
                requested = manifest["timesteps_requested"]
                if (row.get("fixture") != fixture or row.get("probe") != manifest["probe"]
                        or row.get("fingerprint") != payload["fingerprint"]
                        or row.get("identity_unchanged") is not True
                        or row.get("timesteps_requested") != requested
                        or not requested <= steps < requested + 4096
                        or row.get("train_generation_seed") != seed
                        or row.get("unseen_generation_seed") != seed + UNSEEN_SEED_OFFSET):
                    raise ValueError("stale run or incorrect effective training protocol")
                artifacts = row.get("artifacts", {})
                if set(artifacts) != {"checkpoint", "vecnormalize", "manifest"}:
                    raise ValueError("missing checkpoint/normalizer/frozen manifest")
                paths = {name: _evidence_path(artifacts[name], report.parent, suffix)
                         for name, suffix in (("checkpoint", ".zip"), ("vecnormalize", ".pkl"),
                                              ("manifest", ".json"))}
                _verify_checkpoint(paths["checkpoint"], row, manifest)
                frozen = json.loads(paths["manifest"].read_text(encoding="utf-8"))
                if (frozen.get("manifest") != manifest or frozen.get("seed") != seed
                        or frozen.get("fixture") != fixture
                        or frozen.get("fingerprint") != payload["fingerprint"]):
                    raise ValueError("run manifest differs from evaluated recipe")
                training, unseen = sanity_session_splits(fixture, seed)
                for split, sessions in (("training", training), ("unseen", unseen)):
                    expected = sessions_sha256(sessions)
                    if (row.get(split + "_dataset_sha256") != expected
                            or frozen.get(split + "_dataset_sha256") != expected):
                        raise ValueError("synthetic dataset hash differs from current fixture")
                for split, sessions in (("train", training[:100]), ("unseen", unseen)):
                    if row[split].get("dataset_sha256") != sessions_sha256(sessions):
                        raise ValueError("evaluation did not use the declared unseen sessions")
                    oracle = float(np.mean([oracle_result(fixture, s).daily_return for s in sessions]))
                    if not np.isclose(row[split].get("oracle_mean_net", np.nan), oracle,
                                      atol=1e-12, rtol=0):
                        raise ValueError("sanity oracle differs from the matched generated sessions")
                train_pass = _passes(fixture, row["train"])
                unseen_pass = _passes(fixture, row["unseen"])
                if (row.get("train_pass") != train_pass or row.get("unseen_pass") != unseen_pass
                        or row.get("passed") != (train_pass and unseen_pass)):
                    raise ValueError("seed pass flag contradicts recomputed metrics")
                if train_pass and unseen_pass:
                    passing.append(seed)
            if len(passing) < 4 or subreport.get("passed") is not True:
                raise ValueError(f"{fixture}: fewer than four of five seeds pass train AND unseen")
        if payload.get("passed") is not True:
            raise ValueError("protocol did not pass")
        return {**payload, "selected_probe": manifest["probe"]}
    except (OSError, ValueError, TypeError, KeyError, zipfile.BadZipFile) as exc:
        raise RuntimeError(f"sanity gate: {exc}") from exc


def require_macro_identity(report: Path, *, availability: Path | None = None,
                           clean: Path | None = None) -> dict:
    """Numerically replay all declared reference payloads before allowing training."""
    from src.research.macro_evidence import require_macro_evidence
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
        result = require_macro_evidence(
            payload,
            availability=availability or ROOT / "config/research/macro_availability.yaml",
            clean=clean or ROOT / "data/pipeline/04_cleaning/output/MACRO_RESEARCH_v2.parquet",
        )
        return {**payload, "verification": result}
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise RuntimeError(f"macro identity gate: {exc}") from exc
