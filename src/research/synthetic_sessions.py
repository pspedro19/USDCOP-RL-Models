"""Deterministic synthetic sessions with known economic optima.

These fixtures are controls for the PPO recipe, never evidence about USD/COP or
gold.  The planted direction is exposed in market feature zero and predicts the
*next* bar only, matching the session environment chronology.
"""

from __future__ import annotations

from dataclasses import asdict
from enum import StrEnum

import numpy as np

from src.research.cost_model import ZERO_COST_PARAMETERS, CostParameters
from src.research.features import GROUPS
from src.research.session_env import BARS_PER_SESSION, OPERABLE_RETURNS, run_session
from src.research.session_gym import SessionSpec

N_MARKET = sum(len(GROUPS[g]) for g in ("precio", "volatilidad", "tendencia", "temporal"))
N_CONTEXT = len(GROUPS["macro"]) + len(GROUPS["regimen"])
FIXTURE_VERSION = "research-grade-synthetic-v1"
SANITY_SEEDS = (42, 123, 456, 789, 1337)
UNSEEN_SEED_OFFSET = 1_000_003


class Fixture(StrEnum):
    NOISE_WITH_COST = "S1"
    SIGNAL_PLANTED = "S2"
    SIGNAL_ABOVE_COST = "S3"
    SIGNAL_BELOW_COST = "S4"


def _alpha(fixture: Fixture) -> float:
    return {
        Fixture.NOISE_WITH_COST: 0.0,
        Fixture.SIGNAL_PLANTED: 0.002,
        Fixture.SIGNAL_ABOVE_COST: 0.002,
        Fixture.SIGNAL_BELOW_COST: 0.00015,
    }[fixture]


def _spread_pips(fixture: Fixture) -> float:
    """Spread per fixture; S2 also disables commission and slippage explicitly.

    Sin esto, S2 y S3 eran la MISMA fixture byte a byte: ambas declaran `alpha = 0.002` y
    ambas heredaban `spread_pips = 3.0`, asi que generaban cierres y features identicos con la
    misma semilla. La bateria decia probar cuatro condiciones y probaba tres, con una contada
    dos veces. Medido el 2026-09-11.

    Lo que cada una debe aislar, segun el diseno:
      * **S2** senal plantada **sin coste** -> ¿la receta aprende la senal siquiera?
      * **S3** la misma senal **pagando coste** -> ¿opera cuando el alfa lo supera?

    Con el mismo coste en ambas, S2 no puede responder su pregunta: un fallo de S2 seria
    indistinguible de un fallo de S3, y un aprobado de S3 hacia redundante a S2.
    """
    return 0.0 if fixture is Fixture.SIGNAL_PLANTED else 3.0


def make_sessions(
    fixture: Fixture | str, n: int = 500, seed: int = 0, spread_pips: float | None = None
) -> list[SessionSpec]:
    """Generate reproducible ``SessionSpec`` objects for one control fixture."""
    fixture = Fixture(fixture)
    if spread_pips is None:
        spread_pips = _spread_pips(fixture)
    if fixture is Fixture.SIGNAL_PLANTED and spread_pips != 0.0:
        raise ValueError("S2 is a zero-cost control; a spread override changes its identity")
    rng = np.random.default_rng(seed)
    alpha = _alpha(fixture)
    sessions: list[SessionSpec] = []
    for i in range(n):
        direction = rng.choice(np.array([-1.0, 1.0]), size=OPERABLE_RETURNS)
        noise = rng.normal(0.0, 0.0004, size=OPERABLE_RETURNS)
        returns = direction * alpha + noise
        close = np.empty(BARS_PER_SESSION, dtype=float)
        close[0] = 4000.0
        close[1:] = close[0] * np.cumprod(1.0 + returns)
        # ObservationContract requires every normalized input to stay within the
        # same ±5 clip used by the market pipeline.  Apply it here as well so a
        # rare normal draw cannot invalidate an otherwise deterministic fixture.
        market = np.clip(rng.normal(0.0, 1.0, size=(BARS_PER_SESSION, N_MARKET)), -5.0, 5.0).astype(
            np.float32
        )
        # The first feature is available at b and predicts r[b+1].  S1 has no
        # predictive relation because its return alpha is zero.
        market[:OPERABLE_RETURNS, 0] = direction.astype(np.float32)
        market[-1, 0] = 0.0
        context = np.zeros(N_CONTEXT, dtype=np.float32)
        sessions.append(
            SessionSpec(
                date=f"synthetic-{i:04d}",
                close=close,
                market=market,
                context=context,
                spread_pips=spread_pips,
                cost_parameters=(
                    ZERO_COST_PARAMETERS if fixture is Fixture.SIGNAL_PLANTED else None
                ),
            )
        )
    return sessions


def oracle_weights(fixture: Fixture | str, spec: SessionSpec) -> np.ndarray:
    """Analytic policy for fixture assertions, not a learned policy."""
    fixture = Fixture(fixture)
    if fixture in (Fixture.NOISE_WITH_COST, Fixture.SIGNAL_BELOW_COST):
        return np.zeros(OPERABLE_RETURNS, dtype=float)
    return np.sign(spec.market[:OPERABLE_RETURNS, 0]).astype(float)


def oracle_result(fixture: Fixture | str, spec: SessionSpec):
    return run_session(
        spec.close,
        oracle_weights(fixture, spec),
        spec.spread_pips,
        date=spec.date,
        cost_parameters=getattr(spec, "cost_parameters", None),
    )


def sanity_session_splits(
    fixture: Fixture | str, seed: int, *, n_train: int = 500, n_eval: int = 100
):
    """Generate training and genuinely unseen sessions using disjoint RNG seeds."""
    from dataclasses import replace

    training = make_sessions(fixture, n=n_train, seed=seed)
    unseen_seed = UNSEEN_SEED_OFFSET + seed
    unseen = make_sessions(fixture, n=n_eval, seed=unseen_seed)
    training = [replace(s, date=f"synthetic-train-{seed}-{i:04d}") for i, s in enumerate(training)]
    unseen = [
        replace(s, date=f"synthetic-unseen-{unseen_seed}-{i:04d}") for i, s in enumerate(unseen)
    ]
    return training, unseen


def fixture_manifest(fixture: Fixture | str) -> dict:
    fixture = Fixture(fixture)
    parameters = ZERO_COST_PARAMETERS if fixture is Fixture.SIGNAL_PLANTED else CostParameters()
    return {
        "version": FIXTURE_VERSION,
        "fixture": fixture.value,
        "alpha": _alpha(fixture),
        "noise_std_per_bar": 0.0004,
        "spread_cop_per_usd": _spread_pips(fixture),
        "fees": asdict(parameters),
        "n_bars": BARS_PER_SESSION,
        "n_market": N_MARKET,
        "n_context": N_CONTEXT,
        "planted_feature": 0,
        "signal_predicts": "next_bar_direction",
        "zero_total_cost": fixture is Fixture.SIGNAL_PLANTED,
    }


def sanity_protocol_manifest(probe: str = "flat_init_no_turn", timesteps: int = 100_000) -> dict:
    from src.research.ppo_recipe import (
        FROZEN_SANITY_PROBE,
        effective_recipe,
        library_versions,
        training_code_hashes,
    )

    if probe != FROZEN_SANITY_PROBE:
        raise ValueError("research-grade sanity uses the frozen probe; no search is authorized")
    if timesteps <= 0:
        raise ValueError("timesteps must be positive")
    return {
        "schema_version": "research-grade-sanity-v1",
        "probe": probe,
        "timesteps_requested": timesteps,
        "seeds": list(SANITY_SEEDS),
        "n_training_sessions": 500,
        "n_training_eval_sessions": 100,
        "n_unseen_sessions": 100,
        "unseen_seed_rule": {"offset": UNSEEN_SEED_OFFSET, "operation": "seed+offset"},
        "fixtures": {f.value: fixture_manifest(f) for f in Fixture},
        "effective_recipe": effective_recipe(probe),
        "source_sha256": training_code_hashes(),
        "library_versions": library_versions(),
        "pass_rule": {
            "minimum_passing_seeds": 4,
            "required_seeds": list(SANITY_SEEDS),
            "require_train_and_unseen": True,
            "S1_S4": {"max_mean_abs_exposure_exclusive": 0.1, "min_mean_net_exclusive": -0.005},
            "S2": {"min_fraction_of_matched_oracle": 0.70, "require_zero_total_cost": True},
            "S3": {"min_mean_net_exclusive": 0.0},
        },
    }


def sanity_fingerprint(probe: str = "flat_init_no_turn", timesteps: int = 100_000) -> str:
    from src.research.ppo_recipe import canonical_sha256

    return canonical_sha256(sanity_protocol_manifest(probe, timesteps))


def validate_sanity_manifest(report: dict) -> dict:
    """Reject stale code/config/fees/library evidence even if its booleans say PASS."""
    from src.research.ppo_recipe import canonical_sha256

    if report.get("schema_version") != "research-grade-sanity-v1":
        raise ValueError("sanity evidence lacks the research-grade manifest")
    manifest = report.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError("missing sanity manifest")
    expected = sanity_protocol_manifest(
        manifest.get("probe"), manifest.get("timesteps_requested", 0)
    )
    if manifest != expected or report.get("fingerprint") != canonical_sha256(expected):
        raise ValueError("stale sanity evidence: code, fixture, fees or effective recipe changed")
    return manifest
