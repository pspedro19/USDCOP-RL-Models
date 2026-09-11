"""Brazo RL forward: la política congelada de la tesis, sellada en el mismo ledger.

Contract: CTR-RESEARCH-FORWARD-001 · Date: 2026-08-25

## Qué hace comparable a esto con el brazo LLM

Nada de lo que hay aquí es nuevo. El modelo es el mismo `.zip` que produjo el resultado del
hold-out, el spec lo construye `build_live_spec` —verificado elemento a elemento contra el del
batch— y la liquidación usa `run_session`, el mismo motor que dio las tablas de la tesis.

Lo único que cambia es **cuándo**: en vez de evaluar 584 sesiones pasadas de golpe, decide una
sesión al día y sella la decisión antes de que exista el resultado, en el mismo ledger
encadenado que el LLM.

Sin eso no hay comparación posible. Un brazo medido sobre el pasado y otro sobre el futuro no
se comparan: se yuxtaponen.

## Los dos brazos, y por qué son dos

| `arm_id` | Decisiones | Qué responde |
|---|---|---|
| `ppo_regime_fwd_k1` | 59, nativo | ¿la política que la tesis evaluó se comporta igual sobre datos nuevos? |
| `ppo_regime_fwd_k59` | 1, sostenida | comparable cabeza a cabeza con el LLM, que decide una vez |

`k59` no es un capricho: el LLM emite **un** score antes de la apertura. Comparar eso contra un
agente que decide 59 veces mezclaría el efecto de la información con el de la frecuencia, y la
curva de frecuencia del hold-out ya demostró que la frecuencia domina — a `k=59` el bruto de
`ppo_regime` cae de +27,95% a −5,95%.

## La asimetría de información, otra vez

El RL necesita la barra 0 para construir su observación, así que **sella a las 08:00**, no a
las 07:15 como el LLM. Los dos sellan antes de que exista `r_1`, así que los dos son causalmente
limpios — pero el RL ve una barra que el LLM no ve. Va declarado en el pre-registro y en
`information_edge` de cada registro, no en una nota al pie.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.research.live_spec import (build_live_spec, build_live_spec_partial)  # noqa: E402
from src.research.llm_forward.canonical import sha256_text  # noqa: E402
from src.research.llm_forward.decide import arm_spec, load_preregistration  # noqa: E402
from src.research.llm_forward.ledger import Ledger, LedgerError  # noqa: E402
from src.research.llm_forward.paths import DECISIONS_PATH, PREREG_PATH  # noqa: E402
from src.research.llm_forward.schema import (Decision, DecisionRecord,  # noqa: E402
                                             utc_now_iso)
from src.research.session_env import EXPOSURE_LEVELS, OPERABLE_RETURNS  # noqa: E402
from src.research.session_gym import SessionTradingEnv, position_state  # noqa: E402
from src.research.dataset import SEED_M5  # noqa: E402

MODELS_DIR = REPO / "data" / "thesis" / "ppo"


def _direction(w: float) -> str:
    return "flat" if w == 0.0 else ("long" if w > 0 else "short")


def decide_weights(spec, model_path: Path, hold_all_session: bool) -> np.ndarray:
    """Senda de exposición de la política congelada para esta sesión.

    Con `hold_all_session` se toma **solo la primera decisión** y se sostiene hasta el cierre
    — el equivalente de `k=59` en la curva de frecuencia del hold-out. Sin él, el agente
    decide en las 59 barras, que es su política nativa.
    """
    from stable_baselines3 import PPO

    model = PPO.load(str(model_path), device="cpu")
    env = SessionTradingEnv([spec], seed=0, shuffle=False)
    obs, _ = env.reset()

    if hold_all_session:
        action, _ = model.predict(obs, deterministic=True)
        return np.full(OPERABLE_RETURNS, float(EXPOSURE_LEVELS[int(action)]))

    weights, done = [], False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(int(action))
        weights.append(float(EXPOSURE_LEVELS[int(action)]))
    return np.asarray(weights, dtype=float)


def decide_first_weight(partial, model_path: Path) -> float:
    """Primera decisión usando exclusivamente la barra 0 ya cerrada."""
    from stable_baselines3 import PPO

    model = PPO.load(str(model_path), device="cpu")
    obs = np.concatenate([partial.market[0], position_state(0.0, 0, 0.0, 0.0, 0),
                          partial.context]).astype(np.float32)
    action, _ = model.predict(obs, deterministic=True)
    return float(EXPOSURE_LEVELS[int(action)])


def run(session_date: str, arm_id: str = "ppo_regime_fwd_k59",
        now_override: datetime | None = None, dry_run: bool = False) -> int:
    """Sella la decisión del brazo RL para una sesión."""
    spec_yaml, prereg_hash = load_preregistration(PREREG_PATH)
    arm = arm_spec(spec_yaml, arm_id)
    if arm["kind"] != "rl_frozen":
        raise ValueError(f"{arm_id} es de tipo {arm['kind']!r}; este job sella brazos RL.")

    model_path = REPO / arm["model"]
    if not model_path.is_file():
        raise FileNotFoundError(f"falta el modelo congelado {model_path}")

    hold = int(arm["decisions_per_session"]) == 1
    if hold:
        # k59 is the only arm eligible to seal on the first bar. The native 59-decision
        # arm remains legacy/excluded until per-bar state persistence is implemented.
        m5 = __import__("pandas").read_parquet(SEED_M5)
        t = __import__("pandas").to_datetime(m5["time"])
        first = m5[t.dt.date == __import__("pandas").Timestamp(session_date).date()].head(1)
        partial = build_live_spec_partial(session_date, first)
        session_spec = build_live_spec(session_date)
        w = decide_first_weight(partial, model_path)
        weights = np.full(OPERABLE_RETURNS, w)
        partial_sealed = True
    else:
        session_spec = build_live_spec(session_date)
        weights = decide_weights(session_spec, model_path, hold_all_session=False)
        partial_sealed = False

    w0 = float(weights[0])
    n_changes = int(np.count_nonzero(np.diff(np.concatenate([[0.0], weights]))))

    if dry_run:
        print(f"{arm_id} {session_date}: w0={w0:+.2f} cambios={n_changes} "
              f"spread={session_spec.spread_pips:.2f} (dry-run, nada escrito)")
        return 0

    now = now_override or datetime.now(timezone.utc)
    # El RL sella en la barra 0, no antes de la apertura: necesita esa barra para construir
    # su observacion. Lo que importa para la limpieza causal es que sea antes de `r_1`, que
    # nace al cierre de la barra 1.
    record = DecisionRecord(
        seq=-1,
        decision_id=f"{session_date}::{arm_id}",
        session_date=session_date,
        emitted_at_utc=utc_now_iso(),
        cutoff_utc=now.isoformat(timespec="seconds"),
        session_open_utc=now.isoformat(timespec="seconds"),
        sealed_before_open=False,
        preregistration_sha256=prereg_hash,
        # No hay prompt: se hashea la senda de exposicion, que es el analogo — el objeto que
        # define la decision y que tiene que quedar sellado antes del resultado.
        prompt_sha256=sha256_text(",".join(f"{w:+.2f}" for w in weights)),
        model=model_path.name,
        provider="rl_frozen",
        temperature=0.0,
        seed=None,
        corpus=[],
        decision=Decision(
            score=w0,
            direction=_direction(w0),
            confidence=1.0,     # politica determinista: no emite incertidumbre
            rationale=(f"politica congelada, {len(weights)} decisiones, {n_changes} cambios, "
                       f"spread esperado {session_spec.spread_pips:.2f} pips"),
        ),
        abstained=False,
        abstain_reason=None,
        usage=None,
        # La senda completa: para el brazo nativo la decision ES la senda, no solo `w_0`.
        decision_path=[float(w) for w in weights],
        # El contrato de costos del dia, sellado ANTES del resultado.
        spread_pips=float(session_spec.spread_pips),
        information_edge=(
            "ve la barra 0 de la sesion para construir su observacion; el brazo LLM no. "
            "Los dos sellan antes de que exista r_1."
        ),
        sealed_before_next_bar=partial_sealed,
        bar_index=0 if partial_sealed else None,
        bar_received_at_utc=now.isoformat(timespec="seconds") if partial_sealed else None,
    )

    ledger = Ledger(DECISIONS_PATH, key_field="decision_id")
    try:
        written = ledger.append(record)
    except LedgerError as exc:
        print(f"  [rechazado] {exc}", file=sys.stderr)
        return 1

    print(f"  sellado seq={written['seq']} {arm_id} w0={w0:+.2f} "
          f"hash={written['record_hash'][:12]}...")
    return 0


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Sella la decision del brazo RL congelado.")
    ap.add_argument("--session-date", required=True)
    ap.add_argument("--arm", default="ppo_regime_fwd_k59")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    return run(args.session_date, arm_id=args.arm, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
