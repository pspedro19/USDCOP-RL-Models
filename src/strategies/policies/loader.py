"""Policy spec loader + factory (BL-47 R6-R8).

Reads the declarative specs under ``config/policies/*.yaml`` and returns a
``Policy`` (``src/contracts/policy.py``). The factory branches ONLY on
``engine.type`` and ``engine.implementation.mode`` — never on ``id``
(invariant 1 / §7.3 of ``.claude/rules/strategy-engines.md``).

Security (invariant 6): a spec may reference a coded policy by module path,
but ONLY under ``src.strategies.policies.`` — a YAML can never name an
arbitrary import target, and it can never carry code. Declarative specs go
through the whitelist AST of ``src/contracts/policy_dsl.py``; there is no
eval/exec/SQL path anywhere in this loader.

Freeze semantics (invariant 2): ``policy_hash`` is DERIVED from the economic
content of the spec (implementation + params + required features + resolution
+ fallbacks). ``governance.policy_hash``, when declared, must equal the
derived value — that is what makes "congelar la receta ES congelar la
estrategia" checkable instead of aspirational.

Spec: .claude/specs/planes/05-rule-based-strategies.md §3, §4, §7
Contract: CTR-POLICY-001 (consumed, not modified — BL-46 owns it)
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.contracts.policy import (
    ENGINE_TYPES,
    FALLBACK_MODES,
    policy_canonical_hash,
)
from src.contracts.policy_dsl import DeclarativePolicy

#: Only modules under this package may be named by a spec (no arbitrary import).
ALLOWED_MODULE_PREFIX = "src.strategies.policies."

IMPLEMENTATION_MODES = ("declarative", "coded_policy")

#: Explicit fallback vocabulary (invariant 9: sin default explícito no hay freeze).
#:
#: NOT a local tuple: both names are THE SAME OBJECT as
#: ``src.contracts.policy.FALLBACK_MODES`` (and therefore as
#: ``policy_engine.FALLBACK_MODES``), so "lo que el validador acepta" and "lo
#: que el motor sabe ejecutar" cannot drift — they are one value, asserted with
#: identity in tests/unit/test_policy_specs.py.
#:
#: They used to be two tuples and ``STALE_INPUT_POLICIES`` also listed ``HOLD``,
#: which ``evaluate_policy`` rejects: a spec declaring it PASSED validation and
#: then died with ValueError in the engine (K-034). ``HOLD`` needs an explicit
#: previous exposure the evaluator never receives; see the note on
#: ``FALLBACK_MODES`` in the contract for why it is not aliased to ``FLAT``.
MISSING_INPUT_POLICIES = FALLBACK_MODES
STALE_INPUT_POLICIES = FALLBACK_MODES

#: Migration states of BL-47. ``SPEC_ONLY`` = the spec is the documento de
#: record but no runnable policy exists yet (fail-closed on build_policy).
MIGRATION_STATES = ("SPEC_ONLY", "PARITY_PENDING", "PARITY_GREEN", "CUTOVER")

_REPO_ROOT = Path(__file__).resolve().parents[3]


class PolicySpecError(ValueError):
    """A spec that violates the policy contract — always fail-closed."""


def policy_specs_dir() -> Path:
    return _REPO_ROOT / "config" / "policies"


# ---------------------------------------------------------------------------
# Canonical hash of the ECONOMIC content (presentation/comments excluded)
# ---------------------------------------------------------------------------

def canonical_policy_payload(spec: Mapping[str, Any]) -> dict[str, Any]:
    """The subset of the spec that decides. Changing any of it = new version."""
    engine = spec.get("engine", {})
    inputs = spec.get("inputs", {})
    policy = spec.get("policy", {})
    return {
        "id": spec.get("id"),
        "version": spec.get("version"),
        "engine": {
            "type": engine.get("type"),
            "retrain": engine.get("retrain"),
            "implementation": engine.get("implementation", {}),
        },
        "inputs": {
            "feature_set_id": inputs.get("feature_set_id"),
            "resample_policy_id": inputs.get("resample_policy_id"),
            "required_features": list(inputs.get("required_features", [])),
            "optional_features": list(inputs.get("optional_features", [])),
            "decision_point": inputs.get("decision_point"),
            "execution_ref": inputs.get("execution_ref"),
            # `max_snapshot_age` entra CONDICIONALMENTE (CXD-610). Razon de fondo:
            # el umbral de frescura decide CUANDO opera la policy -- con `P1D` una
            # serie de ayer bloquea y con `P30D` pasa-- asi que pertenece al
            # subconjunto que decide, igual que la ventana o el umbral de la regla.
            # Dejarlo fuera permitiria mover el comportamiento sin que el
            # `policy_hash` se moviera un bit, que es congelar la receta y dejar el
            # gatillo suelto.
            #
            # Condicional y no incondicional porque incluir la clave con `None`
            # cambiaria el payload de los CUATRO specs vigentes -- ninguno la
            # declara-- y con el sus `policy_hash` ya publicados y congelados. Un
            # re-freeze masivo por una mejora de contrato es exactamente el tipo de
            # ruido que hace que los muros de congelacion dejen de creerse.
            **({"max_snapshot_age": inputs["max_snapshot_age"]}
               if "max_snapshot_age" in inputs else {}),
        },
        "policy": {
            "params": policy.get("params", {}),
            "resolution": policy.get("resolution", {}),
            "rules": policy.get("rules", []),
            "missing_input_policy": policy.get("missing_input_policy"),
            "stale_input_policy": policy.get("stale_input_policy"),
        },
    }


def canonical_policy_hash(spec: Mapping[str, Any]) -> str:
    """Freeze digest of a spec's economic content — via the family SSOT.

    The PROJECTION (which keys decide) is this module's business; the
    SERIALISATION is not. Delegating keeps one hash idiom in the family
    (INTEGRATION-CONTRACT.md F-02) without moving a single frozen digest.
    """
    return policy_canonical_hash(canonical_policy_payload(spec))


# ---------------------------------------------------------------------------
# Loading + structural validation
# ---------------------------------------------------------------------------

def load_policy_spec(path: str | Path) -> dict[str, Any]:
    """Parse + structurally validate one spec (``yaml.safe_load``: no tags)."""
    p = Path(path)
    try:
        spec = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - malformed yaml
        raise PolicySpecError(f"{p.name}: YAML inválido ({exc})") from exc
    if not isinstance(spec, Mapping):
        raise PolicySpecError(f"{p.name}: el spec debe ser un mapping")
    spec = dict(spec)
    spec["_source_path"] = str(p)
    validate_policy_spec(spec)
    return spec


def load_all_policy_specs(directory: str | Path | None = None) -> list[dict[str, Any]]:
    d = Path(directory) if directory is not None else policy_specs_dir()
    return [load_policy_spec(f) for f in sorted(d.glob("*.yaml"))]


def validate_policy_spec(spec: Mapping[str, Any]) -> None:
    """The §11 structural checks that do not need data (fail-closed)."""
    name = Path(str(spec.get("_source_path", spec.get("id", "<spec>")))).name

    for field in ("id", "version", "asset", "engine", "inputs", "policy", "governance"):
        if field not in spec:
            raise PolicySpecError(f"{name}: falta el bloque obligatorio '{field}'")

    engine = spec["engine"]
    if engine.get("type") not in ENGINE_TYPES:
        raise PolicySpecError(
            f"{name}: engine.type debe ser uno de {ENGINE_TYPES}, es {engine.get('type')!r}"
        )
    impl = engine.get("implementation", {})
    if impl.get("mode") not in IMPLEMENTATION_MODES:
        raise PolicySpecError(
            f"{name}: engine.implementation.mode debe ser uno de {IMPLEMENTATION_MODES}"
        )

    # §11: rule_based ⇒ retrain=never, sin capability train, sin model snapshot.
    retrain = engine.get("retrain")
    if engine["type"] == "rule_based":
        if retrain != "never":
            raise PolicySpecError(
                f"{name}: rule_based exige engine.retrain=never (es {retrain!r})"
            )
        if engine.get("model_snapshot_id") or engine.get("model_snapshot_ids"):
            raise PolicySpecError(
                f"{name}: una política de reglas no tiene pesos entrenados — "
                "no puede declarar model_snapshot_id(s)"
            )
        if "train" in (spec.get("capabilities") or []):
            raise PolicySpecError(f"{name}: rule_based no puede declarar capability 'train'")
    elif retrain is None:
        raise PolicySpecError(f"{name}: engine.retrain es obligatorio (explícito, no implícito)")

    inputs = spec["inputs"]
    for field in ("feature_set_id", "resample_policy_id", "required_features"):
        if not inputs.get(field):
            raise PolicySpecError(f"{name}: inputs.{field} es obligatorio (§6: snapshot explícito)")
    if not isinstance(inputs["required_features"], list):
        raise PolicySpecError(f"{name}: inputs.required_features debe ser una lista")

    policy = spec["policy"]
    if policy.get("missing_input_policy") not in MISSING_INPUT_POLICIES:
        raise PolicySpecError(
            f"{name}: policy.missing_input_policy debe ser uno de {MISSING_INPUT_POLICIES}"
        )
    if policy.get("stale_input_policy") not in STALE_INPUT_POLICIES:
        raise PolicySpecError(
            f"{name}: policy.stale_input_policy debe ser uno de {STALE_INPUT_POLICIES}"
        )
    resolution = policy.get("resolution") or {}
    if "default_target_exposure" not in resolution:
        raise PolicySpecError(
            f"{name}: policy.resolution.default_target_exposure es obligatorio "
            "(invariante 9: sin default explícito no hay freeze)"
        )

    migration = (spec.get("migration") or {}).get("status")
    if migration not in MIGRATION_STATES:
        raise PolicySpecError(
            f"{name}: migration.status debe ser uno de {MIGRATION_STATES}, es {migration!r}"
        )

    # Freeze: el hash declarado debe COINCIDIR con el contenido económico.
    declared = (spec.get("governance") or {}).get("policy_hash")
    derived = canonical_policy_hash(spec)
    if declared and declared != derived:
        raise PolicySpecError(
            f"{name}: governance.policy_hash declarado ({declared}) no coincide con el "
            f"derivado del contenido ({derived}) — cambiar una regla/param exige nueva versión"
        )

    if impl["mode"] == "declarative":
        rules = policy.get("rules")
        if not rules:
            raise PolicySpecError(f"{name}: modo declarative exige policy.rules")
        # Construir la DeclarativePolicy valida el AST contra el whitelist
        # (operador desconocido / string crudo tipo "eval(...)" ⇒ ValueError).
        _build_declarative(spec)
    else:
        module_ref = impl.get("module")
        if not isinstance(module_ref, str) or ":" not in module_ref:
            raise PolicySpecError(
                f"{name}: coded_policy exige implementation.module 'paquete.modulo:Clase'"
            )
        if not module_ref.startswith(ALLOWED_MODULE_PREFIX):
            raise PolicySpecError(
                f"{name}: implementation.module debe vivir bajo {ALLOWED_MODULE_PREFIX!r} "
                "(un YAML jamás nombra un import arbitrario)"
            )


# ---------------------------------------------------------------------------
# Factory — ramifica SOLO por engine.type / implementation.mode
# ---------------------------------------------------------------------------

def _build_declarative(spec: Mapping[str, Any]) -> DeclarativePolicy:
    policy = spec["policy"]
    dsl_spec: dict[str, Any] = {
        "id": spec["id"],
        "version": str(spec["version"]),
        "policy_hash": canonical_policy_hash(spec),
        "resolution": dict(policy["resolution"]),
        "rules": [dict(r) for r in policy["rules"]],
    }
    pvid = (spec.get("governance") or {}).get("policy_version_id")
    if pvid:
        dsl_spec["policy_version_id"] = pvid
    return DeclarativePolicy(dsl_spec)


def _build_coded(spec: Mapping[str, Any]):
    module_ref = spec["engine"]["implementation"]["module"]
    module_name, _, class_name = module_ref.partition(":")
    if not module_name.startswith(ALLOWED_MODULE_PREFIX):
        raise PolicySpecError(
            f"módulo {module_name!r} fuera del allowlist {ALLOWED_MODULE_PREFIX!r}"
        )
    module = importlib.import_module(module_name)
    try:
        cls = getattr(module, class_name)
    except AttributeError as exc:
        raise PolicySpecError(f"{module_name} no expone {class_name!r}") from exc
    return cls(spec)


def build_policy(spec: Mapping[str, Any]):
    """Return a ``Policy`` for the spec. Branches on engine.type + mode ONLY."""
    validate_policy_spec(spec)
    status = spec["migration"]["status"]
    if status == "SPEC_ONLY":
        raise PolicySpecError(
            f"{spec['id']}: migration.status=SPEC_ONLY — el spec es documento de record, "
            "todavía no hay política ejecutable (fail-closed, BL-47)"
        )
    mode = spec["engine"]["implementation"]["mode"]
    if mode == "declarative":
        return _build_declarative(spec)
    return _build_coded(spec)
