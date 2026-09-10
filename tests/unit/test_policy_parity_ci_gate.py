"""CI must prove parity before a policy can become executable (C-010)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.validation import check_policy_parity as gate


REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "fabric-contracts.yml"


def _spec(policy_id: str, status: str) -> dict:
    return {"id": policy_id, "inputs": {}, "migration": {"status": status}}


def test_ci_zero_eligible_is_red_unless_the_caller_declares_it(monkeypatch, capsys):
    """Cero sujetos ya NO es verde por defecto.

    Este test exigía `== 0`. El contrato cambió en 2026-08-06 y con razón: el mensaje era
    honesto —decía que no había verificado nada— pero CI lee el EXIT CODE, no el texto, y
    `fabric-contracts.yml` tenía un paso corriendo, saliendo verde y sin comprobar una
    sola policy. Estado que lo producía: {PARITY_PENDING: 3, SPEC_ONLY: 1}.

    Ahora el vacío es rojo salvo que el llamador lo DECLARE con `--allow-empty`.
    """
    monkeypatch.setattr(
        gate,
        "load_all_policy_specs",
        lambda: [_spec("record_only", "SPEC_ONLY"), _spec("pending", "PARITY_PENDING")],
    )

    assert gate.main(["--ci-eligible"]) == 1
    assert "--allow-empty no fue declarado" in capsys.readouterr().out


def test_ci_zero_eligible_passes_only_when_explicitly_declared(monkeypatch, capsys):
    """Con la declaración explícita sí pasa, y el mensaje dice que el vacío es declarado.

    La diferencia con lo anterior no es el color: es que ahora alguien tuvo que escribirlo
    en el llamador, con motivo y condición de retiro a la vista.
    """
    monkeypatch.setattr(
        gate,
        "load_all_policy_specs",
        lambda: [_spec("record_only", "SPEC_ONLY"), _spec("pending", "PARITY_PENDING")],
    )

    assert gate.main(["--ci-eligible", "--allow-empty"]) == 0
    assert "VACÍO DECLARADO" in capsys.readouterr().out


def test_ci_empty_policy_registry_is_red(monkeypatch, capsys):
    """Cero por registro/directorio roto no es cero gobernado."""
    monkeypatch.setattr(gate, "load_all_policy_specs", lambda: [])

    assert gate.main(["--ci-eligible"]) == 1
    assert "registro de policies vacío" in capsys.readouterr().out


def test_ci_empty_harness_registry_is_red_even_without_eligible(monkeypatch, capsys):
    """Perder todos los arneses no puede pasar porque hoy todo esté pendiente."""
    monkeypatch.setattr(
        gate,
        "load_all_policy_specs",
        lambda: [_spec("record_only", "SPEC_ONLY"), _spec("pending", "PARITY_PENDING")],
    )
    monkeypatch.setattr(gate, "CHECKS", {})

    assert gate.main(["--ci-eligible"]) == 1
    assert "registro de arneses vacío" in capsys.readouterr().out


def test_ci_eligible_policy_without_harness_is_red(monkeypatch, capsys):
    monkeypatch.setattr(
        gate, "load_all_policy_specs", lambda: [_spec("unharnessed", "PARITY_GREEN")]
    )
    monkeypatch.setattr(gate, "CHECKS", {})

    assert gate.main(["--ci-eligible"]) == 1
    assert "sin arnés de paridad" in capsys.readouterr().out


def test_ci_eligible_policy_with_missing_frozen_data_is_red(monkeypatch, capsys):
    monkeypatch.setattr(
        gate, "load_all_policy_specs", lambda: [_spec("missing_data", "CUTOVER")]
    )

    def unavailable(_spec):
        raise gate.DataUnavailable("fixture congelada ausente")

    monkeypatch.setattr(gate, "CHECKS", {"missing_data": unavailable})
    monkeypatch.setattr(gate, "load_policy_spec", lambda _path: _spec("missing_data", "CUTOVER"))

    assert gate.main(["--ci-eligible"]) == 1
    output = capsys.readouterr().out
    assert "[FAIL] missing_data" in output
    assert "fixture congelada ausente" in output


def test_ci_eligible_policy_with_divergence_is_red(monkeypatch, capsys):
    monkeypatch.setattr(
        gate, "load_all_policy_specs", lambda: [_spec("divergent", "PARITY_GREEN")]
    )
    monkeypatch.setattr(
        gate,
        "CHECKS",
        {"divergent": lambda _spec: (np.array([0.0]), np.array([1.0]))},
    )
    monkeypatch.setattr(gate, "load_policy_spec", lambda _path: _spec("divergent", "PARITY_GREEN"))

    assert gate.main(["--ci-eligible"]) == 1
    assert "barras divergen" in capsys.readouterr().out


def test_fabric_workflow_invokes_strict_eligible_parity_gate():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert (
        "python scripts/validation/check_policy_parity.py --ci-eligible" in workflow
    ), "PARITY_GREEN must be mechanically checked by the fabric CI workflow"

    # ANTI-VACUIDAD DEL PROPIO TEST. Buscar sólo ese substring lo hacía inmune al cambio
    # de contrato: pasaba igual con o sin `--allow-empty`, es decir, no distinguía "el
    # gate verifica" de "el gate declara que no verifica". Se exige ahora que, si el
    # workflow declara el vacío, lo haga con su motivo y su condición de retiro escritos
    # al lado — mismo patrón que las cuarentenas de `ci.yml`.
    if "--allow-empty" in workflow:
        for marca in ("owner:", "cancelacion:", "PARITY_GREEN"):
            assert marca in workflow, (
                f"el workflow pasa `--allow-empty` sin declarar {marca!r}. Un vacío "
                f"aceptado sin motivo, dueño y condición de retiro es el mismo silencio "
                f"que este contrato vino a cerrar, sólo que con un flag delante"
            )
