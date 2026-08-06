# -*- coding: utf-8 -*-
"""Un gate sin sujeto no prueba nada, y su verde tampoco.

EL DEFECTO
----------
`check_policy_parity.py --ci-eligible` sólo verifica policies en `PARITY_GREEN`/`CUTOVER`.
El 2026-08-06 no había ninguna —estado real `{PARITY_PENDING: 3, SPEC_ONLY: 1}`, cero
elegibles— y el script imprimía «0 specs elegibles — nada verificado» y devolvía **0**.

`fabric-contracts.yml` lo ejecuta. O sea: un paso de CI que corría, salía verde y no
comprobaba una sola policy, indefinidamente, hasta que alguien promoviese algo.

Lo justo con quien lo escribió: el mensaje **declaraba** su propia vacuidad, que es más de
lo que hace la mayoría. El problema es que CI lee el código de salida, no el texto.

QUÉ FIJA ESTE FICHERO
---------------------
Los tres casos de la CLI, incluido el que impide que este mismo test sea vacuo: que **con
sujeto elegible el gate verifique de verdad**. Sin ese tercero, todo lo de aquí probaría
únicamente cómo se comporta el gate cuando no hay nada que mirar.

LO QUE NO SE HIZO, A PROPÓSITO
-----------------------------
No se promovió ninguna policy para darle sujeto al gate. Eso habría puesto el número en
verde sin que nada mejorase: exactamente la trampa que este cambio denuncia.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "_policy_parity_gate", REPO / "scripts/validation/check_policy_parity.py"
)
gate = importlib.util.module_from_spec(_spec)
# Registrar en `sys.modules` ANTES de ejecutar: sin esto, un `@dataclass` dentro del
# módulo revienta con `AttributeError: 'NoneType' object has no attribute '__dict__'`.
# Medido al tropezar con ello dos veces en esta sesión.
sys.modules["_policy_parity_gate"] = gate
_spec.loader.exec_module(gate)


def _comando_del_gate() -> str:
    """El `run:` REAL del step de paridad, no el texto del fichero.

    Buscar `"--allow-empty" in workflow` era un falso candado: mis propios comentarios
    mencionan el flag, así que el substring casaba aunque se hubiera borrado del comando.
    Medido con un mutante que quitó el flag del `run:` y dejó 13 verdes.
    Se parsea el YAML y se lee el step por su nombre.
    """
    doc = yaml.safe_load((REPO / ".github/workflows/fabric-contracts.yml").read_text(encoding="utf-8"))
    for job in (doc.get("jobs") or {}).values():
        for step in job.get("steps") or []:
            run = str(step.get("run", ""))
            if "check_policy_parity.py" in run:
                return run
    return ""


def _policy(pid: str, status: str) -> dict:
    return {"id": pid, "migration": {"status": status}}


INERTES = [_policy("record_only", "SPEC_ONLY"), _policy("pending", "PARITY_PENDING")]


def test_zero_eligible_without_the_flag_is_red(monkeypatch, capsys) -> None:
    """Sin declaración explícita, cero sujetos es ROJO.

    Rojo→verde con: devolver el `return 0` incondicional de antes.
    """
    monkeypatch.setattr(gate, "load_all_policy_specs", lambda: INERTES)
    assert gate.main(["--ci-eligible"]) == 1
    assert "--allow-empty no fue declarado" in capsys.readouterr().out


def test_zero_eligible_with_the_flag_is_green_but_says_so(monkeypatch, capsys) -> None:
    """Con `--allow-empty` pasa, y el mensaje deja constancia de que el vacío es declarado.

    Importa el texto además del código: quien lea el log de CI tiene que poder distinguir
    «verifiqué y está bien» de «no había nada que verificar y alguien lo aceptó».
    """
    monkeypatch.setattr(gate, "load_all_policy_specs", lambda: INERTES)
    assert gate.main(["--ci-eligible", "--allow-empty"]) == 0
    assert "VACÍO DECLARADO" in capsys.readouterr().out


def test_the_flag_does_not_silence_a_real_failure(monkeypatch, capsys) -> None:
    """ANTI-VACUIDAD: `--allow-empty` sólo perdona el vacío, no un fallo con sujeto.

    Es el test que impide que este fichero mida nada. Si el flag se hubiera implementado
    como «pasa siempre», los dos de arriba seguirían verdes y el gate estaría muerto —que
    es el defecto original con otra ropa.

    Se usa una policy elegible SIN arnés de paridad: el gate debe seguir dando 1 aunque se
    le pase el flag.
    """
    monkeypatch.setattr(
        gate, "load_all_policy_specs", lambda: [_policy("sin_arnes", "PARITY_GREEN")]
    )
    assert gate.main(["--ci-eligible", "--allow-empty"]) == 1
    salida = capsys.readouterr().out
    assert "VACÍO DECLARADO" not in salida, (
        "con un sujeto elegible el gate no puede reportar vacío: estaría confundiendo "
        "«no hay nada que verificar» con «hay algo y falla»"
    )


def test_the_workflow_declares_why_it_accepts_an_empty_gate() -> None:
    """El llamador declara el vacío con motivo, dueño y condición de retiro.

    Un flag suelto en un YAML es indistinguible de un ablandamiento silencioso. Se exige
    la misma disciplina que a las cuarentenas de `ci.yml`: si hoy no hay sujeto, que se
    lea por qué y qué lo cancela.
    """
    comando = _comando_del_gate()
    assert "--ci-eligible" in comando, "el gate de paridad desapareció del workflow"
    if "--allow-empty" not in comando:
        pytest.skip("el workflow ya no declara vacío: hay sujeto elegible, nada que exigir")
    workflow = (REPO / ".github/workflows/fabric-contracts.yml").read_text(encoding="utf-8")
    for marca in ("owner:", "cancelacion:", "PARITY_GREEN"):
        assert marca in workflow, f"`--allow-empty` sin declarar {marca!r}"


def test_no_policy_was_promoted_to_feed_the_gate() -> None:
    """El candado contra la salida fácil.

    La forma tentadora de «arreglar» esto era promover una policy a `PARITY_GREEN` para
    que el gate tuviera sujeto. Habría puesto el verde sin que nada mejorase. Mientras las
    policies sigan inertes, el workflow debe seguir declarando el vacío; el día que una se
    promueva de verdad, este test recuerda que hay que retirar el flag.
    """
    import yaml

    estados = {
        p.stem: (yaml.safe_load(p.read_text(encoding="utf-8")).get("migration") or {}).get("status")
        for p in sorted((REPO / "config/policies").glob("*.yaml"))
    }
    elegibles = {k: v for k, v in estados.items() if v in {"PARITY_GREEN", "CUTOVER"}}
    comando = _comando_del_gate()
    if elegibles:
        assert "--allow-empty" not in comando, (
            f"ya hay policies elegibles ({sorted(elegibles)}) y el workflow sigue "
            f"declarando vacío: retirar `--allow-empty`, su condición de cancelación se "
            f"cumplió"
        )
    else:
        assert "--allow-empty" in comando, (
            f"no hay policies elegibles (estados: {estados}) y el workflow no declara el "
            f"vacío: el gate correría en rojo permanente sin decir por qué"
        )
