# -*- coding: utf-8 -*-
"""Un gate sin sujeto no prueba nada, y su verde tampoco.

EL DEFECTO
----------
`check_policy_parity.py --ci-eligible` sólo verifica policies en `PARITY_GREEN`/`CUTOVER`.
El 2026-08-06 no había ninguna elegible y el script imprimía «0 specs elegibles — nada
verificado» y devolvía **0**. El conteo exacto de policies inertes no se anota: lo mide el
propio gate en cada corrida, y una cifra escrita a mano sólo puede quedarse obsoleta.

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

import numpy as np
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
    ruta = REPO / ".github/workflows/fabric-contracts.yml"
    doc = yaml.safe_load(ruta.read_text(encoding="utf-8"))
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


def _spec_elegible() -> dict:
    return {"id": "con_arnes", "inputs": {"warmup_bars": 0}}


def test_with_a_subject_the_harness_actually_runs(monkeypatch, capsys) -> None:
    """ANTI-VACUIDAD CENTRAL: con sujeto elegible, la verificación se EJECUTA.

    Sin esto, todo este fichero mediría sólo cómo se comporta el gate cuando no hay nada
    que mirar. Una versión anterior usaba una policy elegible **sin arnés** y exigía
    `exit 1`; Codex señaló (CXD-757) que eso sólo prueba un camino de fallo y no demuestra
    que `--allow-empty` deje correr la verificación real. Tenía razón.

    Aquí el arnés es un espía que devuelve dos arrays IDÉNTICOS: el gate debe llamarlo
    exactamente una vez y salir 0 **aunque se le pase `--allow-empty`**, porque el flag
    perdona el vacío, no la verificación.
    """
    llamadas: list[dict] = []

    def arnes(spec):
        llamadas.append(spec)
        return np.zeros(8, dtype=float), np.zeros(8, dtype=float)

    monkeypatch.setattr(
        gate, "load_all_policy_specs", lambda: [_policy("con_arnes", "PARITY_GREEN")]
    )
    monkeypatch.setattr(gate, "CHECKS", {"con_arnes": arnes})
    monkeypatch.setattr(gate, "load_policy_spec", lambda _p: _spec_elegible())

    assert gate.main(["--ci-eligible", "--allow-empty"]) == 0
    assert len(llamadas) == 1, (
        f"el arnés se ejecutó {len(llamadas)} veces: con un sujeto elegible el gate tiene "
        f"que verificar exactamente una vez, no saltárselo"
    )
    salida = capsys.readouterr().out
    assert "exposición IDÉNTICA" in salida
    assert "VACÍO DECLARADO" not in salida, (
        "con sujeto el gate no puede reportar vacío: confundiría «no había nada que "
        "verificar» con «verifiqué y coincide»"
    )


def test_the_flag_does_not_silence_a_real_divergence(monkeypatch, capsys) -> None:
    """Y con divergencia real, `--allow-empty` NO la perdona.

    El complemento del anterior: si el flag se hubiera implementado como «pasa siempre»,
    aquel daría 0 igualmente y no distinguiríamos nada. Aquí el arnés devuelve arrays que
    difieren y el gate debe salir 1 pese al flag.
    """
    def arnes(spec):
        return np.zeros(8, dtype=float), np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=float)

    monkeypatch.setattr(
        gate, "load_all_policy_specs", lambda: [_policy("con_arnes", "PARITY_GREEN")]
    )
    monkeypatch.setattr(gate, "CHECKS", {"con_arnes": arnes})
    monkeypatch.setattr(gate, "load_policy_spec", lambda _p: _spec_elegible())

    assert gate.main(["--ci-eligible", "--allow-empty"]) == 1
    assert "barras divergen" in capsys.readouterr().out


def test_allow_empty_is_rejected_without_ci_eligible() -> None:
    """El flag sólo tiene sentido frente al conjunto elegible (recomendación CXD-757).

    Suelto invitaría a colarlo en cualquier invocación como si ablandase el gate entero.
    """
    with pytest.raises(SystemExit) as e:
        gate.main(["--allow-empty"])
    assert e.value.code != 0


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
