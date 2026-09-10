"""
Regression: los PNG que sirve el dashboard son reproducibles entre versiones.

Contexto (auditoria de limpieza 2026-08-24). `CLAUDE.md` obliga a versionar
`usdcop-trading-dashboard/public/forecasting/**` para que un clon limpio renderice
todas las paginas. Medido sobre la historia real: 2.788 blobs PNG unicos frente a
1.492 en HEAD; **305,5 MB del pack de 811 MB (38%) son estos graficos**.

Eso solo es sostenible porque matplotlib NO incrusta timestamp en el PNG: mismo
grafico -> mismos bytes -> git deduplica y el peso crece unicamente cuando el
contenido cambia de verdad.

La excepcion era un chunk `tEXt`:

    Software  Matplotlib version3.11.1, https://matplotlib.org/

Con el, un `pip install -U matplotlib` reescribe los 1.492 ficheros de golpe (~163 MB
en un commit) sin que ningun grafico haya cambiado. `src/utils/plot_determinism.py`
lo omite; este guard evita que el cableado se caiga en un refactor.

NOTA sobre los PNG ya versionados: los que estan en HEAD se generaron ANTES del fix y
todavia llevan el chunk. No se reescriben a proposito — hacerlo costaria exactamente
el commit de 163 MB que este cambio existe para evitar. Se limpiaran solos, fichero a
fichero, segun los DAGs los vayan regenerando por cambios reales de contenido.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "src" / "utils" / "plot_determinism.py"

# Generadores que escriben PNG servidos por el dashboard. Un savefig aqui acaba en
# `public/`, asi que todos deben activar el modo determinista.
DASHBOARD_PLOT_GENERATORS = (
    ROOT / "scripts" / "pipeline" / "generate_weekly_forecasts.py",
    ROOT / "scripts" / "pipeline" / "generate_asset_forward_charts.py",
    ROOT / "scripts" / "pipeline" / "generate_usdcop_directional_replay.py",
)

# EXCEPCION DE GOBIERNO, no un olvido.
#
# `scripts/pipeline/train_and_export_smart_simple.py` tambien guarda PNG del dashboard,
# pero esta dentro del MURO DE CONGELACION de tres manifiestos de estrategia
# (`config/strategy_manifests/{usdcop,usdcop_v12,usdcop_v14}.yaml`), que lo declaran en
# `spec_fingerprint_inputs` y `code_reference`. Cualquier edicion — incluida una linea de
# fontaneria como esta — cambia su `code_hash_sha256_16` y tumba
# `test_strategy_manifests.py::test_code_hash_detects_strategy_drift` en los tres.
#
# El repo ya declara esta situacion como gobierno de modelado, no de fontaneria: ver el
# xfail BLOCKED_OPERATOR_DECISION en `test_approval_mutual_exclusion.py`, que dice
# literalmente que re-congelar "requiere aprobacion explicita del operador".
#
# Durante la auditoria de limpieza 2026-08-24 se cableo por error y se REVIRTIO al
# detectarlo (el fichero volvio a ser byte-identico a HEAD). Consecuencia asumida: los PNG
# que emite ese script siguen llevando el chunk `Software`, asi que una subida de matplotlib
# los reescribira. Es un lote pequeno comparado con los 1.492 de `public/forecasting/`.
#
# PARA CERRARLO: el operador aprueba, y en el MISMO commit se cablea
# `enable_deterministic_png()`, se re-congelan los 3 manifiestos (bump de version, hash
# nuevo, `refreeze_note`) y se mueve esta ruta a DASHBOARD_PLOT_GENERATORS.
FROZEN_PENDING_OPERATOR_APPROVAL = (
    ROOT / "scripts" / "pipeline" / "train_and_export_smart_simple.py",
)


def test_helper_module_exists():
    assert HELPER.is_file(), (
        f"falta {HELPER.relative_to(ROOT).as_posix()} — sin el, subir matplotlib "
        "reescribe todos los PNG versionados."
    )


@pytest.mark.parametrize("script", DASHBOARD_PLOT_GENERATORS, ids=lambda p: p.name)
def test_dashboard_generators_enable_deterministic_png(script: Path):
    assert script.is_file(), f"{script.relative_to(ROOT).as_posix()} no existe"
    text = script.read_text(encoding="utf-8")
    assert "enable_deterministic_png()" in text, (
        f"{script.name} guarda PNG que sirve el dashboard pero no llama a "
        "`enable_deterministic_png()`. Anadelo justo despues de "
        '`matplotlib.use("Agg")`.'
    )


@pytest.mark.parametrize("script", DASHBOARD_PLOT_GENERATORS, ids=lambda p: p.name)
def test_enable_is_called_before_any_savefig(script: Path):
    """Activarlo despues del primer `savefig` no sirve de nada."""
    text = script.read_text(encoding="utf-8")
    if "savefig" not in text:  # pragma: no cover
        pytest.skip(f"{script.name} ya no guarda figuras")
    enable_at = text.index("enable_deterministic_png()")
    first_savefig = text.index("savefig")
    assert enable_at < first_savefig, (
        f"{script.name} llama a `enable_deterministic_png()` DESPUES del primer "
        "`savefig`; los graficos guardados antes seguirian llevando el chunk."
    )


def test_helper_omits_software_and_leaves_other_metadata_alone():
    """El parche debe OMITIR (None), no escribir cadena vacia, y no pisar al llamador."""
    tree = ast.parse(HELPER.read_text(encoding="utf-8"))
    src = HELPER.read_text(encoding="utf-8")
    assert '"Software": None' in src, (
        "el chunk debe omitirse con None; la cadena vacia escribe un tEXt vacio y "
        "sigue siendo un byte de diferencia entre versiones."
    )
    assert "merged.update(kwargs.get(\"metadata\") or {})" in src, (
        "un llamador que pase su propio `metadata=` debe seguir mandando."
    )
    assert isinstance(tree, ast.Module)


def test_helper_roundtrip_produces_identical_bytes():
    """Dos guardados del mismo grafico deben ser identicos byte a byte."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import sys
    import tempfile

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.utils.plot_determinism import (
        enable_deterministic_png,
        png_has_software_chunk,
    )

    import matplotlib.pyplot as plt

    enable_deterministic_png()
    fig = plt.figure()
    plt.plot([1, 2, 3])
    try:
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a.png", Path(d) / "b.png"
            fig.savefig(a)
            fig.savefig(b)
            assert not png_has_software_chunk(a), "el chunk `Software` sigue presente"
            assert a.read_bytes() == b.read_bytes(), (
                "dos guardados del mismo grafico difieren: el PNG no es determinista "
                "y git creara un blob nuevo en cada regeneracion."
            )
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "script", FROZEN_PENDING_OPERATOR_APPROVAL, ids=lambda p: p.name
)
def test_frozen_generator_stays_untouched_until_approved(script: Path):
    """El fichero congelado NO debe llevar el cableado sin re-congelar los manifiestos.

    Este guard corre en la direccion contraria a los de arriba: falla si alguien anade
    `enable_deterministic_png()` aqui sin hacer el re-freeze en el mismo commit. Sin el,
    la correccion "obvia" de un futuro lector rompe tres manifiestos en silencio.
    """
    if not script.is_file():  # pragma: no cover
        pytest.skip(f"{script.name} ya no existe")
    text = script.read_text(encoding="utf-8")
    manifests = ROOT / "config" / "strategy_manifests"
    still_frozen = [
        m.name
        for m in sorted(manifests.glob("usdcop*.yaml"))
        if script.relative_to(ROOT).as_posix() in m.read_text(encoding="utf-8")
    ]
    if "enable_deterministic_png()" in text:
        assert not still_frozen, (
            f"{script.name} lleva el cableado de PNG deterministas pero sigue congelado "
            f"en {still_frozen}. Cambiarlo altera su code_hash y tumba "
            "test_strategy_manifests.py. Re-congela los manifiestos (bump de version, "
            "hash nuevo, refreeze_note) en el MISMO commit, y mueve la ruta a "
            "DASHBOARD_PLOT_GENERATORS."
        )
    else:
        assert still_frozen, (
            f"{script.name} ya no aparece en ningun manifiesto congelado: la razon para "
            "eximirlo desaparecio. Cablea `enable_deterministic_png()` y muevelo a "
            "DASHBOARD_PLOT_GENERATORS."
        )


# ---------------------------------------------------------------------------
# El parche no puede romper el import de pyplot (2026-08-25)
# ---------------------------------------------------------------------------

def test_patch_preserves_qualname_so_pyplot_can_still_import():
    """Sin `functools.wraps`, aplicar el parche ANTES de importar pyplot mata el proceso.

    `matplotlib.pyplot`, al importarse, construye `plt.savefig` desde `Figure.savefig` y
    valida su `__qualname__` en `_add_pyplot_note`. Un qualname que no empiece por `Figure.`
    levanta `RuntimeError: Wrapped method from unexpected class`.

    El modo de fallo es desproporcionado y despistante: **pyplot deja de importarse**, así que
    revientan por ImportError módulos que no dibujan nada y que no tienen relación aparente
    con los gráficos. Ocurrió: ocho errores de colección en `tests/unit`, dos de ellos por
    esto, en ficheros llamados `test_zoo_generator_contract` y `test_forecast_regime_gate`.
    """
    import subprocess
    import sys as _sys

    code = (
        "import sys; sys.path.insert(0, r'%s');"
        "from src.utils.plot_determinism import enable_deterministic_png;"
        "assert enable_deterministic_png() is True;"
        "import matplotlib; matplotlib.use('Agg');"
        "import matplotlib.pyplot as plt;"          # el import que fallaba
        "from matplotlib.figure import Figure;"
        "assert Figure.savefig.__qualname__.startswith('Figure.'), Figure.savefig.__qualname__;"
        "print('OK')"
    ) % str(ROOT)
    proc = subprocess.run([_sys.executable, "-c", code], capture_output=True, text=True,
                          timeout=180)
    assert proc.returncode == 0, (
        "parchear antes de importar pyplot rompe el import:\n" + proc.stderr[-1200:]
    )
    assert "OK" in proc.stdout


def test_patch_is_idempotent():
    """Dos llamadas no pueden envolver el wrapper: cada capa añade una indirección y la
    segunda vuelve a romper el `__qualname__`."""
    from src.utils.plot_determinism import enable_deterministic_png

    enable_deterministic_png()
    assert enable_deterministic_png() is False, "el parche se aplicó dos veces"
