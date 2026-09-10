"""
Regression: un test escrito es un test que se ejecuta.

Contract: CTR-TEST-HYGIENE-001 · Date: 2026-08-24

## Dos veces en un día, el mismo modo de fallo silencioso

1. **`src/tests/`** contenía 33 tests que **nunca se ejecutaban**: `pyproject.toml` fija
   `testpaths = ["tests"]`, así que vivían fuera del alcance de la colección. Al moverlos a
   `tests/unit/` salió a la luz que uno llevaba tiempo en rojo — asertaba `cooldown=3` mientras
   el código devolvía `5`.

2. **`test_session_env_and_costs.py`** definía 17 funciones como `test9_...`, `test10_...`,
   `test13_...`. `python_functions = ["test_*"]` es un glob que exige el guion bajo: pytest
   recolectó **1 de 18** y reportó `1 passed` en verde.

En ninguno de los dos casos hubo error, aviso ni salida roja. Un test que no corre es peor que
no tener test: ocupa el sitio de la comprobación que sí habría corrido.

Este módulo hace ruidoso ese silencio.
"""

from __future__ import annotations

import ast
import fnmatch
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests"

# Los mismos patrones que declara `pyproject.toml`. Se leen de ahí, no se copian: un guard
# que duplica la config deja de proteger en cuanto la config cambia.
def _pytest_cfg() -> dict:
    tomllib = pytest.importorskip("tomllib")
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["tool"]["pytest"]["ini_options"]


CFG = _pytest_cfg()
FUNC_PATTERNS = CFG.get("python_functions", ["test*"])
FILE_PATTERNS = CFG.get("python_files", ["test_*.py"])
TESTPATHS = CFG.get("testpaths", ["tests"])


def _looks_like_a_test(name: str) -> bool:
    """Alguien que escribe esto pretendía que fuese un test."""
    return name.lower().startswith("test")


def _would_be_collected(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in FUNC_PATTERNS)


def _test_files() -> list[Path]:
    out = []
    for p in TESTS.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        if any(fnmatch.fnmatch(p.name, pat) for pat in FILE_PATTERNS):
            out.append(p)
    return sorted(out)


def test_no_function_looks_like_a_test_without_being_collected():
    """`test9_foo` parece un test y no lo es: `test_*` exige el guion bajo."""
    offenders: list[str] = []
    for path in _test_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:      # pragma: no cover - otro test lo cazará
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if _looks_like_a_test(node.name) and not _would_be_collected(node.name):
                    offenders.append(
                        f"{path.relative_to(ROOT).as_posix()}::{node.name}:{node.lineno}")
    assert not offenders, (
        f"funciones que parecen tests y NO se recolectan (patrones {FUNC_PATTERNS}):\n  "
        + "\n  ".join(offenders)
        + "\nPytest no avisa de esto: reporta verde sobre los que sí casan."
    )


# ---------------------------------------------------------------------------
# DEUDA CONGELADA: ficheros `test_*.py` que viven fuera de `testpaths` y por tanto
# NUNCA se ejecutan — ni en local ni en CI (verificado 2026-08-24: los workflows solo
# invocan `tests/unit`, `tests/integration` y `tests/regression`).
#
# El mas grave de la lista es `test_pretrade_gate.py`: cubre el `PreTradeGate`, que
# `rbac.md` regla 6 describe como "el ultimo gate antes del exchange" con semantica
# fail-safe (error => BLOCK). Su test existe y no corre.
#
# No se mueven aqui porque varios dependen de imports relativos a su servicio y moverlos
# sin correrlos primero cambiaria un problema silencioso por uno ruidoso sin arreglarlo.
# Se congelan para que la lista no CREZCA sin que nadie se entere.
# ---------------------------------------------------------------------------
UNCOLLECTED_DEBT = {
    "scripts/data/test_banrep_bop_scraper.py",
    "services/signalbridge_api/test_api.py",
    "services/signalbridge_api/tests/test_auth_flow.py",
    "services/signalbridge_api/tests/test_pretrade_gate.py",
    "services/signalbridge_api/tests/test_user_approval_flow.py",
    "src/strategies/spx500_regime_gated_v1/test_strategy.py",
    "usdcop-trading-dashboard/tests/test_interpretability_schema.py",
}


def test_no_NEW_test_file_lives_outside_the_collected_paths():
    """Un fichero `test_*.py` fuera de `testpaths` no se ejecuta jamás."""
    collected_roots = [ROOT / p for p in TESTPATHS]
    skip_dirs = {"__pycache__", "node_modules", ".git", "vendor", ".claude",
                 "build", "dist", ".next", "coverage"}
    orphans: list[str] = []
    for path in ROOT.rglob("test_*.py"):
        if any(part in skip_dirs for part in path.parts):
            continue
        if any(str(path).startswith(str(r)) for r in collected_roots):
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel not in UNCOLLECTED_DEBT:
            orphans.append(rel)
    assert not orphans, (
        f"ficheros de test fuera de testpaths={TESTPATHS} — NUNCA se ejecutan:\n  "
        + "\n  ".join(orphans)
        + "\nMuévelos bajo tests/ o dejarán de proteger nada. Precedente: `src/tests/` "
          "escondió 33 tests, uno de ellos en rojo."
    )


def test_every_test_file_has_at_least_one_collectable_function():
    """Un fichero de test sin tests recolectables es una comprobación que no existe."""
    empty: list[str] = []
    for path in _test_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:      # pragma: no cover
            continue
        has = any(
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and _would_be_collected(n.name)
            for n in ast.walk(tree)
        )
        if not has:
            empty.append(path.relative_to(ROOT).as_posix())
    assert not empty, (
        "ficheros de test sin ninguna función recolectable:\n  " + "\n  ".join(empty)
    )


def test_the_uncollected_debt_does_not_silently_shrink():
    """Si alguien arregla uno, que lo saque de la lista — o el guard miente."""
    fixed = sorted(d for d in UNCOLLECTED_DEBT if not (ROOT / d).is_file())
    assert not fixed, (
        f"estos ya no existen y siguen declarados como deuda: {fixed}. "
        "Bórralos de UNCOLLECTED_DEBT en el mismo commit que los movió."
    )


# ---------------------------------------------------------------------------
# DEUDA DECLARADA (2026-08-25): la suite COMPLETA no termina en Windows
# ---------------------------------------------------------------------------
#
# `python -m pytest tests/unit tests/integration tests/regression` muere con
# `Windows fatal exception: access violation` mientras importa
# `matplotlib.backends.backend_agg`, despues de que torch y tqdm ya esten cargados.
# Reproducido tambien con `tests/unit` A SOLAS, sin colectar un solo fichero de
# `tests/regression`: **no lo introduce el carril de investigacion**.
#
# Ademas hay **6 errores de coleccion preexistentes**, todos por colisiones de rutas del
# `sys.path` que arma `tests/conftest.py`:
#
#   features.calculators · services.dlq_service · utils.hash_utils · airflow.dags · regime
#   (+ un `TypeError: code() argument 13 must be str, not int`)
#
# El caso de `utils` esta diagnosticado: `conftest` inserta `airflow/dags` ANTES que `src`,
# asi que `utils` resuelve a `airflow/dags/utils/__init__.py`, que **lanza al importarse**
# si falta `POSTGRES_PASSWORD` en el entorno. El import fallido deja `utils` sin `__path__`
# y el mensaje que se ve es el desconcertante "'utils' is not a package".
#
# Consecuencia practica, y la razon de que esto este escrito aqui: **la suite completa NO
# se puede usar como prueba de "todo verde"**. Hay que correrla por tramos. Una corrida
# que se corta a los 7 minutos y reporta "N passed" esta reportando el tramo que le dio
# tiempo, no la suite.
#
# No se arregla en este commit: son problemas de infraestructura de test preexistentes y
# ajenos al alcance de la tesis. Se declaran para que nadie los descubra otra vez desde cero.

SUITE_KNOWN_ISSUES = {
    "windows_agg_access_violation": (
        "la suite completa muere importando matplotlib.backends.backend_agg tras torch/tqdm; "
        "reproducible con tests/unit a solas"
    ),
    "conftest_path_shadowing": (
        "conftest inserta airflow/dags antes que src: `utils`, `features`, `services`, "
        "`regime` resuelven al paquete equivocado (6 errores de coleccion)"
    ),
}


def test_the_suite_known_issues_are_documented_not_forgotten():
    """Este test no arregla nada: obliga a que la deuda siga escrita mientras exista.

    Si alguien arregla el `sys.path` de `conftest` o el crash de Agg, que borre la entrada
    en el mismo commit. Un README que nadie relee no cumple esa funcion; un test si.
    """
    assert set(SUITE_KNOWN_ISSUES) == {"windows_agg_access_violation",
                                       "conftest_path_shadowing"}
    for name, detail in SUITE_KNOWN_ISSUES.items():
        assert len(detail) > 60, f"{name}: la deuda tiene que ser accionable, no un titular"
