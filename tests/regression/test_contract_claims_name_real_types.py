"""Un muro que nombra un tipo inexistente no es un muro: es una frase.

BL-15 declara como brecha viva que "no existe todavía un `book_construction` real que
consuma `strategy_output`". Al ir a cerrarla apareció algo más básico: **`strategy_output`
no existe en ninguna parte del código**. Es un nombre sin referente, afirmado en tiempo
presente en tres sitios como si fuera una barrera de tipos vigente:

    src/contracts/forecast_output.py:8
        "The allocator/book accepts exclusively ``strategy_output`` records validated by
         contract"
    src/orchestration/dataset_uri.py:68
        "forecast_output is DIAGNOSTIC and allocator accepts strategy_output only"
    usdcop-trading-dashboard/lib/contracts/forecast-output.contract.ts:8
        (espejo del anterior)

La mitad que SÍ existe funciona: `ForecastOutput` se rechaza por tipo. La otra mitad —el
canal accionable que supuestamente se acepta— **no tiene tipo al que apuntar**, y ningún
allocator valida nada por contrato: `scripts/analysis/book_construction.py` lee trades
crudos de un JSON.

Es la misma familia que K-049: una afirmación que se LEE como invariante aplicado y no
aplica nada. La diferencia con un test que miente es que aquí ni siquiera hay test — hay
prosa en un docstring, que es donde el siguiente ingeniero va a buscar la verdad.

Este candado exige que **todo tipo que un contrato declare como aceptado o rechazado
exista de verdad**. No opina sobre cuál debe ser el diseño: sólo prohíbe prometer una
barrera contra un nombre que no resuelve.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# Sitios donde un contrato afirma qué canal acepta o rechaza el allocator/libro.
CLAIM_SITES = (
    Path("src/contracts/forecast_output.py"),
    Path("src/orchestration/dataset_uri.py"),
    Path("usdcop-trading-dashboard/lib/contracts/forecast-output.contract.ts"),
)

# El candado se ancla al SUJETO de la afirmación arquitectónica —`allocator` o `book`—,
# no a cualquier "accepts" del fichero.  La primera versión de este regex no lo hacía y
# marcaba prosa inglesa corriente (`accepts compact`, `accepts them`) como si fueran tipos
# colgantes: un candado ruidoso se desactiva a la semana, así que se acota al reclamo real.
# El hueco admite saltos de línea y prefijos de comentario (` * `, `# `): la misma frase
# vive en un docstring Python de una línea y en un bloque JSDoc partido en dos.  Excluir
# `\n` dejaba escapar el espejo TypeScript, que tiene EXACTAMENTE el mismo defecto — un
# candado que sólo mira un lado del espejo es peor que ninguno, porque da por cubierto lo
# que no mira.
CLAIM_RE = re.compile(
    r"(?:allocator|book)[^.;]{0,80}?"
    r"accepts?\s+(?:exclusively\s+)?`{0,2}(?P<name>[A-Za-z_][A-Za-z0-9_]*)`{0,2}",
    re.IGNORECASE,
)

# Palabras que la gramática captura pero no son nombres de tipo.
NOT_A_TYPE = frozenset({"a", "an", "the", "no", "any", "it", "this", "records", "only"})


def _known_symbols() -> set[str]:
    """Identificadores realmente definidos en los contratos (Py y TS) y en el snake_case
    con que se nombran en prosa.  Se lee el TEXTO, no se importa: importar arrastraría
    dependencias de runtime y convertiría un candado de nombres en un test de entorno."""
    symbols: set[str] = set()
    for path in list((ROOT / "src" / "contracts").rglob("*.py")) + list(
        (ROOT / "usdcop-trading-dashboard" / "lib" / "contracts").rglob("*.ts")
    ):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in re.finditer(
            r"^\s*(?:class|def)\s+([A-Za-z_][A-Za-z0-9_]*)"
            r"|^\s*export\s+(?:interface|type|class|const)\s+([A-Za-z_][A-Za-z0-9_]*)",
            text,
            re.M,
        ):
            name = match.group(1) or match.group(2)
            if name:
                symbols.add(name)
                # `ForecastOutput` se nombra en prosa como `forecast_output`.
                symbols.add(re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower())
    return symbols


@pytest.mark.parametrize("relative", CLAIM_SITES, ids=lambda p: p.as_posix())
def test_contract_claims_name_a_type_that_exists(relative: Path) -> None:
    path = ROOT / relative
    assert path.is_file(), f"sitio de afirmación desaparecido: {relative.as_posix()}"

    known = _known_symbols()
    text = path.read_text(encoding="utf-8", errors="ignore")

    dangling: list[str] = []
    for match in CLAIM_RE.finditer(text):
        name = match.group("name") or match.group("rejected")
        if not name or name.lower() in NOT_A_TYPE:
            continue
        if name not in known:
            line = text[: match.start()].count("\n") + 1
            dangling.append(f"{relative.as_posix()}:{line}: '{name}' no resuelve")

    assert not dangling, (
        "un contrato promete una barrera contra un tipo que NO EXISTE:\n  "
        + "\n  ".join(dangling)
        + "\n\nUn nombre sin referente no rechaza nada. O el tipo se crea, o la frase "
        "deja de afirmarlo en presente. Ver el docstring de este módulo (BL-15)."
    )
