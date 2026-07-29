#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_mutation_review.py — corredor de mutaciones del lote CLAUDE (conveniencia para el revisor)
==============================================================================================

ESTE SCRIPT ES UNA CONVENIENCIA, NO UNA AUTORIDAD.
--------------------------------------------------
Lo escribió CLAUDE, que es la parte REVISADA. Un revisor que dependa de una herramienta
escrita por el revisado no está revisando: está leyendo un resumen que le entrega el
examinado. Por eso el diseño de este fichero está deliberadamente limitado:

  * NO emite veredictos. Imprime el resultado CRUDO (conteos + mensaje del test) y si
    coincide o no con lo declarado. La palabra "APROBADO" no aparece en la salida.
  * NO aporta un solo dato que no esté ya en el dossier. Todo lo que hace este script
    —fichero, línea, cadena vieja, cadena nueva, comando, verde esperado, rojo esperado—
    está escrito literal en `.claude/coordination/integration/REVIEW-DOSSIER-CLAUDE.md`.
    Si no te fías del script, EL CAMINO ES EL DOSSIER: aplicar la mutación a mano, correr
    el comando, y `git checkout -- <fichero>`. Ese camino es el canónico; éste solo ahorra
    tecleo.
  * IMPRIME EL DIFF EXACTO antes de aplicar nada. Si el diff que ves no es la mutación que
    el dossier declara, el script está mintiendo y lo verás en pantalla.
  * VERIFICA LA RESTAURACIÓN POR HASH (sha256 antes / después) y grita si no coincide.
    `try/finally`: el fichero se restaura aunque el test reviente o interrumpas con Ctrl-C.
  * SE NIEGA A CORRER sobre un árbol sucio en los ficheros que va a mutar. Mutar encima de
    WIP produce un rojo que no corresponde a NINGÚN commit y por tanto no es evidencia.

Lo que este script NO puede darte, y que sí es revisión de verdad: leer el test y decidir
si la aserción que se pone roja es la garantía que el BL promete, o solamente *una* que
se rompe de camino. Eso no lo automatiza nadie.

Uso
---
    python scripts/validation/run_mutation_review.py --list
    python scripts/validation/run_mutation_review.py --bl BL-09
    python scripts/validation/run_mutation_review.py            # todos

Exit codes: 0 = se ejecutó todo (independientemente de coincidencias); 1 = abortó
(árbol sucio, cadena no encontrada, restauración corrupta). Un exit 0 NO significa
"BLs aprobados": significa "el corredor hizo su trabajo".
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DASHBOARD = "usdcop-trading-dashboard"
TIMEOUT_S = 1800

# Windows: la consola por defecto es cp1252 y este script imprime σ, é, ─. Sin esto, un
# UnicodeEncodeError abortaría a mitad de una corrida (y con el repo mutado si cayera
# dentro del try). Se fuerza UTF-8 en la salida.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):  # pragma: no cover
        pass


# ─────────────────────────────────────────────────────────────────────────────────────
# Registro de mutaciones.
#
# Cada entrada es una TRANSCRIPCIÓN del bloque `### Verificación ejecutable` del MD del BL
# (`.claude/specs/planes/backlog/BL-NN-*.md`) o del mensaje de commit citado. `expected_*`
# es lo DECLARADO por quien cerró el BL, no lo medido por este script.
#
# `expected_failed = None`  =>  el conteo del rojo NO está registrado en ninguna fuente.
#                               El corredor lo MIDE y lo imprime, pero no puede comparar:
#                               la columna `coincide` sale `n/a`.
# ─────────────────────────────────────────────────────────────────────────────────────
@dataclass
class Mutation:
    bl: str
    commit: str
    cwd: str                      # relativo al repo
    command: str                  # se ejecuta con shell=True, tal cual
    green: str                    # verde declarado en el MD
    path: str                     # relativo al repo
    old: str
    new: str
    expected_text: str            # rojo declarado, literal
    expected_failed: int | None
    expected_passed: int | None = None
    note: str = ""
    _extra: dict = field(default_factory=dict)


MUTATIONS: list[Mutation] = [
    Mutation(
        bl="BL-05",
        commit="a6c83a4f",
        cwd=DASHBOARD,
        command=(
            "npx vitest run tests/unit/components/ProductionView.paper-ledger.test.tsx "
            "tests/unit/components/PaperCandidatesPanel.test.tsx"
        ),
        green="16 passed",
        path="usdcop-trading-dashboard/components/gm/views/ProductionView.tsx",
        old="{paperLedger && <PaperCandidatesPanel ledger={paperLedger} />}",
        new="{false && paperLedger && <PaperCandidatesPanel ledger={paperLedger} />}",
        expected_text='1 failed — "Unable to find role=table and name /candidatas/i"',
        expected_failed=1,
        note="muta-2 del MD (celda Sharpe con n_trades=11) queda MANUAL: es código nuevo, no una sustitución.",
    ),
    Mutation(
        bl="BL-09",
        commit="cb1241b2",
        cwd=".",
        command="python -m pytest tests/regression/test_trial_ledger.py -q",
        green="26 passed",
        path="scripts/validation/check_trial_ledger.py",
        old="    errors += check_schema_and_ids(records)\n",
        new="    errors += check_schema_and_ids(records)\n    return errors\n",
        expected_text="10 failed — un caso por cada check desconectado del gate",
        expected_failed=10,
        note="`return errors` insertado en run_all_checks() tras el primer check.",
    ),
    Mutation(
        bl="BL-11",
        commit="cb1241b2",
        cwd=".",
        command="python -m pytest tests/regression/test_trial_ledger.py -q",
        green="26 passed",
        path="scripts/validation/check_trial_ledger.py",
        old=(
            "def check_families(records: list[dict], families_dir: Path = FAMILIES_DIR)"
            " -> list[str]:\n    errors = []"
        ),
        new=(
            "def check_families(records: list[dict], families_dir: Path = FAMILIES_DIR)"
            " -> list[str]:\n    return []\n    errors = []"
        ),
        expected_text="2 failed — celda a trial inexistente + trials_charged que subcuenta",
        expected_failed=2,
    ),
    Mutation(
        bl="BL-13",
        commit="0645dcd1",
        cwd=".",
        command=(
            "python -m pytest tests/regression/test_strategy_manifests.py "
            "tests/regression/test_feature_contracts.py -q"
        ),
        green="49 passed, 1 xfailed  (DECLARADO; hoy son 50 passed — ver dossier)",
        path="src/identity/source_hash.py",
        old='    return data.replace(b"\\r\\n", b"\\n")',
        new="    return data",
        expected_text="8 failed, 41 passed — dos ficheros de test caen desde UNA línea de producción",
        expected_failed=8,
        expected_passed=41,
        note="El xfail(strict) que hacía el verde 49+1 se retiró en 8005ffea; el verde de hoy es 50 passed.",
    ),
    Mutation(
        bl="BL-14",
        commit="0645dcd1",
        cwd=".",
        command="python -m pytest tests/regression/test_strategy_manifests.py -q",
        green="23 passed, 1 xfailed  (DECLARADO; hoy son 24 passed — ver dossier)",
        path="config/strategy_manifests/usdcop.yaml",
        old=(
            "    pointer: outputs/forecasting/h5_weekly_models/latest/\n"
            "    registered_in: MLflow run de forecast_h5_l3_weekly_training (tags git_commit,\n"
            "      iso_week, contract FC-H5-L3-001)\n"
            "    as_of: '2026-07-06'     # ultima corrida L3 local al sellar este re-freeze\n"
            "    artifacts_sha256_16:\n"
            "      ridge_h5.pkl: 8e4618c4d26d3af9\n"
            "      bayesian_ridge_h5.pkl: 0c206dc55908c71a\n"
            "      scaler_h5.pkl: 3302221e2b9dee39\n"
            "      feature_cols_h5.json: c3393242ef998896\n"
        ),
        new=(
            "    pointer: no/existe/\n"
            "    registered_in: ninguna parte\n"
            "    as_of: '1999-01-01'\n"
            "    artifacts_sha256_16:\n"
            "      ridge_h5.pkl: '0000000000000000'\n"
            "      bayesian_ridge_h5.pkl: '0000000000000000'\n"
            "      scaler_h5.pkl: '0000000000000000'\n"
            "      feature_cols_h5.json: '0000000000000000'\n"
        ),
        expected_text="<sin registrar> — el MD dice literalmente 'conteo exacto de failed sin registrar'",
        expected_failed=None,
        note="current_model_snapshot INVENTADO (las 4 cláusulas a la vez).",
    ),
    Mutation(
        bl="BL-20",
        commit="955374d0 + correccion 2026-07-28 (hash del pack nuevo pendiente)",
        cwd=".",
        command="python -m pytest tests/unit/test_interpretability_artifacts.py -q",
        green="21 passed",
        path="scripts/analysis/generate_interpretability.py",
        # La línea histórica `phi = Z * coefs` (fit único global) YA NO EXISTE: la
        # corrección del 2026-07-28 llevó la ruta lineal al mismo walk-forward expanding
        # anual que la de árbol y la atribución vive ahora en `_linear_contributions`.
        # La mutación es la misma idea —FABRICAR las contribuciones— sobre la línea vigente.
        old="    return Zi * coefs, intercept, coefs",
        new="    return np.ones_like(Zi), intercept, coefs",
        expected_text="3 failed, 18 passed — aditividad, no-degeneración y acoplamiento al modelo",
        expected_failed=3,
        expected_passed=18,
        note=(
            "Dos mutaciones quedan MANUALES (otras líneas del mismo fichero, ver dossier): "
            "(a) rama TreeSHAP `phi = np.ones_like(phi)`; (b) candado test-folds/provenance — "
            "`train = prev.iloc[:-HORIZON]` -> `train = prev` quita la purga y deja 1 failed, "
            "20 passed en test_linear_attribution_rows_are_test_folds_never_train_rows."
        ),
    ),
    Mutation(
        bl="BL-25",
        commit="955374d0",
        cwd=".",
        command="python -m pytest tests/unit/test_system_health.py -q",
        green="25 passed",
        path="src/monitoring/system_health.py",
        old="                if te_z > TRACKING_ERROR_SIGMA:",
        new="                if te_z > TRACKING_ERROR_SIGMA * 1000:",
        expected_text="1 failed — el gemelo de 3.50σ deja de disparar withdrawal (GREEN != ORANGE)",
        expected_failed=1,
        note="muta-2 (/1000) y muta-3 (ruido->0) quedan MANUALES; ver dossier.",
    ),
    Mutation(
        bl="BL-31",
        commit="014687cc",
        cwd=".",
        command="python -m pytest tests/regression/test_strangler_cop.py -q",
        green="41 passed",
        path="src/strangler/parity.py",
        old="        if obs.verdict is ParityVerdict.MATCH:",
        new="        if obs.verdict is not ParityVerdict.MISMATCH:",
        expected_text="2 failed — una INVALID deja de romper la racha => PARITY_GREEN sin comparar nada",
        expected_failed=2,
    ),
    Mutation(
        bl="BL-32",
        commit="2a608feb / 225e3524",
        cwd=".",
        command="python -m pytest tests/unit/test_passport_contract.py -q",
        green="83 passed",
        path="src/contracts/passport.py",
        old="    for block, fields in PASSPORT_BLOCK_FIELDS.items():",
        new="    for block, fields in {}.items():",
        expected_text="8 failed, 75 passed",
        expected_failed=8,
        expected_passed=75,
        note=(
            "MISMO candado que el endurecimiento de los dos validadores de passport (225e3524): "
            "el rojo declarado allí es exactamente éste. El lado TS no tiene rojo registrado."
        ),
    ),
    Mutation(
        bl="BL-36",
        commit="2a608feb",
        cwd=".",
        command="python -m pytest tests/regression/test_db_truth_matrix.py -q",
        green="8 passed",
        path=".claude/specs/platform/db-truth-matrix.md",
        old=(
            "| BI `fact_*` | DEPRECATED | E1+E2: cableadas pero (según perfil) 0 filas |"
            " 7 ficheros referencian `bi.fact_forecasts` |"
        ),
        new=(
            "| BI `fact_*` | AUTORITATIVA (escritor único) | E1+E2: cableadas pero (según perfil)"
            " 0 filas | 0 ficheros referencian `bi.fact_forecasts` |"
        ),
        expected_text="<sin registrar> — el MD dice literalmente 'conteo exacto de failed sin registrar'",
        expected_failed=None,
    ),
    Mutation(
        bl="BL-39",
        commit="014687cc",
        cwd=".",
        command="python -m pytest tests/regression/test_feature_contracts.py -q",
        green="26 passed",
        path="src/forecasting/enhance_v2.py",
        old='            macro["rate_diff_ibr_ust2y"] = (macro[IBR] - macro[UST2Y]).shift(1)',
        new='            macro["rate_diff_ibr_ust2y"] = (macro[IBR] - macro[UST2Y])',
        expected_text="4 failed — 2 muros de hash + los 2 muros nuevos de CAUSALIDAD",
        expected_failed=4,
        note="muta-2 (re-registrar el hash con la fuga dentro) queda MANUAL: toca 5 ocurrencias en otro fichero.",
    ),
    Mutation(
        bl="BL-42",
        commit="955374d0",
        cwd=".",
        command="python -m pytest tests/regression/test_return_units.py -q",
        green="28 passed, 3 skipped   (los 3 skips exigen Postgres arriba)",
        path="scripts/pipeline/train_and_export_smart_simple.py",
        old='        "total_return_pct": round(total_return, 2),',
        new='        "total_return_pct": round(total_return / 100.0, 6),',
        expected_text="2 failed — un decimal bajo sufijo _pct (14.46 pp -> 0.144616)",
        expected_failed=2,
    ),
    Mutation(
        bl="SYNTH-503",
        commit="46119274 (rojo) / a0a15e91 (fix)",
        cwd=DASHBOARD,
        command="npx vitest run tests/unit/api/synthetic-backtest-honesty.test.ts",
        green="4 passed",
        path="usdcop-trading-dashboard/app/api/backtest/route.ts",
        old=(
            "  } catch (error) {\n"
            "    // NEVER swallow this silently: an empty `catch {}` is what let the synthetic\n"
            "    // fallback masquerade as a real backtest for months.\n"
            "    const reason = error instanceof Error ? `${error.name}: ${error.message}`"
            " : String(error);\n"
            "    return backendUnavailable(`${BACKEND_URL}/v1/backtest unreachable (${reason})`);\n"
            "  }\n"
        ),
        new=(
            "  } catch {\n"
            "    // MUTACIÓN: revertido al fallback sintético pre-a0a15e91 (200 + success:true).\n"
            "    const startMs = Date.now();\n"
            "    const trades = generateSyntheticTrades({\n"
            "      startDate: body.start_date as string,\n"
            "      endDate: body.end_date as string,\n"
            "      modelId: body.model_id as string,\n"
            "    });\n"
            "    return NextResponse.json({\n"
            "      success: true,\n"
            "      source: 'generated',\n"
            "      trade_count: trades.length,\n"
            "      trades,\n"
            "      summary: calculateBacktestSummary(trades),\n"
            "      processing_time_ms: Date.now() - startMs,\n"
            "    });\n"
            "  }\n"
        ),
        expected_text=(
            "1 failed, 3 passed — 'HTTP 200 con N trades FABRICADOS ... y aun así declara success:true'"
        ),
        expected_failed=1,
        expected_passed=3,
        note="Solo la ruta /api/backtest. Las otras dos (stream, load-trades) quedan MANUALES; ver dossier.",
    ),
]

# BLs del dossier que NO están aquí a propósito (mutación no automatizable):
#   BL-06 — la mutación es un FICHERO NUEVO (lib/telemetry/RogueProbe.tsx), no una
#           sustitución dentro de un fichero existente. Manual, receta literal en el dossier.
MANUAL_ONLY = {
    "BL-06": "la mutación crea un fichero nuevo (lib/telemetry/RogueProbe.tsx), no sustituye texto",
}


# ─────────────────────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────────────────────
ANSI = re.compile(r"\x1B\[[0-9;]*[A-Za-z]")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_is_dirty(rel_path: str) -> str | None:
    """Devuelve la línea de `git status --porcelain` si el fichero está sucio, o None."""
    out = subprocess.run(
        ["git", "status", "--porcelain", "--", rel_path],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    line = (out.stdout or "").strip()
    return line or None


def print_diff(rel_path: str, old: str, new: str, start_line: int) -> None:
    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{rel_path}",
        tofile=f"b/{rel_path}",
        n=0,
    )
    print(f"  DIFF QUE SE VA A INTRODUCIR (a partir de la línea {start_line}):")
    for i, line in enumerate(diff):
        if i < 2:  # cabeceras ---/+++
            print(f"    {line.rstrip()}")
        else:
            print(f"    {line.rstrip()}")
    print()


def parse_counts(raw: str) -> dict[str, int]:
    """Extrae conteos de pytest -q y de vitest. Devuelve {} si no reconoce nada."""
    txt = ANSI.sub("", raw)
    counts: dict[str, int] = {}

    # vitest:  " Tests  1 failed | 3 passed (4)"
    m = re.search(r"^\s*Tests\s+(.+)$", txt, re.MULTILINE)
    if m:
        for n, kind in re.findall(r"(\d+)\s+(failed|passed|skipped|todo)", m.group(1)):
            counts[kind] = int(n)
        if counts:
            return counts

    # pytest -q: "=== 8 failed, 41 passed in 3.2s ===" / "=== 26 passed in 5s ==="
    tail = "\n".join(txt.strip().splitlines()[-8:])
    for n, kind in re.findall(r"(\d+)\s+(failed|passed|skipped|xfailed|xpassed|error[s]?)", tail):
        counts[kind.rstrip("s") if kind.startswith("error") else kind] = int(n)
    return counts


def fmt_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "<no parseado — leer salida cruda>"
    order = ["failed", "error", "passed", "skipped", "xfailed", "xpassed", "todo"]
    return ", ".join(f"{counts[k]} {k}" for k in order if k in counts)


def run_command(cwd: str, command: str) -> tuple[int, str]:
    proc = subprocess.run(
        command, cwd=str(REPO / cwd), shell=True,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=TIMEOUT_S,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def locate(text: str, needle: str) -> tuple[str, int]:
    """Devuelve (variante_encontrada, n_ocurrencias). Prueba LF y CRLF."""
    n = text.count(needle)
    if n:
        return needle, n
    crlf = needle.replace("\n", "\r\n")
    if crlf != needle:
        n = text.count(crlf)
        if n:
            return crlf, n
    return needle, 0


# ─────────────────────────────────────────────────────────────────────────────────────
# Ejecución de UNA mutación
# ─────────────────────────────────────────────────────────────────────────────────────
def run_one(mut: Mutation) -> dict:
    print("=" * 92)
    print(f"{mut.bl}   commit {mut.commit}")
    print("=" * 92)
    print(f"  fichero : {mut.path}")
    print(f"  comando : ({mut.cwd}) {mut.command}")
    print(f"  verde   : {mut.green}")
    print(f"  esperado: {mut.expected_text}")
    if mut.note:
        print(f"  nota    : {mut.note}")
    print()

    target = REPO / mut.path
    if not target.is_file():
        print(f"  ABORTADO: {mut.path} no existe.")
        return {"bl": mut.bl, "obtenido": "ABORTADO (fichero inexistente)", "coincide": "—"}

    dirty = git_is_dirty(mut.path)
    if dirty:
        print(f"  ABORTADO — ÁRBOL SUCIO en el fichero a mutar:  {dirty}")
        print("  Mutar sobre WIP produce un rojo que no corresponde a NINGÚN commit, así que")
        print("  no sería evidencia de nada. Commitea o descarta ese cambio y vuelve a correr.")
        return {"bl": mut.bl, "obtenido": "ABORTADO (árbol sucio)", "coincide": "—"}

    original_bytes = target.read_bytes()
    sha_before = hashlib.sha256(original_bytes).hexdigest()
    text = original_bytes.decode("utf-8")

    needle, n = locate(text, mut.old)
    if n != 1:
        print(f"  ABORTADO: la cadena a mutar aparece {n} veces (se exige exactamente 1).")
        print("  El código cambió respecto a lo que el dossier transcribe: revísalo A MANO.")
        return {"bl": mut.bl, "obtenido": f"ABORTADO (ocurrencias={n})", "coincide": "—"}

    replacement = mut.new.replace("\n", "\r\n") if needle != mut.old else mut.new
    start_line = text[: text.index(needle)].count("\n") + 1
    print(f"  sha256 antes : {sha_before}")
    print()
    print_diff(mut.path, needle.replace("\r\n", "\n"), replacement.replace("\r\n", "\n"), start_line)

    rc, raw = -1, ""
    try:
        target.write_bytes(text.replace(needle, replacement, 1).encode("utf-8"))
        print("  MUTADO. Ejecutando…\n")
        rc, raw = run_command(mut.cwd, mut.command)
    finally:
        target.write_bytes(original_bytes)
        sha_after = sha256_of(target)
        if sha_after != sha_before:
            print()
            print("!" * 92)
            print("!! RESTAURACIÓN CORRUPTA — el fichero NO volvió a su estado original.")
            print(f"!!   antes : {sha_before}")
            print(f"!!   después: {sha_after}")
            print(f"!!   {mut.path}")
            print("!! RESTAURA A MANO ANTES DE SEGUIR:  git checkout -- " + mut.path)
            print("!" * 92)
            sys.exit(1)

    tail = "\n".join(ANSI.sub("", raw).strip().splitlines()[-30:])
    print("  ── SALIDA CRUDA (últimas 30 líneas) " + "─" * 55)
    for line in tail.splitlines():
        print("  | " + line)
    print("  " + "─" * 90)

    counts = parse_counts(raw)
    obtenido = fmt_counts(counts)
    print(f"  exit code    : {rc}")
    print(f"  obtenido     : {obtenido}")
    print(f"  sha256 después: {sha_after}   RESTAURACIÓN VERIFICADA OK")

    if mut.expected_failed is None:
        coincide = "n/a (<sin registrar>)"
    else:
        ok = counts.get("failed", 0) == mut.expected_failed
        if mut.expected_passed is not None:
            ok = ok and counts.get("passed", -1) == mut.expected_passed
        coincide = "sí" if ok else "NO"
    print(f"  coincide con lo declarado: {coincide}")
    print("  (el veredicto sobre si esa aserción ES la garantía del BL no lo da este script)")
    print()
    return {"bl": mut.bl, "obtenido": obtenido, "coincide": coincide,
            "esperado": mut.expected_text}


# ─────────────────────────────────────────────────────────────────────────────────────
def cmd_list() -> None:
    print("BLs automatizados en este corredor:\n")
    print(f"  {'BL':<12} {'commit':<26} {'fichero mutado':<58} esperado")
    print("  " + "-" * 130)
    for m in MUTATIONS:
        exp = m.expected_text if len(m.expected_text) < 60 else m.expected_text[:57] + "..."
        print(f"  {m.bl:<12} {m.commit:<26} {m.path:<58} {exp}")
    print("\nBLs del dossier que NO se automatizan (hazlos A MANO con la receta del dossier):\n")
    for bl, why in MANUAL_ONLY.items():
        print(f"  {bl:<12} {why}")
    print("\nDossier (fuente de todo lo de arriba, y camino canónico si no te fías del script):")
    print("  .claude/coordination/integration/REVIEW-DOSSIER-CLAUDE.md")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Corredor de mutaciones (CONVENIENCIA, no autoridad). Ver cabecera del fichero.",
    )
    ap.add_argument("--list", action="store_true", help="lista los BLs disponibles y sale")
    ap.add_argument("--bl", help="corre un solo BL (p.ej. --bl BL-09)")
    args = ap.parse_args()

    if args.list:
        cmd_list()
        return 0

    selected = MUTATIONS
    if args.bl:
        key = args.bl.strip().upper()
        selected = [m for m in MUTATIONS if m.bl.upper() == key]
        if not selected:
            print(f"'{args.bl}' no está en el corredor.")
            if key in MANUAL_ONLY:
                print(f"  Es MANUAL: {MANUAL_ONLY[key]}")
            print("  `--list` muestra los disponibles.")
            return 1

    print()
    print("Este script es una CONVENIENCIA escrita por la parte revisada. No emite veredictos.")
    print("Camino canónico: .claude/coordination/integration/REVIEW-DOSSIER-CLAUDE.md")
    print()

    results = [run_one(m) for m in selected]

    print("=" * 92)
    print("RESUMEN — conteos crudos, sin veredicto")
    print("=" * 92)
    print(f"  {'BL':<12} {'esperado (declarado)':<62} {'obtenido':<28} coincide")
    print("  " + "-" * 128)
    for r in results:
        exp = r.get("esperado", "—")
        exp = exp if len(exp) < 62 else exp[:59] + "..."
        print(f"  {r['bl']:<12} {exp:<62} {r['obtenido']:<28} {r['coincide']}")
    print()
    print("  'coincide=sí' significa SOLO que el conteo cuadra con lo declarado. Que el rojo")
    print("  PRUEBE la garantía del BL es un juicio que corresponde al revisor, leyendo el test.")
    if any(r["coincide"] == "NO" for r in results):
        print()
        print("  HAY AL MENOS UN 'NO'. Eso significa una de dos cosas, y hay que distinguirlas:")
        print("  o el candado se degradó, o la transcripción del dossier está mal. Ninguna de las")
        print("  dos se resuelve mirando esta tabla: hay que abrir el test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
