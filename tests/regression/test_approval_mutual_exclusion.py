"""Exclusión mutua REAL sobre el artefacto del Voto 2 — Python↔Python y Python↔Node.

Contract: CTR-APPROVAL-STORE-001 (extiende CXD-057)

Por qué este fichero existe aparte de ``test_approval_store_private.py``
=======================================================================
El lado Node del Voto 2 ya publica bajo lock (``lib/approvals/store.ts::
commitApprovalTransition``: ``O_EXCL`` + relectura del disco bajo el lock + tmp/fsync/
rename). El lado Python tenía el lock DISPONIBLE (``acquire_approval_lock``) pero sus
ESCRITORES reales no lo tomaban: leían, mutaban en memoria y hacían ``open(path,"w")``.
Resultado: un export o un publish concurrente con un Voto 2 podía pisar el artefacto, y
el lock del handler no protegía de nada, porque el otro escritor ni lo miraba.

**Un test in-process no puede probar esto.** Cualquier doble/fake serializa por
construcción (un solo intérprete, un solo hilo de escritura), así que pasaría igual de
verde con y sin lock. Aquí se lanzan PROCESOS DE VERDAD:

  * ``test_naive_read_modify_write_really_races`` — TESTIGO. Corre el patrón viejo
    (leer → mutar → ``open(w)``) en N procesos reales y demuestra que produce **más de
    un ganador**. Sin este testigo, el test del store no probaría nada: podría estar
    pasando porque la carrera nunca ocurre en esta máquina.
  * ``test_store_transition_yields_exactly_one_winner`` — los MISMOS N procesos por
    ``commit_approval_transition`` ⇒ **exactamente uno** gana, el resto recibe conflicto,
    y el artefacto queda en un estado coherente (sin mezcla de identidades).
  * ``test_python_is_excluded_by_a_real_node_lock`` / ``test_node_is_excluded_by_the_
    python_lock`` — un proceso ``node`` REAL toma/observa el lockfile con el MISMO
    primitivo (``fs.open(lock,'wx')`` = ``O_CREAT|O_EXCL``) y el MISMO sufijo, leído en
    tiempo de ejecución DESDE ``store.ts`` (si el sufijo se toca en un solo lado, este
    test se cae, que es justo lo que debe pasar).

Fail-closed: cuando el lock no se puede tomar, el escritor Python **aborta con motivo**
(``ApprovalLockTimeout``) y NO escribe. Jamás escritura a ciegas.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.contracts import approval_store as store  # noqa: E402

N_PROCS = 6
PENDING = {
    "status": "PENDING_APPROVAL",
    "strategy": "smart_simple_v11",
    "strategy_name": "Smart Simple v1.1",
    "gates": [{"gate": "deflated_sharpe", "passed": False, "value": 0.05, "threshold": 0.95}],
    "created_at": "2026-07-01T00:00:00Z",
    "last_updated": "2026-07-01T00:00:00Z",
}


# ────────────────────────────────────────────────────────────── utilidades de proceso


def _seed(tmp_path: Path) -> Path:
    target = tmp_path / "approval_state.json"
    target.write_text(json.dumps(PENDING, indent=2), encoding="utf-8")
    return target


def _spawn(script: Path, args: list[str], approvals_dir: Path) -> subprocess.Popen:
    env = {**os.environ, "APPROVALS_DATA_DIR": str(approvals_dir), "PYTHONPATH": str(REPO)}
    return subprocess.Popen(
        [sys.executable, str(script), *args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
    )


def _race(script: Path, approvals_dir: Path, hold_s: float) -> list[tuple[int, str, str]]:
    """Lanza N procesos que arrancan a la vez (barrera de reloj) y devuelve sus salidas."""
    start = time.time() + 2.0  # margen para que los N intérpretes arranquen e importen
    procs = [_spawn(script, [f"w{i}", f"{start:.6f}", f"{hold_s}"], approvals_dir)
             for i in range(N_PROCS)]
    out = []
    for p in procs:
        so, se = p.communicate(timeout=120)
        out.append((p.returncode, so.strip(), se.strip()))
    return out


# ══════════════════════════ 1 · TESTIGO: el patrón viejo SÍ se pisa ══════════════════

_NAIVE = '''
import json, os, sys, time
who, start, hold = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
path = os.path.join(os.environ["APPROVALS_DATA_DIR"], "approval_state.json")
while time.time() < start:
    time.sleep(0.001)
# read -> modify -> open(w): exactamente lo que hacian los escritores Python.
with open(path, encoding="utf-8") as fh:
    cur = json.load(fh)
if cur.get("status") != "PENDING_APPROVAL":
    print("CONFLICT " + str(cur.get("status"))); sys.exit(3)
time.sleep(hold)                      # la latencia real (I/O, backtest, red)
cur["status"] = "APPROVED"; cur["approved_by"] = who; cur["last_updated"] = who
with open(path, "w", encoding="utf-8") as fh:
    json.dump(cur, fh, indent=2)
print("WON " + who); sys.exit(0)
'''


def test_naive_read_modify_write_really_races(tmp_path):
    """TESTIGO — sin este rojo, el verde del store no significa nada.

    El patrón que tenían ``run_btc_pipeline`` / ``publish_gold_*`` /
    ``backtest_2026_production`` / ``train_and_export_smart_simple``: leer, decidir sobre
    lo leído, escribir. Con N procesos reales, TODOS leen ``PENDING_APPROVAL`` y TODOS
    escriben: varios "ganadores" y el último pisa al resto.
    """
    script = tmp_path / "naive_writer.py"
    script.write_text(textwrap.dedent(_NAIVE), encoding="utf-8")
    target = _seed(tmp_path)

    results = _race(script, tmp_path, hold_s=0.30)
    winners = [o for _, o, _ in results if o.startswith("WON")]

    assert len(winners) > 1, (
        f"la carrera no se materializó en esta máquina (ganadores={winners}); el test "
        f"del store no probaría nada. Salidas: {results}")
    # y el artefacto refleja a UNO solo — los demás "ganadores" creen haber ganado.
    on_disk = json.loads(target.read_text(encoding="utf-8"))
    assert on_disk["approved_by"] in {w.split()[1] for w in winners}


# ═══════════════════ 2 · el store: exactamente un ganador, resto en conflicto ════════


def test_windows_permission_error_with_visible_lock_is_contention(tmp_path, monkeypatch):
    """EACCES transitorio sobre un lock visible respeta el contrato de timeout."""
    target = tmp_path / "approval_state.json"
    lock = Path(str(target) + store.LOCK_SUFFIX)
    lock.write_text("held", encoding="utf-8")

    def denied(*args, **kwargs):
        raise PermissionError("simulated Windows lock deletion race")

    monkeypatch.setattr(store.os, "open", denied)
    with pytest.raises(store.ApprovalLockTimeout, match="approval state busy"):
        with store.acquire_approval_lock(target, timeout_s=0):
            pytest.fail("un lock en contencion no puede adquirirse")


def test_permission_error_without_visible_lock_is_not_disguised(tmp_path, monkeypatch):
    """Un ACL/EACCES real conserva PermissionError; no se convierte en BUSY."""
    target = tmp_path / "approval_state.json"

    def denied(*args, **kwargs):
        raise PermissionError("simulated ACL denial")

    monkeypatch.setattr(store.os, "open", denied)
    with pytest.raises(PermissionError, match="simulated ACL denial"):
        with store.acquire_approval_lock(target, timeout_s=0):
            pytest.fail("un path sin permisos no puede adquirirse")


def test_old_orphan_lock_is_never_reclaimed_automatically(tmp_path):
    """Un huérfano visible queda fail-closed hasta una operación comprobada."""
    target = tmp_path / "approval_state.json"
    lock = Path(str(target) + store.LOCK_SUFFIX)
    lock.write_text('{"pid": 999999}', encoding="utf-8")
    os.utime(lock, (1, 1))

    with pytest.raises(store.ApprovalLockTimeout, match="verify no approval writer"):
        with store.acquire_approval_lock(target, timeout_s=0):
            pytest.fail("un lock viejo no puede robarse por mtime")
    assert lock.exists(), "el lock huérfano solo se retira tras comprobar writers vivos"


def test_old_live_lock_is_never_reclaimed_automatically(tmp_path):
    """Un titular lento conserva exclusión aunque su mtime supere el umbral retirado.

    La mutación que reintroduce ``unlink`` es observable aquí en POSIX/CI. En Windows,
    el handle abierto impide el borrado antes de que el código pueda robar el lock; el
    detector portable de abajo cubre allí la decisión de intentar el reclaim.
    """
    target = tmp_path / "approval_state.json"
    lock = Path(str(target) + store.LOCK_SUFFIX)
    lock.write_text(f'{{"pid": {os.getpid()}}}', encoding="utf-8")
    os.utime(lock, (1, 1))
    holder = os.open(lock, os.O_RDONLY)
    try:
        with pytest.raises(store.ApprovalLockTimeout):
            with store.acquire_approval_lock(target, timeout_s=0):
                pytest.fail("un titular vivo no puede perder el lock por mtime")
        assert lock.exists()
    finally:
        os.close(holder)


def test_old_visible_lock_never_attempts_unlink(tmp_path, monkeypatch):
    """C036 prohíbe incluso intentar reclamar por edad, con independencia del SO."""
    target = tmp_path / "approval_state.json"
    lock = Path(str(target) + store.LOCK_SUFFIX)
    lock.write_text('{"pid": 999999}', encoding="utf-8")
    os.utime(lock, (1, 1))

    original_unlink = Path.unlink
    attempted_lock_unlinks: list[Path] = []

    def spy_unlink(path: Path, *args, **kwargs):
        if path == lock:
            attempted_lock_unlinks.append(path)
            # Hace determinista la observación también en POSIX: un reclaim
            # defectuoso no llega a adquirir el lock después de registrarse.
            raise PermissionError("simulated visible-lock sharing violation")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", spy_unlink)
    with pytest.raises(store.ApprovalLockTimeout):
        with store.acquire_approval_lock(target, timeout_s=0):
            pytest.fail("un lock visible no puede reclamarse por edad")

    assert attempted_lock_unlinks == [], (
        "C036 prohíbe llamar unlink sobre un lock visible, no sólo que el SO rechace el borrado"
    )


_STORE = '''
import json, os, sys, time
sys.path.insert(0, os.environ["REPO_ROOT"])
from src.contracts.approval_store import (
    ApprovalConflict, ApprovalLockTimeout, commit_approval_transition)

who, start, hold = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
while time.time() < start:
    time.sleep(0.001)

def precondition(cur):
    # Se evalua DENTRO del lock, sobre el estado releido del disco. El sleep emula la
    # latencia real del escritor sin tocar codigo de produccion (equivalente al
    # `slowWrites` del Vitest del Voto 2).
    time.sleep(hold)
    if cur.get("status") != "PENDING_APPROVAL":
        return "already " + str(cur.get("status"))
    return None

try:
    nxt = commit_approval_transition(
        None,
        precondition=precondition,
        mutate=lambda cur: {**cur, "status": "APPROVED", "approved_by": who,
                            "last_updated": who},
    )
    print("WON " + who); sys.exit(0)
except ApprovalConflict as e:
    print("CONFLICT " + str(e)); sys.exit(3)
except ApprovalLockTimeout as e:
    print("BUSY " + str(e)); sys.exit(4)
'''


def test_store_transition_yields_exactly_one_winner(tmp_path):
    """Los MISMOS N procesos reales, por el store ⇒ 1 ganador, N-1 conflictos."""
    script = tmp_path / "store_writer.py"
    script.write_text(textwrap.dedent(_STORE), encoding="utf-8")
    target = _seed(tmp_path)

    env_repo = os.environ.get("REPO_ROOT")
    os.environ["REPO_ROOT"] = str(REPO)
    try:
        results = _race(script, tmp_path, hold_s=0.30)
    finally:
        if env_repo is None:
            os.environ.pop("REPO_ROOT", None)
        else:
            os.environ["REPO_ROOT"] = env_repo

    winners = [o for _, o, _ in results if o.startswith("WON")]
    losers = [o for _, o, _ in results if o.startswith(("CONFLICT", "BUSY"))]

    assert len(winners) == 1, f"esperado 1 ganador, salidas={results}"
    assert len(losers) == N_PROCS - 1, f"el resto debe recibir conflicto, salidas={results}"
    assert [rc for rc, _, _ in results].count(0) == 1

    # Estado coherente: la identidad del ganador y nada mezclado.
    on_disk = json.loads(target.read_text(encoding="utf-8"))
    assert on_disk["status"] == "APPROVED"
    assert on_disk["approved_by"] == winners[0].split()[1]
    assert list(tmp_path.glob("*.tmp-*")) == []      # publicación atómica: sin basura
    assert list(tmp_path.glob("*.lock")) == []       # el lock se libera siempre


def test_lock_is_released_even_when_the_transition_conflicts(tmp_path, monkeypatch):
    """Un conflicto no puede dejar el artefacto bloqueado para siempre."""
    monkeypatch.setenv("APPROVALS_DATA_DIR", str(tmp_path))
    target = _seed(tmp_path)
    with pytest.raises(store.ApprovalConflict):
        store.commit_approval_transition(
            None, precondition=lambda cur: "no", mutate=lambda cur: cur)
    assert not (tmp_path / "approval_state.json.lock").exists()
    # y el artefacto sigue intacto
    assert json.loads(target.read_text(encoding="utf-8"))["status"] == "PENDING_APPROVAL"


def test_missing_artifact_fails_closed_unless_creation_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setenv("APPROVALS_DATA_DIR", str(tmp_path))
    with pytest.raises(store.ApprovalMissing):
        store.commit_approval_transition(
            "x_1", mutate=lambda cur: {"status": "PENDING_APPROVAL", "strategy": "x_1"})
    assert list(tmp_path.glob("*.json")) == []

    st = store.commit_approval_transition(
        "x_1", create_if_absent=True,
        mutate=lambda cur: {**cur, "status": "PENDING_APPROVAL", "strategy": "x_1"})
    assert st["status"] == "PENDING_APPROVAL"
    assert (tmp_path / "approval_state_x_1.json").is_file()


def test_transition_validates_before_publishing(tmp_path, monkeypatch):
    """La mutación pasa por el MISMO validador que ``write_approval``: un documento
    inválido no llega al disco y el anterior queda íntegro."""
    monkeypatch.setenv("APPROVALS_DATA_DIR", str(tmp_path))
    target = _seed(tmp_path)
    with pytest.raises(ValueError):
        store.commit_approval_transition(
            None, mutate=lambda cur: {**cur, "status": "MAYBE"})
    assert json.loads(target.read_text(encoding="utf-8"))["status"] == "PENDING_APPROVAL"

    with pytest.raises(ValueError):
        store.commit_approval_transition(
            None, mutate=lambda cur: {**cur, "gates": [{"value": float("inf")}]})
    assert json.loads(target.read_text(encoding="utf-8"))["gates"][0]["value"] == 0.05


# ═══════════════ 3 · exclusión CRUZADA con un proceso `node` REAL ════════════════════

NODE = shutil.which("node")
STORE_TS = REPO / "usdcop-trading-dashboard" / "lib" / "approvals" / "store.ts"

# El script node deriva el sufijo del lock DESDE store.ts: si alguien lo cambia en un
# solo lado, estos tests se caen — que es exactamente la señal que hace falta.
_NODE_HOLD = r"""
const fs = require('fs');
const src = fs.readFileSync(process.argv[2], 'utf-8');
const m = src.match(/export const LOCK_SUFFIX = '([^']+)'/);
if (!m) { console.error('NO_LOCK_SUFFIX_IN_STORE_TS'); process.exit(9); }
const lock = process.argv[3] + m[1];
const holdMs = Number(process.argv[4]);
let fh;
try { fh = fs.openSync(lock, 'wx'); }        // O_CREAT|O_EXCL — el mismo primitivo
catch (e) { console.log('EEXIST'); process.exit(3); }
fs.writeSync(fh, JSON.stringify({ pid: process.pid, at: new Date().toISOString() }));
console.log('HELD');
setTimeout(() => { fs.closeSync(fh); fs.rmSync(lock, { force: true }); process.exit(0); }, holdMs);
"""


@pytest.mark.skipif(NODE is None, reason="node no disponible en esta máquina")
def test_python_is_excluded_by_a_real_node_lock(tmp_path):
    """Node (proceso REAL) sostiene el lock ⇒ Python aborta con motivo y NO escribe."""
    script = tmp_path / "hold_lock.js"
    script.write_text(_NODE_HOLD, encoding="utf-8")
    target = _seed(tmp_path)
    os.environ["APPROVALS_DATA_DIR"] = str(tmp_path)
    try:
        p = subprocess.Popen([NODE, str(script), str(STORE_TS), str(target), "3000"],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # esperar a que node confirme que TIENE el lock
        assert p.stdout.readline().strip() == "HELD"

        t0 = time.monotonic()
        with pytest.raises(store.ApprovalLockTimeout):
            store.commit_approval_transition(
                None, timeout_s=0.6,
                mutate=lambda cur: {**cur, "status": "APPROVED", "approved_by": "python"})
        assert time.monotonic() - t0 >= 0.5, "debe ESPERAR el lock, no fallar al instante"

        # fail-closed: ni un byte del artefacto tocado
        assert json.loads(target.read_text(encoding="utf-8")) == PENDING
        assert list(tmp_path.glob("*.tmp-*")) == []
        p.kill()
        p.communicate(timeout=30)
    finally:
        os.environ.pop("APPROVALS_DATA_DIR", None)


@pytest.mark.skipif(NODE is None, reason="node no disponible en esta máquina")
def test_node_is_excluded_by_the_python_lock(tmp_path):
    """Python sostiene el lock ⇒ el `wx` de node (proceso REAL) recibe EEXIST."""
    script = tmp_path / "hold_lock.js"
    script.write_text(_NODE_HOLD, encoding="utf-8")
    target = _seed(tmp_path)

    with store.acquire_approval_lock(target):
        r = subprocess.run([NODE, str(script), str(STORE_TS), str(target), "10"],
                           capture_output=True, text=True, timeout=60)
        assert r.stdout.strip() == "EEXIST", (r.returncode, r.stdout, r.stderr)
        assert r.returncode == 3

    # liberado el lock de Python, node lo toma sin problema (no quedó envenenado)
    r = subprocess.run([NODE, str(script), str(STORE_TS), str(target), "10"],
                       capture_output=True, text=True, timeout=60)
    assert r.stdout.strip() == "HELD", (r.returncode, r.stdout, r.stderr)


def test_lock_protocol_is_identical_on_both_sides():
    """Ambos lados esperan igual y ninguno roba locks por una edad ambigua."""
    ts = STORE_TS.read_text(encoding="utf-8")
    wait_ms = int(re.search(r"LOCK_WAIT_MS = ([\d_]+)", ts).group(1).replace("_", ""))
    retry_ms = int(re.search(r"LOCK_RETRY_MS = ([\d_]+)", ts).group(1).replace("_", ""))
    assert wait_ms / 1000.0 == store._LOCK_WAIT_S, (wait_ms, store._LOCK_WAIT_S)
    assert f"time.sleep({retry_ms / 1000.0})" in \
        (REPO / "src" / "contracts" / "approval_store.py").read_text(encoding="utf-8")
    assert "LOCK_STALE" not in ts
    assert "_LOCK_STALE" not in \
        (REPO / "src" / "contracts" / "approval_store.py").read_text(encoding="utf-8")


def test_python_locks_the_resolved_artifact_not_the_scoped_name(tmp_path, monkeypatch):
    """Paridad de OBJETIVO, no solo de sufijo.

    TypeScript resuelve primero (``readApprovalState``) y bloquea ``record.file``. Si
    Python bloqueara siempre ``approval_state_<sid>.json`` mientras el artefacto REAL es
    el singleton, los dos lados tomarían locks distintos sobre el mismo fichero y la
    exclusión sería decorativa.
    """
    monkeypatch.setenv("APPROVALS_DATA_DIR", str(tmp_path))
    singleton = _seed(tmp_path)  # strategy = smart_simple_v11, sin fichero scoped

    seen: list[str] = []

    def spy(cur):
        seen.extend(p.name for p in tmp_path.glob("*.lock"))
        return None

    store.commit_approval_transition(
        "smart_simple_v11", precondition=spy,
        mutate=lambda cur: {**cur, "status": "APPROVED", "approved_by": "a"})

    assert seen == [singleton.name + store.LOCK_SUFFIX], seen
    assert json.loads(singleton.read_text(encoding="utf-8"))["status"] == "APPROVED"


# ══════════════════════ 4 · los ESCRITORES pasan por el store ═══════════════════════

#: Escritores del artefacto ya cableados al store (lock compartido + validación + atómico).
WIRED_WRITERS = [
    "scripts/pipeline/run_btc_pipeline.py",
    "scripts/pipeline/publish_gold_dynexit.py",
    "scripts/pipeline/publish_gold_trend_simple.py",
    "scripts/pipeline/replay_backtest_universal.py",
    "scripts/analysis/backtest_2026_production.py",
]

#: El ÚLTIMO eslabón. Cablearlo cambia los bytes de un fichero congelado por
#: ``config/strategy_manifests/{usdcop,usdcop_v12,usdcop_v14}.yaml`` y exige re-freeze
#: (bump de versión + refreeze_note + 9 hashes) — decisión de GOBIERNO de modelado, no
#: de fontanería. Ver el reporte: BLOCKED_OPERATOR_DECISION.
BLOCKED_WRITER = "scripts/pipeline/train_and_export_smart_simple.py"

_TRANSITION = re.compile(r"commit_approval_transition")


@pytest.mark.parametrize("rel", WIRED_WRITERS)
def test_writer_publishes_through_the_store_transition(rel):
    src = (REPO / rel).read_text(encoding="utf-8")
    assert _TRANSITION.search(src), (
        f"{rel} no publica por commit_approval_transition — sin el lock compartido, un "
        "Voto 2 concurrente y este escritor se pisan")


@pytest.mark.parametrize("rel", WIRED_WRITERS)
def test_writer_has_no_direct_write_to_the_approval_artifact(rel):
    """El camino correcto tiene que ser el ÚNICO: sin ``open(ap,'w')`` ni ``write_text``
    sobre la ruta de aprobación, y sin ``_dump`` propio hacia ella."""
    src = (REPO / rel).read_text(encoding="utf-8")
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    offenders = re.findall(
        r"(open\(\s*(?:ap_path|approval_path|_approval_path\([^)]*\))[^\n]*[\"']w[\"']"
        r"|(?:ap_path|approval_path)\.write_text\("
        r"|_dump\(\s*_approval_path)", code)
    assert offenders == [], f"{rel} sigue escribiendo el artefacto a mano: {offenders}"


@pytest.mark.xfail(
    strict=True,
    reason="BLOCKED_OPERATOR_DECISION — cablear train_and_export_smart_simple.py cambia "
           "su code_hash canónico y tumba el muro de congelación de usdcop/usdcop_v12/"
           "usdcop_v14 (3 manifiestos × 3 hashes). El re-freeze es gobierno de modelado, "
           "no fontanería: requiere aprobación explícita del operador. Cuando se apruebe, "
           "cablear + re-freeze + BORRAR este xfail en el MISMO commit.")
def test_last_link_train_and_export_publishes_through_the_store():
    src = (REPO / BLOCKED_WRITER).read_text(encoding="utf-8")
    assert _TRANSITION.search(src), (
        f"{BLOCKED_WRITER}: el export y --reset-approval siguen haciendo "
        "read-modify-open(w) SIN el lock interproceso")
