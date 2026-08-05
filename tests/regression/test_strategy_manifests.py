"""Every champion has a frozen manifest, and the freeze is enforced by hash.

Contract: CTR-STRAT-MANIFEST-001 (Fase 0, plan 2026-07-21 — Codex step 1 + run-experiment)

A manifest is only immutable if something breaks when its subject changes. The teeth here:
`code_hash_sha256_16` is the hash of the strategy's source files at freeze time. Editing any
of those files without re-freezing the manifest turns this test red — which converts "we
changed the strategy" from a silent event into a conscious versioning decision, exactly what
the run-experiment skill demands of frozen SSOT configs ("config congelado al arrancar; si
hace falta ajustar, se ABORTA y se abre un experimento nuevo").

The clock is asserted per asset because the library carries two incompatible annualization
conventions (onboard-asset: BTC sqrt365; xasset engine: 252 after intersect) and a metric
cited without its clock is unfalsifiable.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from pathlib import Path

import pytest
import yaml

# THE hashing method under test lives in PRODUCTION, not here (BL-13 red-team fix).
# It used to be defined inside this file, so the freeze wall verified itself against its
# own implementation: there was nothing in production to mutate and the guarantee was
# circular. `src/identity/source_hash.py` is now the single implementation, shared with
# the feature-catalog gate (scripts/validation/validate_feature_catalog.py) — mutating it
# turns BOTH walls red.
from src.identity.source_hash import canonical_lf, files_code_hash

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = ROOT / "config" / "strategy_manifests"
LEDGER = ROOT / "registries" / "ledger.jsonl"
FORECASTING_SSOT = ROOT / "config" / "forecasting_ssot.yaml"
NORM_SNAPSHOTS = ROOT / "config" / "features" / "normalization_snapshots"
DAG_REGISTRY = ROOT / "airflow" / "dags" / "contracts" / "dag_registry.py"
BACKLOG = ROOT / ".claude" / "specs" / "planes" / "backlog"

EXPECTED_CLOCKS = {"usdcop": 52, "xauusd": 252, "btcusdt": 365, "spx500": 252}


def _champions() -> dict[str, str]:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "normalize_champions", ROOT / "scripts" / "pipeline" / "normalize_champions.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CHAMPION_BY_ASSET


def _manifest(asset: str) -> dict:
    p = MANIFESTS / f"{asset}.yaml"
    assert p.is_file(), f"champion asset {asset} has no frozen manifest at {p}"
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def test_every_champion_has_a_manifest_that_names_it():
    for asset, sid in _champions().items():
        m = _manifest(asset)
        assert m["strategy_id"] == sid, (
            f"{asset}: manifest freezes {m['strategy_id']!r} but the champion authority says "
            f"{sid!r}. A champion change requires a new manifest in the same commit."
        )


def test_clock_is_declared_and_correct():
    for asset, expect in EXPECTED_CLOCKS.items():
        m = _manifest(asset)
        got = m["clock"]["periods_per_year"]
        assert got == expect, f"{asset}: clock {got} != {expect}"


@pytest.mark.parametrize(
    "manifest_path", sorted(MANIFESTS.glob("*.yaml")), ids=lambda p: p.name)
def test_code_hash_detects_strategy_drift(manifest_path):
    """Editing a frozen strategy's source without re-freezing must fail CI.

    Red-team BL-14 hallazgo #1: this test used to iterate only over _champions(),
    so the paper candidates (usdcop_v12/usdcop_v14) drifted silently when commits
    277a072/878be51 edited their frozen execution YAMLs post-freeze. The freeze
    wall must cover EVERY manifest that declares `files:` — champion or not —
    hence the parametrization over the manifest glob (each file is its own red X).

    Hash method (CXD-041/043): hash canónico LF — each file's bytes with
    CRLF->LF normalized, i.e. the git blob content under .gitattributes text
    eol=lf. Reproducible from a clean checkout on any OS via
    `git show :<path>`. Hashing raw working-tree bytes (the old method)
    produced hashes only a Windows CRLF checkout could reproduce.
    """
    m = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert m.get("files") and m.get("code_hash_sha256_16"), (
        f"{manifest_path.name}: a frozen manifest must declare files: + "
        "code_hash_sha256_16 — a manifest without a hash is not frozen"
    )
    # mutación que lo pone rojo: en src/identity/source_hash.py, `canonical_lf` devuelve
    # `data` sin el .replace(b"\r\n", b"\n") (enhance_v2.py es CRLF en el working tree).
    current = files_code_hash(ROOT / f for f in m["files"])
    assert current == m["code_hash_sha256_16"], (
        f"{manifest_path.name} ({m.get('strategy_id')}): strategy source drifted from "
        f"its frozen manifest (manifest={m['code_hash_sha256_16']}, current={current}). "
        "Either revert the source change, or consciously re-freeze: bump the manifest "
        "version, update the hash, add a refreeze_note, and count the look as a trial "
        "if any result was observed."
    )


def test_code_hash_method_is_line_ending_invariant(tmp_path):
    """CXD-041/043 regression: the SAME logical content must hash identically
    whether the checkout produced CRLF (Windows) or LF (Linux/git blob).

    Red with the old method (raw bytes): sha256(CRLF bytes) != sha256(LF bytes),
    so the declared hashes were only reproducible on the machine that froze
    them. Green with the canonical method: CRLF->LF normalization makes both
    checkouts hash to the git-blob value.
    """
    content_lf = b"strategy = 1\nreturn strategy\n"
    content_crlf = content_lf.replace(b"\n", b"\r\n")
    f_lf = tmp_path / "as_linux_checkout.py"
    f_crlf = tmp_path / "as_windows_checkout.py"
    f_lf.write_bytes(content_lf)
    f_crlf.write_bytes(content_crlf)

    # Sanity: the OLD method (raw bytes) is OS-dependent — this is the bug.
    old_lf = hashlib.sha256(f_lf.read_bytes()).hexdigest()[:16]
    old_crlf = hashlib.sha256(f_crlf.read_bytes()).hexdigest()[:16]
    assert old_lf != old_crlf, "raw-byte hashing must differ across EOLs (the CXD-041 bug)"

    # The canonical PRODUCTION method is EOL-invariant and equals the git-blob (LF) hash.
    # mutación que lo pone rojo: `canonical_lf` en src/identity/source_hash.py devuelve
    # `data` tal cual (o cualquier constante) en vez de normalizar CRLF->LF.
    new_lf = hashlib.sha256(canonical_lf(f_lf.read_bytes())).hexdigest()[:16]
    new_crlf = hashlib.sha256(canonical_lf(f_crlf.read_bytes())).hexdigest()[:16]
    assert new_lf == new_crlf == old_lf, (
        "canonical LF hash must be identical for CRLF and LF checkouts and equal "
        "to the hash of the LF (git blob) content")


def test_component_code_hash_and_spec_fingerprint_are_canonical():
    """BL-14 lineage hashes use the SAME canonical LF method as the manifest hash.

    components[0].code_hash_sha256_16 = canonical hash over code_reference files
    (the manifests document 'mismo metodo que el hash del manifiesto');
    spec_fingerprint_sha256_16 = canonical hash over the declared
    spec_fingerprint_inputs (code_reference + the frozen hyperparams YAML).
    """
    for p in sorted(MANIFESTS.glob("*.yaml")):
        m = yaml.safe_load(p.read_text(encoding="utf-8"))
        for comp in m.get("components") or []:
            # mutación que lo pone rojo: quitar la normalización CRLF->LF de
            # src/identity/source_hash.py::canonical_lf (producción, no el test).
            assert files_code_hash(
                ROOT / f for f in comp["code_reference"]
            ) == comp["code_hash_sha256_16"], (
                f"{p.name}: components[0].code_hash_sha256_16 does not match the "
                "canonical LF hash of its code_reference files")
            inputs = comp.get("spec_fingerprint_inputs")
            assert inputs, (
                f"{p.name}: component must declare spec_fingerprint_inputs so the "
                "fingerprint is recomputable (CXD-041)")
            assert files_code_hash(
                ROOT / f for f in inputs
            ) == comp["spec_fingerprint_sha256_16"], (
                f"{p.name}: spec_fingerprint_sha256_16 does not match the canonical "
                "LF hash of its declared inputs")


def test_manifests_declare_action_surface():
    """BL-13: every frozen manifest declares its surface, and diagnostic never champions.

    `surface` is the discriminator between tradeable strategies (action) and
    look-only research surfaces (diagnostic). A diagnostic surface must never be
    the champion the registry serves — normalize_champions enforces it at runtime;
    this test enforces it at freeze time.
    """
    champions = set(_champions().values())
    manifests = sorted(MANIFESTS.glob("*.yaml"))
    assert manifests, f"no frozen manifests found under {MANIFESTS}"
    for p in manifests:
        m = yaml.safe_load(p.read_text(encoding="utf-8"))
        surface = m.get("surface")
        assert surface in {"action", "diagnostic"}, (
            f"{p.name}: surface is {surface!r} — every frozen manifest must declare "
            "surface: action|diagnostic (BL-13)"
        )
        if surface == "diagnostic":
            assert m["strategy_id"] not in champions, (
                f"{p.name}: {m['strategy_id']!r} declares surface=diagnostic but is a "
                "champion in CHAMPION_BY_ASSET — diagnostic surfaces can never be champions"
            )


def test_composite_declares_components():
    """BL-14 / FABRIC §16: composite strategies declare their frozen-RECIPE predictor.

    Any manifest whose model.kind is ml_ensemble (or that declares a components
    block) must expose components[0] with role=decision_input and recipe_frozen
    true. What is frozen is the RECIPE (features, hyperparams, weekly expanding
    retrain, train-only scaler) — never the weights: each Sunday retrain produces
    a registered model_snapshot under the same spec. Rule-based manifests without
    an ML model are exempt.
    """
    for p in sorted(MANIFESTS.glob("*.yaml")):
        m = yaml.safe_load(p.read_text(encoding="utf-8"))
        model = m.get("model") or {}
        is_composite = model.get("kind") == "ml_ensemble" or "components" in m
        if not is_composite:
            continue  # rule-based (no ML model): not required to declare components
        comps = m.get("components")
        assert isinstance(comps, list) and comps, (
            f"{p.name}: model.kind=ml_ensemble but no components block — declare the "
            "predictor as a frozen RECIPE (BL-14)"
        )
        first = comps[0]
        assert first.get("role") == "decision_input", (
            f"{p.name}: components[0].role is {first.get('role')!r}, expected "
            "'decision_input' (BL-14)"
        )
        assert first.get("recipe_frozen") is True, (
            f"{p.name}: components[0].recipe_frozen must be true — the recipe is frozen, "
            "snapshots are registered; never claim immutable weights (FABRIC §16)"
        )
        # BL-14 remediacion (rechazo funcional Codex): linaje por componente
        for key in ("spec_fingerprint_sha256_16", "code_reference",
                    "code_hash_sha256_16", "feature_set", "current_model_snapshot"):
            assert key in first, (
                f"{p.name}: components[0] missing lineage field {key!r} (BL-14)")


# ═══════════════════════════════════════════════════════════════════════════
# BL-14 red-team: el bloque `components` se comprobaba EXISTENTE, no CIERTO.
# Un `current_model_snapshot` completamente inventado (pointer "no/existe/",
# as_of 1999-01-01, hashes "0"*16, registered_in "ninguna parte") pasaba verde
# porque test_composite_declares_components solo verifica que la CLAVE está.
# Lo de abajo lo hace RESOLUBLE: el puntero contra el SSOT de paths, el as_of
# contra el reloj y el sello del manifiesto, los hashes contra el snapshot de
# normalización registrado (otro artefacto en git), y `registered_in` contra el
# registro real de DAGs.
# ═══════════════════════════════════════════════════════════════════════════

SHA16_RE = re.compile(r"^[0-9a-f]{16}$")
NULL_SHA16 = "0" * 16
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# Un snapshot no puede ser anterior al primer dato de su propia ventana de
# entrenamiento (expanding window 2020-01-01 -> último viernes; CLAUDE.md y
# `retrain_policy: weekly_expanding` "ventana 2020->" en los manifiestos).
MIN_SNAPSHOT_AS_OF = dt.date(2020, 1, 1)
PENDING_RE = re.compile(r"^pending-(BL-\d+)$")

# Un aplazamiento solo vale mientras el trabajo que lo justifica sigue abierto.
_ESTADOS_CERRADOS = {"DONE", "IMPLEMENTED", "APPROVED", "CLOSED"}


def _frontmatter_status(md_path):
    """Lee `status:` del front-matter de un BL. Devuelve '' si no lo declara."""
    texto = md_path.read_text(encoding="utf-8")
    if not texto.startswith("---"):
        return ""
    cabecera = texto.split("---", 2)[1]
    hit = re.search(r"^status:\s*(\S+)", cabecera, re.M)
    return hit.group(1).strip().upper() if hit else ""



def _manifests_with_components() -> list[tuple[Path, dict]]:
    out = []
    for p in sorted(MANIFESTS.glob("*.yaml")):
        m = yaml.safe_load(p.read_text(encoding="utf-8"))
        if m.get("components"):
            out.append((p, m))
    assert out, "ningún manifiesto declara components — el bloque BL-14 desapareció"
    return out


def _declared_model_dirs() -> set[str]:
    """Directorios de artefactos de modelo DECLARADOS por el SSOT de forecasting."""
    ssot = yaml.safe_load(FORECASTING_SSOT.read_text(encoding="utf-8"))
    return {str(v).strip("/") for k, v in (ssot.get("paths") or {}).items()
            if k.startswith("models_")}


def _known_dag_ids() -> set[str]:
    return set(re.findall(r'^[A-Z0-9_]+\s*=\s*"([a-z0-9_]+)"',
                          DAG_REGISTRY.read_text(encoding="utf-8"), re.M))


def _norm_snapshot_for(strategy_id: str) -> dict | None:
    """Snapshot de normalización REGISTRADO (en git) para esa estrategia, si existe."""
    for p in sorted(NORM_SNAPSHOTS.glob("*.yaml")):
        snap = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if snap.get("strategy_id") == strategy_id:
            return snap
    return None


def _ledger_records() -> list[dict]:
    return [json.loads(line) for line in
            LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_current_model_snapshot_is_resolvable():
    """BL-14 red-team: el snapshot declarado debe RESOLVER, no solo existir.

    mutación que lo pone rojo: en config/strategy_manifests/usdcop.yaml,
    `current_model_snapshot: {pointer: "no/existe/", as_of: "1999-01-01",
    artifacts_sha256_16: {ridge_h5.pkl: "0000000000000000"},
    registered_in: "ninguna parte"}` (cada una de las 4 claves cae por separado).
    """
    model_dirs = _declared_model_dirs()
    dag_ids = _known_dag_ids()
    champions = set(_champions().values())
    today = dt.date.today()

    for p, m in _manifests_with_components():
        for comp in m["components"]:
            snap = comp.get("current_model_snapshot")
            where = f"{p.name}/{comp.get('component_id')}"
            assert isinstance(snap, dict), (
                f"{where}: current_model_snapshot debe ser un mapping con linaje, "
                f"no {type(snap).__name__} (BL-14)")

            # 1. pointer: ruta relativa a un directorio de modelos DECLARADO en el SSOT.
            pointer = snap.get("pointer")
            assert isinstance(pointer, str) and pointer.strip(), (
                f"{where}: pointer debe ser una ruta no vacía")
            assert not Path(pointer).is_absolute() and ":" not in pointer, (
                f"{where}: pointer {pointer!r} debe ser relativo al repo, no absoluto")
            assert pointer.strip("/") in model_dirs, (
                f"{where}: pointer {pointer!r} no es ninguno de los directorios de "
                f"modelos declarados en config/forecasting_ssot.yaml::paths "
                f"({sorted(model_dirs)}) — un puntero que no nombra el sitio donde el "
                "pipeline escribe no es resoluble, es prosa")

            # 2. as_of: ISO, dentro del reloj del repo y no posterior al sello.
            as_of_raw = str(snap.get("as_of"))
            assert ISO_DATE_RE.match(as_of_raw), (
                f"{where}: as_of {as_of_raw!r} no es una fecha ISO YYYY-MM-DD")
            as_of = dt.date.fromisoformat(as_of_raw)
            assert MIN_SNAPSHOT_AS_OF <= as_of <= today, (
                f"{where}: as_of {as_of} fuera de rango — un snapshot no puede ser "
                f"anterior a su ventana de entrenamiento ({MIN_SNAPSHOT_AS_OF}) ni "
                f"venir del futuro ({today})")
            frozen_at = m.get("manifest_frozen_at")
            if frozen_at:
                assert as_of <= dt.date.fromisoformat(str(frozen_at)), (
                    f"{where}: as_of {as_of} es POSTERIOR al sello del manifiesto "
                    f"{frozen_at} — el manifiesto congelado solo puede declarar un "
                    "snapshot que ya existía al sellarlo")

            # 3. artifacts_sha256_16: hex de 16, jamás el hash nulo.
            arts = snap.get("artifacts_sha256_16")
            assert isinstance(arts, dict) and arts, (
                f"{where}: artifacts_sha256_16 debe declarar al menos un artefacto")
            for name, sha in arts.items():
                assert Path(str(name)).suffix in {".pkl", ".json"}, (
                    f"{where}: artefacto {name!r} sin extensión de artefacto conocida")
                assert isinstance(sha, str) and SHA16_RE.match(sha), (
                    f"{where}: {name} declara sha {sha!r} — debe ser hex minúscula de "
                    "16 chars (STRING, no int)")
                assert sha != NULL_SHA16, (
                    f"{where}: {name} declara el hash nulo {NULL_SHA16} — un placeholder "
                    "de ceros es un artefacto inventado, no un linaje")

            # 4. registered_in: donde se declare, debe nombrar un DAG del registro real.
            registered_in = snap.get("registered_in")
            if m["strategy_id"] in champions:
                assert registered_in, (
                    f"{where}: la campeona sirve estos pesos — su snapshot rotativo debe "
                    "declarar registered_in (dónde queda registrada cada corrida)")
            if registered_in:
                named = [d for d in dag_ids if d in str(registered_in)]
                assert named, (
                    f"{where}: registered_in {str(registered_in)[:60]!r} no nombra ningún "
                    "DAG de airflow/dags/contracts/dag_registry.py — 'registrado' en un "
                    "sitio que no existe es exactamente lo que este muro impide")

            # 5. Resolución CRUZADA contra el snapshot de normalización registrado
            #    (config/features/normalization_snapshots/*.yaml, también en git).
            norm = _norm_snapshot_for(m["strategy_id"])
            if norm is None:
                continue
            art = norm.get("artifact") or {}
            art_path = str(art.get("path", ""))
            assert str(Path(art_path).parent).replace("\\", "/") == pointer.strip("/"), (
                f"{where}: pointer {pointer!r} no coincide con el directorio del "
                f"artefacto registrado en el snapshot de normalización ({art_path})")
            assert arts.get(Path(art_path).name) == art.get("sha256_16"), (
                f"{where}: {Path(art_path).name} declara {arts.get(Path(art_path).name)!r} "
                f"pero el snapshot de normalización registrado sella "
                f"{art.get('sha256_16')!r} — dos registros del MISMO binario que no "
                "coinciden significa que uno miente")
            assert str((norm.get("training") or {}).get("as_of")) == as_of_raw, (
                f"{where}: as_of {as_of_raw} != training.as_of "
                f"{(norm.get('training') or {}).get('as_of')} del snapshot registrado")
            fs_id = norm.get("feature_set_id")
            fs_path = ROOT / "config" / "features" / "feature_sets" / f"{fs_id}.yaml"
            if fs_path.is_file():
                fs = yaml.safe_load(fs_path.read_text(encoding="utf-8")) or {}
                cols_name = Path(str(fs.get("source_file", ""))).name
                if cols_name in arts:
                    assert arts[cols_name] == norm.get("ordered_feature_hash_sha256_16"), (
                        f"{where}: {cols_name} declara {arts[cols_name]!r} pero el "
                        "snapshot de normalización sella ordered_feature_hash "
                        f"{norm.get('ordered_feature_hash_sha256_16')!r}")


def test_component_declares_forecast_lineage_key():
    """BL-14 exige heredar el linaje FT; hoy vale un placeholder, pero NO cualquiera.

    Borrar `forecast_trial_ids_legacy` no mordía. Ahora el componente debe declarar
    su linaje predictivo, y si lo aplaza el aplazamiento tiene que apuntar a un BL
    que EXISTA en el backlog (una excusa verificable, no prosa libre).

    mutación que lo pone rojo: borrar `forecast_trial_ids_legacy` de
    config/strategy_manifests/usdcop.yaml (o ponerle `pending-BL-99`, un BL inexistente).
    """
    for p, m in _manifests_with_components():
        for comp in m["components"]:
            where = f"{p.name}/{comp.get('component_id')}"
            keys = [k for k in ("forecast_trial_ids", "forecast_trial_ids_legacy")
                    if k in comp]
            assert keys, (
                f"{where}: el componente no declara linaje FT "
                "(forecast_trial_ids / forecast_trial_ids_legacy) — BL-14 exige heredar "
                "los trials predictivos que hicieron posible esta acción (ADR-0022 §2)")
            value = comp[keys[0]]
            if isinstance(value, list):
                continue  # linaje real: lo resuelve el test de abajo contra el ledger
            match = PENDING_RE.match(str(value).strip())
            assert match, (
                f"{where}: {keys[0]}={value!r} no es ni una lista de trial_ids ni un "
                "aplazamiento con la forma 'pending-BL-<n>'")
            bl = match.group(1)
            duenos = list(BACKLOG.glob(f"{bl}-*.md"))
            assert duenos, (
                f"{where}: aplaza el linaje a {bl}, que NO existe en "
                f"{BACKLOG.relative_to(ROOT)} — un placeholder que no apunta a trabajo "
                "real es una deuda invisible")

            # El aplazamiento CADUCA. Exigir solo que el fichero exista no cierra nada: un
            # fichero existe para siempre, asi que `pending-BL-10` seguiria siendo valido
            # despues de que BL-10 cerrara. Lo encontro una verificacion propia el
            # 2026-07-28, y contradecia lo que yo mismo habia anunciado por el canal.
            estado = _frontmatter_status(duenos[0])
            assert estado not in _ESTADOS_CERRADOS, (
                f"{where}: sigue aplazando el linaje a {bl}, pero {bl} ya esta en "
                f"'{estado}'. Un aplazamiento a trabajo TERMINADO no es una deuda "
                "declarada: es un placeholder rancio. Sustituye el valor por los "
                "forecast_trial_ids reales del ledger.")


def test_component_forecast_trial_ids_resolve_in_ledger():
    """BL-14 + BL-12: los FT heredados por el componente deben EXISTIR en el ledger.

    Mismo candado que check_trial_ledger.check_provenance_wall aplica a las familias:
    cada trial citado existe, es kind=forecast, es del mismo activo y vive en un
    cluster que efectivamente deflacta la acción de ese activo (ADR-0022 §3).

    Historial: nació `xfail(strict=True)` porque BL-10 (backfill legacy_estimate FT,
    de Codex) seguía abierto y los tres manifiestos COP declaraban el placeholder
    `pending-BL-10`; el strict garantizaba un XPASS ruidoso el día que el linaje real
    llegara. BL-10 cerró (status IMPLEMENTED) el 2026-07-28 y los componentes ya
    declaran FT-0001..FT-0048 — los tres bloques legacy COMPLETOS de la familia
    `usdcop_direction` (cluster ml_meta), derivados del ledger con la consulta
    `asset=usdcop AND kind=forecast AND label=legacy_estimate`, nunca elegidos a mano:
    sus celdas son `estimated_block`/`decomposable: false` y desagregarlas sería
    fabricar granularidad. El marcador se retiró: con linaje real esto es un muro
    normal y verde, y un xfail sobre trabajo terminado sería el mismo placeholder
    rancio que `test_component_declares_forecast_lineage_key` persigue.

    mutación que lo pone rojo: en config/strategy_manifests/usdcop.yaml sustituir
    cualquiera de los ids por uno inexistente (p.ej. FT-9999), por un AT- en vez de un
    FT-, o por un FT de otro cluster/activo (p.ej. FT-0049, cluster `vol`).
    """
    records = _ledger_records()
    by_id = {r["trial_id"]: r for r in records}

    for p, m in _manifests_with_components():
        asset = m["asset_id"]
        action_clusters = {r["cluster"] for r in records
                           if r["asset"] == asset and r["kind"] == "action"}
        for comp in m["components"]:
            where = f"{p.name}/{comp.get('component_id')}"
            declared = comp.get("forecast_trial_ids",
                                comp.get("forecast_trial_ids_legacy"))
            assert isinstance(declared, list), (
                f"{where}: el linaje FT sigue siendo el placeholder {declared!r} — "
                "un componente sin FT resolubles no tiene provenance (ADR-0022 §2)")
            assert declared, f"{where}: lista de forecast_trial_ids vacía"
            for tid in declared:
                rec = by_id.get(tid)
                assert rec is not None, (
                    f"{where}: cita {tid} que NO existe en registries/ledger.jsonl")
                assert rec["kind"] == "forecast", (
                    f"{where}: {tid} es kind={rec['kind']!r}, no forecast — heredar un AT "
                    "no es heredar linaje predictivo")
                assert rec["asset"] == asset, (
                    f"{where}: {tid} es del activo {rec['asset']!r}, no de {asset!r}")
                assert rec["cluster"] in action_clusters, (
                    f"{where}: {tid} vive en cluster {rec['cluster']!r} y la acción de "
                    f"{asset} deflacta en {sorted(action_clusters)} — un FT de otro "
                    "cluster NO entra en su N_cluster (ADR-0022 §3)")


def test_registry_carries_surface_and_diagnostic_never_visible():
    """BL-13 / C-005: `surface` must reach the registry the dashboard serves.

    The frozen YAMLs declaring surface is necessary but not sufficient — the registry
    (public/data/registry.json) is what the frontend actually reads, and a wall that
    exists only in files the frontend never opens is a wall by convention (the exact
    gap Codex rejected: strategies=18, surface_present=0). Every registry entry must
    declare surface, and a diagnostic entry may never be visible nor champion.
    """
    reg = json.loads((ROOT / "usdcop-trading-dashboard/public/data/registry.json")
                     .read_text(encoding="utf-8"))
    champions = set(_champions().values())
    assert reg["strategies"], "registry lists no strategies"
    for s in reg["strategies"]:
        sid = s.get("strategy_id")
        assert s.get("surface") in {"action", "diagnostic"}, (
            f"registry entry {sid!r}: surface is {s.get('surface')!r} — every registry "
            "entry must declare surface: action|diagnostic (BL-13/C-005). "
            "Run scripts/pipeline/normalize_champions.py to stamp+refresh."
        )
        if s["surface"] == "diagnostic":
            assert s.get("status") == "archived", (
                f"registry entry {sid!r}: surface=diagnostic with visible status "
                f"{s.get('status')!r} — diagnostic surfaces exist to be looked at, never traded"
            )
            assert sid not in champions, (
                f"registry entry {sid!r}: surface=diagnostic but champion in "
                "CHAMPION_BY_ASSET — a diagnostic surface can never be the champion served"
            )


def test_diagnostic_champion_forces_red_exit_and_archival(tmp_path, monkeypatch):
    """BL-13 verificación: entrada diagnostic con status CHAMPION => exit rojo, REAL.

    Runs normalize_champions end-to-end in a sandbox where the champion authority
    names a strategy whose frozen manifest declares surface: diagnostic. Both modes
    must exit red, the bundle manifest must be forced to archived, and the refreshed
    registry must carry surface=diagnostic + status=archived — i.e. the contradiction
    is not just printed, it is neutralized in the artifact the dashboard serves.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "normalize_champions_sandbox",
        ROOT / "scripts" / "pipeline" / "normalize_champions.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    frozen = tmp_path / "frozen"
    frozen.mkdir()
    (frozen / "aaa.yaml").write_text(
        yaml.safe_dump({"strategy_id": "diag_x", "surface": "diagnostic"}),
        encoding="utf-8")

    public = tmp_path / "public"
    bundle = public / "strategies" / "diag_x"
    bundle.mkdir(parents=True)
    manifest = {
        "strategy_id": "diag_x", "asset_id": "aaa", "symbol": "AAA/USD",
        "chart_symbol": "AAAUSD", "display_name": "Diagnostic X",
        "pipeline_type": "rule_based", "timeframe": "weekly",
        "status": "experimental", "backtests": [], "model_versions": [],
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (public / "registry.json").write_text(json.dumps({
        "generated_at": "sandbox", "assets": [],
        "strategies": [{"strategy_id": "diag_x", "asset_id": "aaa",
                        "status": "experimental"}],
        "default": {"asset_id": "aaa", "strategy_id": "diag_x"},
    }), encoding="utf-8")

    monkeypatch.setattr(mod, "FROZEN_MANIFESTS", frozen)
    monkeypatch.setattr(mod, "PUBLIC_DATA", public)
    monkeypatch.setattr(mod, "CHAMPION_BY_ASSET", {"aaa": "diag_x"})

    assert mod.normalize(check_only=True) != 0, (
        "check mode must exit red when a diagnostic surface is champion")
    assert mod.normalize(check_only=False) != 0, (
        "enforce mode must still exit red: a diagnostic champion is a contradiction "
        "to surface, not a state to normalize into silence")

    after = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert after["status"] == "archived", "diagnostic surface must be forced to archived"
    assert after.get("surface") == "diagnostic", (
        "enforce must stamp the frozen surface into the bundle manifest")

    reg = json.loads((public / "registry.json").read_text(encoding="utf-8"))
    entry = next(s for s in reg["strategies"] if s["strategy_id"] == "diag_x")
    assert entry.get("surface") == "diagnostic" and entry.get("status") == "archived", (
        f"registry must be refreshed with the neutralized truth, got {entry!r} — "
        "the dashboard reads registry.json, not the frozen YAMLs"
    )


def test_surface_contract_is_mirrored_in_typescript():
    """C-005 mirror rule: the TS contracts must carry the same optional surface field.

    Mirror map (contract-change skill): strategy_schema.py ↔ strategy.contract.ts and
    strategy_manifest.py ↔ strategy-manifest.contract.ts — same commit, both sides.
    """
    import re

    dash = ROOT / "usdcop-trading-dashboard" / "lib" / "contracts"
    union = re.compile(r"StrategySurface\s*=\s*'action'\s*\|\s*'diagnostic'")
    field = re.compile(r"^\s*surface\?\s*:\s*StrategySurface", re.M)
    for name in ("strategy.contract.ts", "strategy-manifest.contract.ts"):
        src = (dash / name).read_text(encoding="utf-8")
        assert union.search(src), (
            f"{name}: missing `type StrategySurface = 'action' | 'diagnostic'` (C-005)")
        assert field.search(src), (
            f"{name}: missing optional `surface?: StrategySurface` field (C-005)")


def _load_contract_module():
    """Load src/contracts/strategy_manifest.py by path (the module is a JSON-only leaf;
    importing it via `src.contracts` would eager-import the ML stack)."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "strategy_manifest_under_test", ROOT / "src" / "contracts" / "strategy_manifest.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclass processing requires the module registered
    spec.loader.exec_module(mod)
    return mod


def _load_normalize_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "normalize_champions_sandbox2",
        ROOT / "scripts" / "pipeline" / "normalize_champions.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dataclasses_reject_unknown_surface_fail_closed():
    """BL-13/C-005 remedio-2 (Codex hallazgo 1): construction is the wall.

    An unknown surface must raise ValueError at CONSTRUCTION time in both
    StrategyBundleManifest and RegistryStrategyEntry; absence/None keeps the
    ACKed legacy semantics (-> "action"). 'banana' -> 'action' must be impossible.
    """
    sm = _load_contract_module()
    base = dict(
        strategy_id="s1", asset_id="usdcop", symbol="USD/COP", chart_symbol="USDCOP",
        display_name="S1", pipeline_type="rule_based", timeframe="weekly", status="paper",
    )
    with pytest.raises(ValueError, match="surface"):
        sm.StrategyBundleManifest(**base, surface="unknown_surface")
    with pytest.raises(ValueError, match="surface"):
        sm.RegistryStrategyEntry(
            strategy_id="s1", asset_id="usdcop", status="paper", display_name="S1",
            pipeline_type="rule_based", timeframe="weekly",
            manifest="strategies/s1/manifest.json", surface="unknown_surface")
    with pytest.raises(ValueError, match="surface"):
        sm.StrategyBundleManifest.from_dict({**base, "surface": "banana"})

    # ACKed legacy semantics: absence (or None) -> "action", still constructs.
    assert sm.StrategyBundleManifest(**base).surface == "action"
    assert sm.StrategyBundleManifest(**base, surface=None).surface == "action"
    assert sm.RegistryStrategyEntry(
        strategy_id="s1", asset_id="usdcop", status="paper", display_name="S1",
        pipeline_type="rule_based", timeframe="weekly",
        manifest="strategies/s1/manifest.json").surface == "action"


def test_registry_builder_raises_on_invalid_surface_never_coerces(tmp_path):
    """BL-13/C-005 remedio-2 (Codex hallazgo 1): RegistryBuilder must RAISE on an
    invalid manifest surface — never silently coerce it to 'action' and serve the
    strategy as tradeable."""
    sm = _load_contract_module()
    bundle = tmp_path / "strategies" / "bad_surface"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text(json.dumps({
        "strategy_id": "bad_surface", "asset_id": "usdcop", "symbol": "USD/COP",
        "chart_symbol": "USDCOP", "display_name": "Bad", "pipeline_type": "rule_based",
        "timeframe": "weekly", "status": "paper", "surface": "unknown_surface",
        "backtests": [], "model_versions": [],
    }), encoding="utf-8")

    builder = sm.RegistryBuilder(tmp_path, generated_at="2026-07-27T00:00:00Z")
    with pytest.raises(ValueError, match="surface"):
        builder.build(write_manifests=False)

    # Sanity: a valid manifest still builds and carries its surface through.
    (bundle / "manifest.json").write_text(json.dumps({
        "strategy_id": "bad_surface", "asset_id": "usdcop", "symbol": "USD/COP",
        "chart_symbol": "USDCOP", "display_name": "Bad", "pipeline_type": "rule_based",
        "timeframe": "weekly", "status": "archived", "surface": "diagnostic",
        "backtests": [], "model_versions": [],
    }), encoding="utf-8")
    idx = builder.build(write_manifests=False)
    assert idx.strategies[0].surface == "diagnostic"


def test_frozen_yaml_invalid_surface_exits_red_in_both_modes(tmp_path, monkeypatch):
    """BL-13/C-005 remedio-2 (Codex hallazgo 1): a frozen YAML whose surface is
    outside {action, diagnostic} is a hard ERROR (exit 1) in BOTH modes — not a
    row _frozen_surfaces silently skips. Enforce mode must not write anything."""
    mod = _load_normalize_module()

    frozen = tmp_path / "frozen"
    frozen.mkdir()
    (frozen / "bad.yaml").write_text(
        yaml.safe_dump({"strategy_id": "x_strat", "surface": "unknown_surface"}),
        encoding="utf-8")

    public = tmp_path / "public"
    bundle = public / "strategies" / "x_strat"
    bundle.mkdir(parents=True)
    manifest_body = json.dumps({
        "strategy_id": "x_strat", "asset_id": "aaa", "symbol": "AAA/USD",
        "chart_symbol": "AAAUSD", "display_name": "X", "pipeline_type": "rule_based",
        "timeframe": "weekly", "status": "experimental", "surface": "action",
        "backtests": [], "model_versions": [],
    })
    (bundle / "manifest.json").write_text(manifest_body, encoding="utf-8")
    (public / "registry.json").write_text(json.dumps({
        "generated_at": "sandbox", "assets": [],
        "strategies": [{"strategy_id": "x_strat", "asset_id": "aaa",
                        "status": "experimental", "surface": "action"}],
        "default": {"asset_id": "aaa", "strategy_id": "x_strat"},
    }), encoding="utf-8")

    monkeypatch.setattr(mod, "FROZEN_MANIFESTS", frozen)
    monkeypatch.setattr(mod, "PUBLIC_DATA", public)
    monkeypatch.setattr(mod, "CHAMPION_BY_ASSET", {"aaa": "x_strat"})

    assert mod.normalize(check_only=True) != 0, (
        "--check must exit red when a frozen YAML declares an unknown surface")
    assert mod.normalize(check_only=False) != 0, (
        "enforce must exit red too: an invalid frozen surface is fixed at the "
        "source, never normalized into silence")
    assert (bundle / "manifest.json").read_text(encoding="utf-8") == manifest_body, (
        "enforce must not rewrite bundles while the frozen authority is invalid")

    # Same sandbox with a VALID frozen surface: both modes go green again.
    (frozen / "bad.yaml").write_text(
        yaml.safe_dump({"strategy_id": "x_strat", "surface": "action"}),
        encoding="utf-8")
    assert mod.normalize(check_only=True) == 0
    assert mod.normalize(check_only=False) == 0


def test_ts_runtime_surface_validator_mirrors_python_whitelist():
    """K-024 remedio-2 (Codex hallazgo 2): parity is SEMANTIC, not just a closed
    union type. The TS contract must expose a RUNTIME validator (pattern of
    policy.contract.ts / forecast-output.contract.ts) whose whitelist is the SAME
    tuple as Python strategy_manifest.SURFACES, rejecting unknown values."""
    import re

    sm = _load_contract_module()
    dash = ROOT / "usdcop-trading-dashboard" / "lib" / "contracts"
    src = (dash / "strategy-manifest.contract.ts").read_text(encoding="utf-8")

    m = re.search(r"STRATEGY_SURFACES\s*=\s*\[([^\]]*)\]\s*as\s*const", src)
    assert m, (
        "strategy-manifest.contract.ts: missing runtime whitelist "
        "`export const STRATEGY_SURFACES = [...] as const` (K-024)")
    ts_values = tuple(re.findall(r"'([^']+)'", m.group(1)))
    assert ts_values == tuple(sm.SURFACES), (
        f"TS runtime whitelist {ts_values} != Python SURFACES {tuple(sm.SURFACES)} — "
        "same values, same order, both sides")

    fn = re.search(
        r"export function validateStrategySurface\s*\(raw: unknown\): string\[\]"
        r"(.*?)\n\}", src, re.S)
    assert fn, (
        "strategy-manifest.contract.ts: missing runtime validator "
        "`export function validateStrategySurface(raw: unknown): string[]` (K-024)")
    assert "STRATEGY_SURFACES" in fn.group(1), (
        "validateStrategySurface must consult the STRATEGY_SURFACES whitelist, "
        "not a re-typed copy")
    assert re.search(r"export function assertStrategySurface", src), (
        "strategy-manifest.contract.ts: missing fail-closed accessor "
        "assertStrategySurface (absence -> 'action', unknown -> throw)")

    # strategy.contract.ts re-exports the SAME validator (single runtime source).
    src2 = (dash / "strategy.contract.ts").read_text(encoding="utf-8")
    assert "validateStrategySurface" in src2 and "strategy-manifest.contract" in src2, (
        "strategy.contract.ts must re-export the runtime surface validator from "
        "strategy-manifest.contract (one whitelist, two entry points)")


def test_registry_champion_matches_manifest():
    reg = json.loads((ROOT / "usdcop-trading-dashboard/public/data/registry.json")
                     .read_text(encoding="utf-8"))
    live = {s["asset_id"]: s["strategy_id"] for s in reg["strategies"]
            if s.get("status") != "archived"}
    for asset, sid in _champions().items():
        if asset in live:
            assert live[asset] == sid, (
                f"{asset}: registry serves {live[asset]!r} but manifest/authority freeze {sid!r}"
            )


def test_manifest_files_are_tracked_in_git():
    """K-026: un freeze solo es real si TODOS sus files: estan en git.

    Un hash computado sobre archivos untracked es ficcion verificable solo en una
    maquina (hallazgo CXD-031: spx500 referenciaba policies/engine.py fuera del
    commit). Cada path de `files:` debe existir en `git ls-files`.
    """
    import subprocess
    tracked = set(subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines())
    for p in sorted(MANIFESTS.glob("*.yaml")):
        m = yaml.safe_load(p.read_text(encoding="utf-8"))
        for f in m.get("files", []):
            assert f in tracked, (
                f"{p.name}: files: entry {f!r} NO esta trackeado en git — el freeze "
                "es ficcion (K-026). Trackearlo legitimamente o refreeze autorizado."
            )


def test_fabric_contracts_ci_executes_feature_contract_wall():
    """The feature-catalog wall must remain wired into the always-run CI job.

    This guard deliberately lives outside ``test_feature_contracts.py``: placing it
    in the protected module would let one workflow edit remove both the wall and its
    guard at once.
    """
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "fabric-contracts.yml").read_text(
            encoding="utf-8"
        )
    )
    steps = workflow["jobs"]["python-contracts"]["steps"]
    commands = "\n".join(
        str(step.get("run", "")) for step in steps if isinstance(step, dict)
    )
    assert re.search(
        r"(?:^|\s)tests/regression/test_feature_contracts\.py(?:\s|$)", commands
    ), (
        "fabric-contracts.yml/python-contracts must execute the complete "
        "tests/regression/test_feature_contracts.py wall"
    )
