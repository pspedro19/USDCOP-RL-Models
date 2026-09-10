# -*- coding: utf-8 -*-
"""Candados BL-09-r2 / BL-11-r2 / BL-12-r3 — gobernanza del ledger doble FT/AT.

Remedia los RECHAZOS textuales de Codex:

* BL-09/BL-11: *"10/10 tests pero 8/10 familias legacy exentas, sin DSR family/cluster/global
  ni gate; 218/237 cutoff null; `trend_regime` comprime 94 trials a 8 celdas."*
* BL-12-r2: *"el SHA sella ledger 237 / USD-COP 109 frente a header 111; el guard de prosa
  es evadible por case/orden."*

**Cada test de mutación es la demostración del ROJO**: parte del árbol REAL (verde), lo rompe
en `tmp_path` de una sola forma, y exige que el validador lo detecte. Un candado que nunca
estuvo rojo no es candado. Ninguna mutación toca `registries/ledger.jsonl` — el contenido
BL-10 de Codex es append-only e intocable; se copian los YAML/README a tmp.
"""
from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import pytest
import yaml

from scripts.validation import check_trial_ledger as ledger
from scripts.validation import report_ledger_dsr as dsr_report

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def records():
    return ledger.load_ledger(ledger.LEDGER_PATH)


@pytest.fixture()
def families_copy(tmp_path):
    """Copia editable de registries/families/ (el original NUNCA se muta)."""
    target = tmp_path / "families"
    shutil.copytree(ledger.FAMILIES_DIR, target)
    return target


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _dump(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


# =========================================================================================
# VERDE — estado actual del árbol
# =========================================================================================

def test_ledger_bl10_content_is_intact(records):
    """El contenido de Codex (BL-10) no se toca.

    ACTUALIZADO 2026-08-25 (H-TESIS-RL-01): 239 -> **241** globales, 184 -> **186** AT,
    COP 111 -> **113**. Los dos trials nuevos son `AT-0185` (`ppo_regime`) y `AT-0186`
    (`ppo_backbone`), familia `usdcop_rl_intraday`, cobrados por evaluar ambas
    configuraciones sobre el bloque de SELECCION.

    Estas constantes estan a proposito escritas a mano: obligan a que cualquier variacion
    del ledger sea un cambio DELIBERADO con su justificacion, en vez de un recuento
    automatico que absorbe en silencio un trial que alguien olvido registrar.
    """
    lineage = Counter(record["trial_id"][:2] for record in records)
    per_asset = Counter(record["asset"] for record in records)
    assert len(records) == 243
    assert lineage["FT"] == 55 and lineage["AT"] == 188   # +2 H-TESIS-RL-01, +2 carril forward
    assert per_asset == {"usdcop": 115, "xauusd": 77, "btcusdt": 34, "spx500": 17}
    # Lo que este bloque protege es que el ledger sea APPEND-ONLY: los asientos de
    # reconciliación de Codex siguen ahí, en su sitio y encadenados. Ya no son los últimos
    # —H-TESIS-RL-01 añadió AT-0185/AT-0186 el 2026-08-25— así que se comprueba su posición
    # explícita en vez de «los dos últimos», que era una forma de decir «nadie ha añadido
    # nada» y habría bloqueado el primer trial legítimo.
    assert [record["trial_id"] for record in records[237:239]] == ["FT-0054", "FT-0055"]
    assert [record["trial_id"] for record in records[239:241]] == ["AT-0185", "AT-0186"]
    assert [record["trial_id"] for record in records[-2:]] == ["AT-0187", "AT-0188"]
    assert ledger.check_hash_chain(records) == []


def test_hardened_validator_is_green():
    assert ledger.run_all_checks() == []


def test_dsr_gate_script_is_green():
    assert dsr_report.main() == 0


def test_every_ledger_family_has_a_declared_yaml(records):
    """BL-11-r2: CERO exenciones legacy (antes 8/10 familias no tenían YAML)."""
    declared = set(ledger.load_families())
    ledger_families = set(record["family"] for record in records)
    assert ledger_families == declared
    assert len(declared) == 12   # +usdcop_rl_intraday, +usdcop_llm_forward


def test_no_ledger_row_has_an_unclassified_null_cutoff(records):
    """BL-09-r2: los 218 `cutoff: null` quedan clasificados, no ignorados."""
    assert ledger.check_cutoff_classification(records) == []
    nulls = [r for r in records if r.get("cutoff") is None]
    assert len(nulls) == 218, "si el ledger crece, la clasificación debe crecer con él"


def test_trend_regime_granularity_is_declared_cell_by_cell():
    """BL-11-r2: 94 trials ya no son 8 celdas opacas — cada celda declara su granularidad."""
    family = _load(ledger.FAMILIES_DIR / "trend_regime.yaml")
    kinds = Counter(cell["cell_kind"] for cell in family["cells"])
    assert kinds["atomic"] == 6
    assert kinds["grid_block"] == 1
    assert kinds["estimated_block"] == 1
    grid = next(c for c in family["cells"] if c["cell_kind"] == "grid_block")
    product = len(grid["grid_axes"]["vol_target"]) * len(grid["grid_axes"]["ma_window"])
    assert product + len(grid["extra_variants"]) == grid["n_trials"] == 14
    block = next(c for c in family["cells"] if c["cell_kind"] == "estimated_block")
    assert block["decomposable"] is False and block["granularity_note"].strip()
    assert (ROOT / block["provenance_source"]).exists()


def test_readme_declared_totals_match_the_ledger(records):
    """BL-12-r3: la cabecera publicada es maquinal y coincide con el ledger."""
    declared = ledger.parse_declared_totals()
    assert declared["n_global"] == len(records) == 243
    assert declared["per_asset"]["usdcop"] == 115
    assert ledger.check_declared_totals(records) == []


def test_vol_forecast_families_deflate_at_cluster_level():
    """BL-11-r2: la misma mecánica en 4 activos deflacta con el cluster, no con 4 N=1."""
    families = ledger.load_families()
    vol = [f for f in families.values()
           if f.get("hypothesis_key") == "har_rv_5d_vs_ewma_vol_forecast"]
    assert len(vol) == 4
    assert all(f["deflation_scope"] == "cluster" for f in vol)


def test_ft_to_at_wall_has_a_real_provenance_case():
    """BL-12: `vol_sizing` es el caso REAL de muralla cruzada con FT trazados."""
    family = _load(ledger.FAMILIES_DIR / "vol_sizing.yaml")
    assert family["provenance"]["crosses_wall"] is True
    assert family["provenance"]["forecast_trial_ids"] == ["FT-0050", "FT-0051"]
    by_id = {r["trial_id"]: r for r in ledger.load_ledger(ledger.LEDGER_PATH)}
    for trial_id in family["provenance"]["forecast_trial_ids"]:
        assert by_id[trial_id]["kind"] == "forecast"
        assert by_id[trial_id]["cluster"] == family["cluster_id"]  # entran al N_cluster


def test_three_dsr_levels_are_actually_computed_for_the_production_track(records):
    """BL-09-r2: DSR family/cluster/global COMPUTADOS (no informativos) y por debajo del bar.

    Se reporta tal cual sale: smart_simple_v11 NO pasa el bar 0.95 con ningún N ni sigma.
    """
    families = ledger.load_families()
    family = families["smart_simple"]
    n_family, n_cluster, n_global, cluster_of = dsr_report.final_counts(records)
    candidate = family["governance"]["candidates"][0]
    result = dsr_report.compute_candidate_dsr(
        candidate, n_family["smart_simple"], n_cluster[cluster_of["smart_simple"]], n_global,
        family["deflation_scope"],
    )
    assert result["method"] == "computed"
    assert result["n_x3"] == {"family": 60, "cluster": 138, "global": 243}
    assert result["dsr_family"] == 0.6368
    assert result["dsr_cluster"] == 0.6212
    assert result["dsr_global"] == pytest.approx(0.6112, abs=5e-4)  # N_global crece, deflacta mas
    assert result["claim_allowed"] is False
    # el DSR es no-creciente en n_trials: family >= cluster >= global
    assert result["dsr_family"] >= result["dsr_cluster"] >= result["dsr_global"]
    # y el gate usa la sigma MENOS favorable de la rejilla
    assert result["dsr_family"] == min(result["dsr_grid"]["family"].values())


def test_dsr_bar_is_not_negotiable():
    assert dsr_report.DSR_BAR == 0.95


# =========================================================================================
# ROJO — mutaciones: cada una demuestra que el candado muerde
# =========================================================================================

def test_red_missing_family_yaml_is_rejected(records, families_copy):
    (families_copy / "smart_simple.yaml").unlink()
    errors = ledger.check_family_declaration_coverage(records, families_copy)
    assert any("smart_simple" in e and "NO existe" in e for e in errors)


def test_red_cell_without_cell_kind_is_rejected(records, families_copy):
    path = families_copy / "vol_sizing.yaml"
    family = _load(path)
    del family["cells"][0]["cell_kind"]
    _dump(path, family)
    errors = ledger.check_cell_granularity(records, families_copy)
    assert any("cell_kind" in e for e in errors)


def test_red_documented_cell_disguised_as_estimated_block_is_rejected(records, families_copy):
    """Una celda 100% documentada no puede esconderse como bloque no descomponible."""
    path = families_copy / "vol_sizing.yaml"
    family = _load(path)
    family["cells"][0]["cell_kind"] = "estimated_block"
    family["cells"][0]["decomposable"] = False
    family["cells"][0]["granularity_note"] = "excusa"
    family["cells"][0]["provenance_source"] = "registries/README.md"
    _dump(path, family)
    errors = ledger.check_cell_granularity(records, families_copy)
    assert any("NO puede ser" in e and "estimated_block" in e for e in errors)


def test_red_grid_axes_that_do_not_multiply_are_rejected(records, families_copy):
    path = families_copy / "trend_regime.yaml"
    family = _load(path)
    grid = next(c for c in family["cells"] if c.get("cell_kind") == "grid_block")
    grid["grid_axes"]["ma_window"] = [200]          # 4x1 + 2 = 6 != 14
    _dump(path, family)
    errors = ledger.check_cell_granularity(records, families_copy)
    assert any("grid_axes cubre" in e for e in errors)


def test_red_estimated_block_with_fake_provenance_source_is_rejected(records, families_copy):
    path = families_copy / "smart_simple.yaml"
    family = _load(path)
    family["cells"][0]["provenance_source"] = ".claude/no/existe.md"
    _dump(path, family)
    errors = ledger.check_cell_granularity(records, families_copy)
    assert any("no existe en el árbol" in e for e in errors)


def test_red_null_cutoff_without_classification_is_rejected(records, families_copy):
    """El agujero exacto del rechazo: 218 cutoff null pasando en silencio."""
    path = families_copy / "smart_simple.yaml"
    family = _load(path)
    del family["cells"][0]["cutoff_class"]
    _dump(path, family)
    errors = ledger.check_cutoff_classification(records, families_copy)
    assert any("cutoff nulo" in e and "cutoff_class" in e for e in errors)


def test_red_legacy_irrecoverable_on_documented_rows_is_rejected(records, families_copy):
    """Lo documentado tiene cutoff recuperable: no se puede declarar irrecuperable."""
    path = families_copy / "vol_sizing.yaml"
    family = _load(path)
    for cell in family["cells"]:
        cell["cutoff_class"] = "legacy_irrecoverable"
    _dump(path, family)
    errors = ledger.check_cutoff_classification(records, families_copy)
    assert any("PROHIBIDO para filas" in e for e in errors)


def test_red_cutoff_evidence_that_does_not_mention_the_hypothesis_is_rejected(
    records, families_copy
):
    path = families_copy / "vol_sizing.yaml"
    family = _load(path)
    family["cells"][0]["hypothesis_id"] = "H-INEXISTENTE-99"
    _dump(path, family)
    errors = ledger.check_cutoff_classification(records, families_copy)
    assert any("no aparece en" in e for e in errors)


def test_red_family_without_provenance_block_is_rejected(records, families_copy):
    path = families_copy / "exposure_engine.yaml"
    family = _load(path)
    del family["provenance"]
    _dump(path, family)
    errors = ledger.check_provenance_wall(records, families_copy)
    assert any("falta el bloque `provenance`" in e for e in errors)


def test_red_inheriting_forecasts_while_claiming_not_to_cross_the_wall_is_rejected(
    records, families_copy
):
    """ADR-0022 §2 estructural: heredar FT ES cruzar la muralla (+1 AT)."""
    path = families_copy / "exposure_engine.yaml"
    family = _load(path)
    family["provenance"]["forecast_trial_ids"] = ["FT-0052"]
    _dump(path, family)
    errors = ledger.check_provenance_wall(records, families_copy)
    assert any("heredar FT ES cruzar la muralla" in e for e in errors)


def test_red_provenance_from_another_cluster_is_rejected(records, families_copy):
    """Si el FT heredado vive en otro cluster, NO entra en el N_cluster: fraude de N."""
    path = families_copy / "vol_sizing.yaml"
    family = _load(path)
    family["provenance"]["inherits_from_family"] = "usdcop_direction"  # cluster ml_meta
    _dump(path, family)
    errors = ledger.check_provenance_wall(records, families_copy)
    assert any("NO entrarían en su N_cluster" in e for e in errors)


def test_red_provenance_citing_an_action_trial_as_forecast_is_rejected(records, families_copy):
    path = families_copy / "vol_sizing.yaml"
    family = _load(path)
    family["provenance"]["forecast_trial_ids"] = ["AT-0001"]
    _dump(path, family)
    errors = ledger.check_provenance_wall(records, families_copy)
    assert any("NO es kind=forecast" in e for e in errors)


def test_red_stale_header_237_109_is_rejected(records, tmp_path):
    """LA regresión de BL-12-r2: una cabecera stale contra el ledger real.

    Los numeros del nombre (237/109) son los del incidente original y se conservan como
    etiqueta historica; lo que el test inyecta es un desfase RELATIVO al ledger vigente,
    para que siga probando lo mismo cuando el ledger crece.
    """
    readme = tmp_path / "README.md"
    original = (ROOT / "registries" / "README.md").read_text(encoding="utf-8")
    readme.write_text(
        original.replace("n_global: 243", "n_global: 237").replace("usdcop: 115", "usdcop: 109"),
        encoding="utf-8",
    )
    errors = ledger.check_declared_totals(records, readme)
    assert any("n_global=237" in e for e in errors)
    assert any("per_asset.usdcop=109" in e for e in errors)


def test_red_header_evasion_by_case_or_order_no_longer_works(records, tmp_path):
    """El guard es ESTRUCTURAL: no hay mayúscula ni reordenación que lo esquive.

    El guard viejo era `re.findall(r"(\\d+)\\s+globales", body)` sobre prosa: bastaba escribir
    "Globales: 999" (case + orden invertido) para que no encontrara nada. Aquí el número vive
    en un bloque YAML delimitado y se compara campo a campo.
    """
    readme = tmp_path / "README.md"
    original = (ROOT / "registries" / "README.md").read_text(encoding="utf-8")
    readme.write_text(
        original.replace("n_global: 243", "N_GLOBAL: 243").replace("<!-- LEDGER-TOTALS", "<!-- ledger-totals"),
        encoding="utf-8",
    )
    errors = ledger.check_declared_totals(records, readme)
    assert errors, "renombrar/recapitalizar la cabecera debe FALLAR, nunca silenciar el check"


def test_red_deleting_the_totals_block_fails_closed(records, tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# sin bloque de totales\n", encoding="utf-8")
    errors = ledger.check_declared_totals(records, readme)
    assert any("LEDGER-TOTALS" in e for e in errors)


def test_red_splitting_a_shared_mechanic_into_per_asset_families_is_rejected(
    records, families_copy
):
    """FABRIC §31: dividir familias no lava multiplicidad."""
    path = families_copy / "usdcop_vol.yaml"
    family = _load(path)
    family["deflation_scope"] = "family"   # intento de deflactar con N=1
    _dump(path, family)
    errors = ledger.check_family_siblings(records, families_copy)
    assert any("deflation_scope debe ser 'cluster'" in e for e in errors)


def test_red_undeclared_sibling_is_rejected(records, families_copy):
    path = families_copy / "usdcop_vol.yaml"
    family = _load(path)
    family["sibling_families"] = []
    _dump(path, family)
    errors = ledger.check_family_siblings(records, families_copy)
    assert any("sibling_families declara []" in e for e in errors)


def test_red_claiming_edge_without_a_computed_dsr_above_the_bar_is_rejected(
    records, families_copy
):
    """El gate DSR es REAL: `claims_edge: true` sin DSR computado > 0.95 = violación."""
    path = families_copy / "trend_regime.yaml"
    family = _load(path)
    family["governance"]["claims_edge"] = True
    _dump(path, family)
    families = dsr_report.load_families(families_copy)
    errors = dsr_report.check_governance(families, records)
    assert any("claims_edge=True" in e and "COMPUTADO" in e for e in errors)


def test_red_claiming_edge_on_a_published_upper_bound_is_rejected(records, families_copy):
    """Una COTA publicada (N menor) jamás habilita un claim, por bonita que sea."""
    path = families_copy / "smart_simple.yaml"
    family = _load(path)
    family["governance"]["claims_edge"] = True
    family["governance"]["candidates"] = [
        {"id": "cota_bonita",
         "published_dsr": {"value": 0.99, "n_trials_used": 5, "source": "registries/README.md"}}
    ]
    _dump(path, family)
    families = dsr_report.load_families(families_copy)
    errors = dsr_report.check_governance(families, records)
    assert any("claims_edge=True" in e for e in errors)


def test_red_new_row_without_cutoff_is_rejected(records):
    """Fail-closed prospectivo (no muta el ledger: opera sobre copia en memoria)."""
    mutated = [dict(record) for record in records]
    mutated[-1]["env"] = "screening"
    mutated[-1]["cutoff"] = None
    errors = ledger.check_cutoff_classification(mutated, ledger.FAMILIES_DIR)
    assert any("exige cutoff explícito" in e for e in errors)


def test_red_new_non_legacy_cell_must_be_atomic(records, families_copy):
    mutated = [dict(record) for record in records]
    for record in mutated:
        if record["family"] == "smart_simple":
            record["env"] = "screening"
    errors = ledger.check_cell_granularity(mutated, families_copy)
    assert any("debe ser atomic" in e for e in errors)


def test_red_tampering_a_codex_ledger_row_breaks_the_chain(records):
    """El contenido de Codex es append-only: editarlo rompe la cadena (y este test lo prueba)."""
    mutated = [dict(record) for record in records]
    mutated[100]["result"] = "edge!"
    errors = ledger.check_hash_chain(mutated)
    assert any("line_hash no coincide" in e for e in errors)


def test_red_ledger_json_shape_is_stable():
    """La primera línea del ledger sigue siendo la génesis de Codex (hash 0*64)."""
    first = json.loads(ledger.LEDGER_PATH.read_text(encoding="utf-8").splitlines()[0])
    assert first["prev_hash"] == ledger.GENESIS_HASH
    assert first["trial_id"] == "FT-0001"
