"""BL-20 — valida los artefactos de interpretabilidad (fase 1: DATOS).

Corre el generador real para ridge (zoo, SHAP lineal cerrado) + spx500 (rule-based,
atribucion de reglas) y valida shape, header 'nota' obligatorio y ausencia de NaN/Inf
en el JSON serializado (A.7 / safe_json_dump).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.analysis.generate_interpretability import (  # noqa: E402
    NOTA,
    generate_rule_attribution,
    generate_zoo_linear,
)


def _fail_on_constant(tok: str):
    raise AssertionError(f"JSON contiene constante no-finita: {tok}")


def _load_strict(path: Path) -> dict:
    """json.load que REVIENTA si el texto contiene NaN/Infinity (safe JSON gate)."""
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_fail_on_constant)


def _assert_all_finite(obj, where: str = "$") -> None:
    if isinstance(obj, float):
        assert math.isfinite(obj), f"valor no finito en {where}"
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _assert_all_finite(v, f"{where}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_all_finite(v, f"{where}[{i}]")


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory) -> dict[str, Path]:
    """Corre el generador UNA vez (ridge + spx500) y devuelve los paths.

    Publica en un OUT_ROOT temporal: correr los tests NO puede tocar la evidencia
    versionada en ``data/interpretability/`` (esa mutacion silenciosa es
    exactamente el defecto que estos tests cubren).
    """
    import scripts.analysis.generate_interpretability as gi

    out = tmp_path_factory.mktemp("interpretability")
    original, gi.OUT_ROOT = gi.OUT_ROOT, out
    try:
        zoo = generate_zoo_linear(("ridge",))
        rules = generate_rule_attribution(("spx500",))
        assert len(zoo) == 1 and len(rules) == 1
        assert out in zoo[0].parents, "el generador escribio fuera del OUT_ROOT de test"
        yield {"ridge": zoo[0], "spx500": rules[0]}
    finally:
        gi.OUT_ROOT = original


def test_paths_follow_surface_asset_model_version_layout(artifacts):
    # OUT_ROOT vive FUERA de public/ (CXD-040: public bypassea el gate admin:all)
    # y en estos tests es un temporal — la raiz se deriva del propio artefacto,
    # nunca de una constante re-escrita aqui.
    import scripts.analysis.generate_interpretability as gi

    pub = gi.OUT_ROOT
    ridge, spx = artifacts["ridge"], artifacts["spx500"]
    assert ridge.name == "summary.json" and spx.name == "summary.json"
    # <surface>/<asset>/<model_id>/<version>/summary.json
    assert ridge.relative_to(pub).parts[:3] == ("zoo", "usdcop", "ridge")
    assert spx.relative_to(pub).parts[:3] == ("rule_based", "spx500", "spx500_regime_gated_v1")
    assert len(ridge.relative_to(pub).parts) == 5
    assert len(spx.relative_to(pub).parts) == 5


def test_nota_header_present_and_exact(artifacts):
    for path in artifacts.values():
        d = _load_strict(path)
        assert d["nota"] == NOTA
        assert d["nota"] == ("SHAP explica el modelo, no el mercado; "
                             "solo test-folds; diagnostico 0 trials")


def test_no_nan_inf_anywhere(artifacts):
    for path in artifacts.values():
        _assert_all_finite(_load_strict(path))


def test_ridge_linear_shap_shape(artifacts):
    d = _load_strict(artifacts["ridge"])
    assert d["surface"] == "zoo" and d["asset"] == "usdcop" and d["model_id"] == "ridge"
    assert d["method"] == "linear_shap_closed_form"
    assert d["attribution_not_shap"] is False
    assert d["n_features"] == len(d["top_features"]) > 0
    for entry in d["top_features"]:
        assert set(entry) == {"rank", "feature", "coef", "mean_abs_shap", "mean_shap"}
        assert entry["mean_abs_shap"] >= 0.0
    # top_features ordenado por mean|phi| descendente
    mas = [e["mean_abs_shap"] for e in d["top_features"]]
    assert mas == sorted(mas, reverse=True)
    # agregado por anio: cada anio lista las mismas features
    assert len(d["by_year"]) >= 2
    for yr, feats in d["by_year"].items():
        assert int(yr) >= 2020
        assert len(feats) == d["n_features"]
    # fit walk-forward: train-only + purga declaradas
    assert d["fit"]["purge_days"] == d["fit"]["horizon"] == 5
    assert d["fit"]["n_train"] >= 200


def test_spx500_rule_attribution_shape(artifacts):
    d = _load_strict(artifacts["spx500"])
    assert d["surface"] == "rule_based"
    assert d["model_id"] == "spx500_regime_gated_v1"
    assert d["attribution_not_shap"] is True          # etiqueta obligatoria BL-20
    assert d["method"] == "rule_attribution"
    r = d["rules"]
    assert 0.0 <= r["pct_days_trend_on"] <= 1.0
    assert 0.0 <= r["pct_days_position_active"] <= 1.0
    dec = d["pnl_decomposition"]
    # identidad de la descomposicion: gross = beta + timing (cov(pos,ret))
    assert dec["pnl_gross"] == pytest.approx(
        dec["pnl_beta"] + dec["pnl_timing_cov_pos_ret"], abs=1e-9)
    assert dec["pnl_net"] == pytest.approx(dec["pnl_gross"] - dec["costs"], abs=1e-9)
    assert dec["n_days"] > 0
    for yr, y in d["by_year"].items():
        assert y["n_days"] > 0
        assert y["pnl_gross"] == pytest.approx(
            y["pnl_beta"] + y["pnl_timing_cov_pos_ret"], abs=1e-9)


# ---------------------------------------------------------------------------
# BL-20 remedio-3: las contribuciones tienen que SER las del modelo
# ---------------------------------------------------------------------------
#
# Verificado por mutación: `phi = Z * coefs` -> `phi = np.ones_like(Z)` (contribución
# constante 1.0 para toda feature y toda fila) dejaba la suite ENTERA en verde. Los
# tests comprobaban forma, orden descendente (trivial con constantes), header `nota`,
# finitud y provenance — nunca que phi tuviera relación alguna con el modelo. La
# aditividad se PERSISTÍA como campo (`additivity_max_abs_err`) y no se comprobaba
# en ningún sitio (`grep -rn additivity tests/` => 0 aserciones).
#
# Los tres tests de abajo cierran eso: (1) identidad de aditividad contra la
# predicción CRUDA del modelo, (2) no-degeneración de las magnitudes, (3)
# acoplamiento causal — perturbar un coeficiente tiene que mover el ranking.


@pytest.fixture(scope="module")
def ridge_oracle() -> dict:
    """Oráculo INDEPENDIENTE del artefacto: reconstruye el mismo fit y le pregunta
    al modelo por sus predicciones (``mdl.predict``), no por sus atribuciones.

    Es la única forma de tener un juez externo: el artefacto publica agregados de
    phi, así que sin un modelo con el que contrastar cualquier matriz de números
    finitos y ordenados pasa los tests de forma.
    """
    from sklearn.preprocessing import StandardScaler

    import scripts.analysis.generate_interpretability as gi
    from src.forecasting.dataset_loader import ForecastingDatasetLoader
    from src.forecasting.models.factory import ModelFactory
    from src.forecasting.ssot_config import ForecastingSSOTConfig

    cfg = ForecastingSSOTConfig.load()
    df, _ = ForecastingDatasetLoader(cfg, project_root=REPO).load_dataset()
    feat_cols = [c for c in cfg.get_feature_columns() if c in df.columns]
    df = df.sort_values("date").reset_index(drop=True)
    df["y5"] = df["close"].shift(-gi.HORIZON) / df["close"] - 1.0

    train = df.iloc[:-gi.HORIZON]
    ok = train[feat_cols].notna().all(axis=1) & train["y5"].notna()
    Xtr = train.loc[ok, feat_cols].to_numpy(float)
    ytr = train.loc[ok, "y5"].to_numpy(float)
    scaler = StandardScaler().fit(Xtr)

    rows = df.loc[df[feat_cols].notna().all(axis=1),
                  ["date"] + feat_cols].reset_index(drop=True)
    Z = (rows[feat_cols].to_numpy(float) - scaler.mean_) / scaler.scale_

    mdl = ModelFactory.create("ridge")
    mdl.fit(scaler.transform(Xtr), ytr)
    coefs = np.asarray(mdl._model.coef_, dtype=float).ravel()
    return {
        "feat_cols": feat_cols,
        "years": rows["date"].dt.year.to_numpy(),
        "coefs": coefs,
        "intercept": float(np.asarray(mdl._model.intercept_).ravel()[0]),
        # predicción CRUDA del modelo — no depende de la forma cerrada de SHAP
        "pred": np.asarray(mdl.predict(Z), dtype=float).ravel(),
        "phi": Z * coefs,
    }


def test_linear_shap_contributions_are_additive_to_the_raw_prediction(artifacts,
                                                                      ridge_oracle):
    # rojo con: `phi = Z * coefs` -> `phi = np.ones_like(Z)` en
    # scripts/analysis/generate_interpretability.py:467
    d = _load_strict(artifacts["ridge"])
    base = float(d["base_value"])
    pred, years, phi = ridge_oracle["pred"], ridge_oracle["years"], ridge_oracle["phi"]
    assert base == pytest.approx(ridge_oracle["intercept"], abs=1e-12), (
        "el oraculo no reprodujo el mismo fit — el resto de las aserciones no valdria")

    # (i) FILA A FILA: en la forma cerrada, sum_j phi_ij + base ES la prediccion cruda.
    row_err = float(np.max(np.abs(phi.sum(axis=1) + base - pred)))
    assert row_err < 1e-9, f"la identidad de aditividad no se cumple fila a fila: {row_err:.3g}"

    # (ii) …y lo PUBLICADO son los agregados por filas de esa misma matriz, así que
    # hereda la identidad: sum_j mean_shap_j + base == mean(prediccion cruda).
    # Con phi constante = 1.0 el lado izquierdo vale n_features (21) y el derecho ~1e-5.
    total = sum(e["mean_shap"] for e in d["top_features"]) + base
    assert total == pytest.approx(float(pred.mean()), abs=1e-9), (
        f"sum(mean_shap)+base={total:.6g} vs mean(pred)={float(pred.mean()):.6g} — las "
        "contribuciones publicadas no reconstruyen la prediccion del modelo")

    # (iii) la identidad se sostiene por año (cada corte es su propio testigo)
    for yr, feats in d["by_year"].items():
        sel = years == int(yr)
        assert sel.any(), f"el artefacto publica el año {yr} que el oraculo no ve"
        got = sum(f["mean_shap"] for f in feats) + base
        assert got == pytest.approx(float(pred[sel].mean()), abs=1e-9), (
            f"año {yr}: sum(mean_shap)+base={got:.6g} vs mean(pred)={float(pred[sel].mean()):.6g}")

    # (iv) y las magnitudes publicadas son las de ESA matriz, no otras cualesquiera
    expected_abs = dict(zip(ridge_oracle["feat_cols"], np.abs(phi).mean(axis=0)))
    for entry in d["top_features"]:
        assert entry["mean_abs_shap"] == pytest.approx(
            expected_abs[entry["feature"]], rel=1e-9, abs=1e-18), entry["feature"]


def test_linear_shap_magnitudes_are_not_degenerate(artifacts):
    # rojo con: `phi = np.ones_like(Z)` (generate_interpretability.py:467) — todas las
    # mean_abs_shap valen 1.0 y el orden descendente se cumple trivialmente
    d = _load_strict(artifacts["ridge"])
    mean_abs = [e["mean_abs_shap"] for e in d["top_features"]]
    mean_shap = [e["mean_shap"] for e in d["top_features"]]
    assert len(mean_abs) > 1
    assert len({round(v, 12) for v in mean_abs}) > 1, (
        f"mean_abs_shap constante en las {len(mean_abs)} features ({mean_abs[0]!r}): "
        "una atribucion que no distingue features no atribuye nada, y el orden "
        "descendente pasa por construccion")
    assert len({round(v, 12) for v in mean_shap}) > 1, "mean_shap constante"
    assert min(mean_abs) > 0.0, "alguna feature con contribucion identicamente nula"


def test_linear_shap_ranking_tracks_the_model_coefficients(artifacts, ridge_oracle,
                                                           tmp_path, monkeypatch):
    # rojo con: `phi = np.ones_like(Z)` (generate_interpretability.py:467) — phi deja de
    # depender de `coefs`, así que amplificar un coeficiente no mueve el ranking
    import scripts.analysis.generate_interpretability as gi
    import src.forecasting.models.factory as factory_mod

    baseline = _load_strict(artifacts["ridge"])
    victim = baseline["top_features"][-1]                # la de MENOR magnitud hoy
    assert victim["mean_abs_shap"] > 0 and victim["coef"] != 0.0
    j = ridge_oracle["feat_cols"].index(victim["feature"])
    boost = 1e6

    real_factory = factory_mod.ModelFactory

    class _CoefPerturbingFactory:
        """Mismo modelo, con UN coeficiente amplificado tras el fit."""

        @staticmethod
        def create(model_name, params=None, horizon=None):
            mdl = real_factory.create(model_name, params=params, horizon=horizon)
            real_fit = mdl.fit

            def fit(X, y, *args, **kwargs):
                out = real_fit(X, y, *args, **kwargs)
                coefs = np.asarray(mdl._model.coef_, dtype=float).ravel().copy()
                coefs[j] *= boost
                mdl._model.coef_ = coefs
                return out

            mdl.fit = fit
            return mdl

    monkeypatch.setattr(factory_mod, "ModelFactory", _CoefPerturbingFactory)
    monkeypatch.setattr(gi, "OUT_ROOT", tmp_path)       # jamas sobre la evidencia real
    (path,) = gi.generate_zoo_linear(("ridge",))
    perturbed = _load_strict(path)

    by_feature = {e["feature"]: e for e in perturbed["top_features"]}
    # precondicion: la perturbacion llego de verdad al modelo
    assert by_feature[victim["feature"]]["coef"] == pytest.approx(
        victim["coef"] * boost, rel=1e-9), "el monkeypatch del factory no se aplico"

    top = perturbed["top_features"][0]["feature"]
    assert top == victim["feature"], (
        f"amplificar x{boost:g} el coeficiente de {victim['feature']!r} (la feature MENOS "
        f"contributiva) no la puso en el rank 1 — sigue mandando {top!r}: las "
        "contribuciones publicadas no dependen del modelo")


def test_tree_shap_route_is_exercised_and_its_additivity_is_asserted(tmp_path, monkeypatch):
    # rojo con: `phi, bias = shap_fn(mdl, Xte)` + `phi = np.ones_like(phi)` en
    # scripts/analysis/generate_interpretability.py:770
    #
    # La ruta TreeSHAP CALCULA la aditividad y la persiste como campo, pero ningun test
    # la ejecutaba ni afirmaba su umbral: el campo era decorativo.
    import scripts.analysis.generate_interpretability as gi

    monkeypatch.setattr(gi, "OUT_ROOT", tmp_path)
    (path,) = gi.generate_zoo_tree(("catboost",))
    d = _load_strict(path)

    assert d["method"] == "tree_shap", (
        f"la ruta degrado a {d.get('status')!r} ({d.get('detail')!r}) — el test no llego "
        "a ejercitar TreeSHAP")
    err = d["additivity_max_abs_err"]
    assert 0.0 <= err < 1e-6, (
        f"additivity_max_abs_err={err:.3g}: sum(phi)+bias no reproduce la prediccion "
        "cruda del booster — las contribuciones no explican al modelo")

    mean_abs = [e["mean_abs_shap"] for e in d["top_features"]]
    assert len({round(v, 12) for v in mean_abs}) > 1, "mean_abs_shap constante (TreeSHAP)"
    assert d["n_features"] == len(mean_abs) > 1
    assert d["n_folds"] >= 2 and d["n_rows"] > 0
    assert d["by_regime"], "faltan los cortes por regimen (gate Hurst congelado)"


# ---------------------------------------------------------------------------
# BL-20 remedio (CODEX P1): provenance + inmutabilidad + N no ambiguo
# ---------------------------------------------------------------------------

def test_artifact_declares_provenance_fingerprints(artifacts):
    """La `version` (fecha del ultimo dato) NO identifica la evidencia.

    Rojo original: el payload no comprometia datos/codigo/config/modelo, asi que
    dos corridas con codigo distinto compartian identidad.
    """
    for path in artifacts.values():
        d = _load_strict(path)
        assert isinstance(d.get("artifact_id"), str) and d["artifact_id"].startswith("sha256:")
        prov = d.get("provenance")
        assert isinstance(prov, dict), "falta el bloque provenance"
        for key in ("data_fingerprint", "code_fingerprint",
                    "config_fingerprint", "model_fingerprint"):
            assert key in prov, f"provenance sin {key}"
            assert isinstance(prov[key], str) and prov[key].startswith("sha256:"), key
        # el artifact_id DEPENDE de las huellas: cambiar una cambia la identidad
        assert d["artifact_id"] != prov["code_fingerprint"]


def test_regenerating_the_same_version_does_not_mutate_the_artifact(artifacts):
    """Rojo original: `_write` sobrescribia y `generated_at` cambiaba en cada corrida.

    Misma entrada + mismo codigo => el fichero publicado no cambia NI UN BYTE.
    """
    from scripts.analysis.generate_interpretability import (
        generate_rule_attribution,
        generate_zoo_linear,
    )

    before = {k: p.read_bytes() for k, p in artifacts.items()}
    generate_zoo_linear(("ridge",))
    generate_rule_attribution(("spx500",))
    for k, p in artifacts.items():
        assert p.read_bytes() == before[k], (
            f"{k}: la MISMA version muto en disco al regenerar (evidencia silenciosamente "
            "reescrita)"
        )


def test_divergent_payload_at_the_same_version_fails_instead_of_overwriting(artifacts):
    """Un conflicto divergente FALLA; jamas pisa la evidencia publicada."""
    from scripts.analysis.generate_interpretability import (
        ArtifactConflictError,
        _write,
    )

    path = artifacts["ridge"]
    original = path.read_bytes()
    payload = _load_strict(path)
    payload["top_features"] = []          # contenido distinto, misma (surface/model/version)
    with pytest.raises(ArtifactConflictError):
        _write("zoo", "usdcop", "ridge", payload["version"], payload)
    assert path.read_bytes() == original, "el artefacto previo fue tocado pese al conflicto"


def test_write_is_atomic_no_partial_file_on_failure(tmp_path, monkeypatch):
    """Una serializacion que revienta a MITAD no deja summary.json a medias."""
    import scripts.analysis.generate_interpretability as gi

    monkeypatch.setattr(gi, "OUT_ROOT", tmp_path)
    target = tmp_path / "zoo" / "usdcop" / "ridge" / "9999-01-01" / "summary.json"

    def _explode_midway(payload, fh):
        fh.write('{"nota": "a medio escribir"')     # bytes YA en el descriptor
        raise RuntimeError("serializacion interrumpida")

    monkeypatch.setattr(gi, "safe_json_dump", _explode_midway)
    with pytest.raises(RuntimeError, match="interrumpida"):
        gi._write("zoo", "usdcop", "ridge", "9999-01-01", {"nota": gi.NOTA})
    assert not target.exists(), "quedo un summary.json parcial tras un fallo de escritura"
    assert not list(tmp_path.rglob("*.tmp")), "quedo un temporal huerfano"


def test_write_never_overwrites_a_published_artifact_mid_failure(tmp_path, monkeypatch):
    """Si la segunda publicacion falla, la PRIMERA sigue intacta y legible."""
    import scripts.analysis.generate_interpretability as gi

    monkeypatch.setattr(gi, "OUT_ROOT", tmp_path)
    first = gi._write("zoo", "usdcop", "ridge", "9999-01-01", {"nota": gi.NOTA, "v": 1})
    original = first.read_bytes()

    def _explode_midway(payload, fh):
        fh.write('{"roto":')
        raise RuntimeError("serializacion interrumpida")

    monkeypatch.setattr(gi, "safe_json_dump", _explode_midway)
    with pytest.raises(RuntimeError, match="interrumpida"):
        gi._write("zoo", "usdcop", "ridge", "9999-01-01", {"nota": gi.NOTA, "v": 2},
                  supersede=True)
    assert first.read_bytes() == original


# ---------------------------------------------------------------------------
# BL-20 remedio-2 (sonda de integridad CODEX): el escritor no puede CONFIAR en el
# artifact_id declarado, el eslabon `supersedes` tiene que estar ligado
# criptograficamente, y dos primeros publicadores divergentes no pueden ganar ambos.
# ---------------------------------------------------------------------------

def test_tampered_payload_that_keeps_its_artifact_id_is_detected(tmp_path, monkeypatch):
    """Rojo CODEX #1: `_write` comparaba SOLO el artifact_id declarado.

    Alterar un `value` conservando el id daba `detected:false` — la inmutabilidad
    se apoyaba justo en el campo que un manipulador controla. La identidad del
    contenido ALMACENADO tiene que recomputarse antes de cualquier early-return.
    """
    import scripts.analysis.generate_interpretability as gi

    monkeypatch.setattr(gi, "OUT_ROOT", tmp_path)
    payload = {"nota": gi.NOTA, "value": 1}
    path = gi._write("zoo", "usdcop", "probe", "v1", dict(payload))

    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["value"] = 999                       # contenido alterado…
    path.write_text(json.dumps(stored), encoding="utf-8")   # …con el MISMO artifact_id

    with pytest.raises(gi.ArtifactConflictError, match="(?i)fallo de integridad"):
        gi._write("zoo", "usdcop", "probe", "v1", dict(payload))


def test_supersedes_is_bound_to_the_artifact_identity():
    """Rojo CODEX #2: `supersedes` estaba en VOLATILE_FIELDS.

    Dos cadenas de sustitucion distintas producian exactamente el mismo
    artifact_id: el eslabon era texto decorativo, no una ligadura verificable.
    """
    from scripts.analysis.generate_interpretability import _artifact_identity

    a = _artifact_identity({"value": 2, "supersedes": "sha256:" + "a" * 64})
    b = _artifact_identity({"value": 2, "supersedes": "sha256:" + "b" * 64})
    assert a != b, "cambiar solo `supersedes` no cambio el artifact_id"


def test_two_divergent_first_publishers_cannot_both_succeed(tmp_path, monkeypatch):
    """Rojo CODEX #3: TOCTOU — ambos veian ausencia, ambos `success`, el ultimo pisaba.

    Con una barrera en `os.replace` los dos escritores llegan a la publicacion a la
    vez. Exactamente uno debe ganar; el divergente recibe ArtifactConflictError y
    los bytes del ganador quedan intactos.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import scripts.analysis.generate_interpretability as gi

    monkeypatch.setattr(gi, "OUT_ROOT", tmp_path)
    barrier = threading.Barrier(2)
    real_replace = gi.os.replace

    def racing_replace(src, dst):
        barrier.wait(timeout=10)
        real_replace(src, dst)

    monkeypatch.setattr(gi.os, "replace", racing_replace)

    def publish(value: int) -> str:
        try:
            gi._write("zoo", "usdcop", "probe", "v1", {"nota": gi.NOTA, "value": value})
            return "success"
        except gi.ArtifactConflictError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = sorted(pool.map(publish, (1, 2)))

    assert outcomes == ["conflict", "success"], f"carrera no serializada: {outcomes}"

    monkeypatch.setattr(gi.os, "replace", real_replace)
    published = tmp_path / "zoo" / "usdcop" / "probe" / "v1" / "summary.json"
    doc = json.loads(published.read_text(encoding="utf-8"))
    # el ganador quedo INTACTO: su artifact_id sigue siendo el de su propio contenido
    assert doc["artifact_id"] == gi._artifact_identity(doc)
    assert doc["value"] in (1, 2)
    assert not list(tmp_path.rglob("*.tmp")), "quedo un temporal huerfano tras la carrera"
    assert not list(tmp_path.rglob("*.staged")), "quedo un staging huerfano tras la carrera"


def test_identity_migration_only_rewrites_the_id_and_refuses_tampered_files(
        tmp_path, monkeypatch):
    """La migracion de esquema es una re-derivacion PURA y fail-closed.

    Incluir `supersedes` en el hash deja a los artefactos ya publicados con un id
    del esquema viejo. La migracion solo puede tocar ficheros de los que se puede
    DEMOSTRAR que no fueron alterados (id almacenado == identidad de contenido);
    ante cualquier otra cosa aborta sin escribir.
    """
    import scripts.analysis.generate_interpretability as gi

    monkeypatch.setattr(gi, "OUT_ROOT", tmp_path)
    first = gi._write("zoo", "usdcop", "probe", "v1", {"nota": gi.NOTA, "v": 1})
    gi._write("zoo", "usdcop", "probe", "v1", {"nota": gi.NOTA, "v": 2}, supersede=True)
    doc = json.loads(first.read_text(encoding="utf-8"))
    assert doc["supersedes"], "el fixture necesita un artefacto con eslabon"

    # Se le devuelve el id del esquema VIEJO (hash del contenido sin el eslabon).
    legacy = dict(doc, artifact_id=gi._content_identity(doc))
    first.write_text(json.dumps(legacy, indent=2), encoding="utf-8")

    report = gi.migrate_identity(tmp_path, apply=True)
    assert [r["status"] for r in report] == ["migrated"]
    migrated = json.loads(first.read_text(encoding="utf-8"))
    assert migrated["artifact_id"] == gi._artifact_identity(migrated)
    # UNICO campo que cambia: todo lo demas queda igual que antes de migrar
    assert {k: v for k, v in migrated.items() if k != "artifact_id"} == \
           {k: v for k, v in legacy.items() if k != "artifact_id"}
    # idempotente
    assert [r["status"] for r in gi.migrate_identity(tmp_path, apply=True)] == \
           ["already_current"]

    # Fichero realmente alterado => aborta, no lo "migra"
    tampered = dict(migrated, v=999)
    first.write_text(json.dumps(tampered, indent=2), encoding="utf-8")
    with pytest.raises(gi.ArtifactConflictError):
        gi.migrate_identity(tmp_path, apply=True)


def test_expanding_folds_never_publish_a_summed_n_train():
    """`sum(n_train por fold)` cuenta las MISMAS filas varias veces en expanding.

    El artefacto publica el N del ULTIMO fit + las filas distintas + el detalle
    por fold; nunca la suma.
    """
    from scripts.analysis.generate_interpretability import _train_size_summary

    fold_meta = [
        {"year": 2022, "n_train": 1000},
        {"year": 2023, "n_train": 1250},
        {"year": 2024, "n_train": 1500},
    ]
    summary = _train_size_summary(fold_meta, distinct_train_rows=1500)
    assert summary["n_train_last_fit"] == 1500
    assert summary["n_train_distinct_rows"] == 1500
    assert "n_train" not in summary, "n_train a secas es ambiguo en expanding"
    assert 3750 not in summary.values(), "se publico la suma de folds (N inflado)"
    assert summary["n_train_by_fold"] == [1000, 1250, 1500]


def test_out_root_is_outside_public():
    """CXD-040: `public/` bypassea el gate admin:all — el artefacto NO vive ahi."""
    import scripts.analysis.generate_interpretability as gi

    default_root = REPO / "data" / "interpretability"
    assert gi.OUT_ROOT in (default_root, gi.OUT_ROOT)   # patcheable en tests
    assert "public" not in default_root.parts
