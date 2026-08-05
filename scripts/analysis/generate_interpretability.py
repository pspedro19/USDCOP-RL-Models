"""BL-20 (parte DATOS) — artefactos de interpretabilidad por (surface, asset, model_id, version).

FABRIC Anexo A.7: "SHAP explica el modelo, no el mercado — sirve para rechazar modelos
absurdos, no para probar verdades". Este generador NO computa métricas de acierto ni
performance: 0 trials (diagnóstico declarado sobre congelados).

Fase 1 (este script):
  (a) Zoo COP — SHAP LINEAL cerrado para ridge / bayesian_ridge / ard. Para un modelo
      lineal sobre features estandarizadas, el valor SHAP exacto (features
      independientes) es phi_j = coef_j * (x_j - mu_j) / sigma_j — forma cerrada, sin
      instalar `shap`. SOLO TEST-FOLDS: EL MISMO walk-forward expanding anual que la
      ruta de árbol (`_annual_expanding_folds`, maquinaria compartida). Cortes:
      global + por año + POR RÉGIMEN (gate Hurst congelado), más kill-flags A.7.

      Corrección 2026-07-28 (defecto de honestidad): hasta esta versión la ruta lineal
      hacía UN solo fit sobre todo el histórico y atribuía sobre TODAS las filas —
      1649 de 1654 eran filas de su propio train — mientras el artefacto llevaba el
      header "solo test-folds". El header era falso para ridge/bayesian_ridge y un
      test lo fijaba en los 6 artefactos. Ahora la afirmación es cierta: n_fits > 1 y
      cada fila se atribuye con un fit que no la vio.
  (b) Rule-based SPX500 — ATRIBUCIÓN DE REGLAS (attribution_not_shap=true): % días
      trend_on (MA200), exposición, y descomposición simple de PnL bruto en
      beta (n*mean(pos)*mean(ret)) + timing (n*cov(pos,ret)), global y por año.
      Reusa scripts/analysis/profitability_adapters.ADAPTERS['spx500'] (mismo código
      que produce los bundles publicados — cero re-derivación).

Fase 2 (BL-20 hueco TreeSHAP, 2026-07-28):
  (c) Zoo COP — TREE SHAP EXACTO para xgboost / lightgbm / catboost. Se usa el TreeSHAP
      nativo de cada booster (mismo algoritmo Lundberg et al.; la comprobación
      sum(phi)+bias == pred cruda se ejecuta y se persiste como `additivity_max_abs_err`):
        xgboost  -> Booster.predict(DMatrix, pred_contribs=True)
        lightgbm -> Booster.predict(X, pred_contrib=True)
        catboost -> get_feature_importance(Pool(X), type='ShapValues')
      SOLO TEST-FOLDS (invariante A.7): walk-forward EXPANDING ANUAL — para el año Y se
      hace fit con filas <= 31-dic-(Y-1) menos purga de 5 días y se atribuyen ÚNICAMENTE
      las filas del año Y (jamás una fila que estuvo en su propio train). Cortes:
      global + temporal (por año) + por régimen (gate Hurst CONGELADO de
      config/execution/smart_simple_v1.yaml, evaluado con retornos <= la propia fila).
      Si el backend de un modelo no está disponible se emite un artefacto tipado
      `tree_shap_unavailable` con la razón — NUNCA valores inventados.

Salida: data/interpretability/<surface>/<asset>/ (fuera de public; servido SOLO via API admin:all)
        <model_id>/<version>/summary.json  (safe JSON: sin NaN/Inf, via safe_json_dump).

0 trials en las tres rutas: no se computa ninguna métrica de acierto/performance
(ni DA, ni R², ni Sharpe) y no se selecciona nada — los hiperparámetros son los
defaults congelados del ModelFactory, sin tuning.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.contracts.strategy_schema import safe_json_dump  # noqa: E402

OUT_ROOT = REPO / "data" / "interpretability"  # FUERA de public/ (CXD-040: public bypassea admin:all)
HORIZON = 5          # mismo H y purga que meta01_zoo_ledger.py
ZOO_LINEAR_MODELS = ("ridge", "bayesian_ridge", "ard")   # SHAP lineal cerrado (coef_)
ZOO_TREE_MODELS = ("xgboost", "lightgbm", "catboost")   # TreeSHAP nativo exacto

# ── Cobertura por ACTIVO (BL-20; alcance original restaurado por el operador 2026-08-05) ──
# El generador nacio cableado a `usdcop`: `_write(...)` YA recibia `asset`, pero los siete
# call-sites pasaban el literal. Gold y BTC declaran su PROPIO zoo de 9 modelos en
# `config/assets/*_forecasting.yaml`, con otro vocabulario para los arboles
# (`xgboost_pure` en vez de `xgboost`). `usdcop` sigue siendo el defecto: sin `--asset`
# el comportamiento es byte a byte el de antes.
ASSET_CONFIGS: dict[str, str | None] = {
    "usdcop": None,                                   # config raiz (forecasting_ssot.yaml)
    "xauusd": "config/assets/xauusd_forecasting.yaml",
    "btcusdt": "config/assets/btcusdt_forecasting.yaml",
}


def _models_for_asset(asset: str, kind: str) -> tuple[str, ...]:
    """`model_id`s DECLARADOS por el activo, jamas una lista fija.

    Inventar un nombre aqui produciria un artefacto que dice explicar un modelo que ese
    activo no tiene — exactamente la clase de afirmacion falsa que BL-20 existe para
    impedir. Los `hybrid_*` quedan FUERA de la ruta de arbol a proposito: mezclan lineal
    y arbol, y TreeSHAP no es correcto sobre ellos (decision declarada en la ficha).
    """
    # COP conserva su vocabulario HISTORICO en linear/tree (`xgboost`, no `xgboost_pure`):
    # sus artefactos ya estan publicados bajo esos nombres y renombrarlos los dejaria
    # huerfanos. La ruta `hybrid` es NUEVA para todos, asi que ahi se lee la config
    # tambien para COP — que declara los mismos nueve modelos que Gold y BTC.
    if asset == "usdcop" and kind in ("linear", "tree"):
        return ZOO_LINEAR_MODELS if kind == "linear" else ZOO_TREE_MODELS
    if asset not in ASSET_CONFIGS:
        raise ValueError(f"activo sin config de forecasting declarada: {asset!r}")
    cfg_rel = ASSET_CONFIGS[asset] or "config/forecasting_ssot.yaml"
    import yaml
    declared = list((yaml.safe_load((REPO / cfg_rel).read_text(encoding="utf-8"))
                     .get("models") or {}).keys())
    if kind == "linear":
        return tuple(m for m in declared if m in ZOO_LINEAR_MODELS)
    if kind == "hybrid":
        return tuple(m for m in declared if m.startswith("hybrid_"))
    return tuple(m for m in declared if m.endswith("_pure"))


MIN_TRAIN = 400      # misma guarda que meta01_zoo_ledger.py (años con menos train se SALTAN)

# Header OBLIGATORIO en cada JSON (BL-20 / A.7).
NOTA = "SHAP explica el modelo, no el mercado; solo test-folds; diagnostico 0 trials"


def _rel(path: Path) -> str:
    """Ruta relativa al repo cuando aplica (en tests OUT_ROOT puede ser un temporal)."""
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


#: Campos que NO forman parte de la identidad del artefacto (solo bitácora).
#: ``supersedes`` SÍ forma parte: si el eslabón de sustitución quedara fuera del
#: hash, dos cadenas A/B distintas producirían el MISMO artifact_id y la
#: sustitución sería texto decorativo en vez de una ligadura verificable.
VOLATILE_FIELDS = ("generated_at", "artifact_id")

#: Eslabón de la cadena: entra en la IDENTIDAD (``artifact_id``) pero no en la
#: comparación de CONTENIDO. Regenerar la misma evidencia sobre un artefacto que
#: ya sustituyó a otro sigue siendo un no-op (la cadena es historia, no ciencia).
CHAIN_FIELDS = ("supersedes",)


class ArtifactConflictError(RuntimeError):
    """Se intentó publicar contenido DISTINTO bajo una identidad ya publicada.

    La evidencia publicada no se pisa: o el contenido es idéntico (no-op) o el
    operador supersede explícitamente (``--supersede``), que queda anotado en el
    artefacto nuevo (``supersedes``). No hay tercera vía silenciosa.
    """


# ---------------------------------------------------------------------------
# Provenance: la identidad del artefacto COMPROMETE datos + código + config + modelo
# ---------------------------------------------------------------------------
#
# Antes, la identidad era ``<surface>/<asset>/<model_id>/<version>`` con
# ``version`` = fecha del último dato. Esa clave NO distingue: dos corridas con
# distinto código, distinto dataset (mismo último día) o distintos
# hiperparámetros compartían ruta, y ``_write`` sobrescribía — la MISMA versión
# podía mutar la evidencia en silencio (y ``generated_at`` cambiaba en cada
# corrida, así que ni siquiera era detectable por comparación de bytes).
#
# Ahora cada payload declara ``provenance`` con cuatro huellas y un
# ``artifact_id`` derivado de TODO el contenido no volátil. Misma identidad =>
# mismos bytes; identidad distinta bajo la misma ruta => error, no overwrite.

def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _canonical_bytes(obj) -> bytes:
    """JSON canónico (claves ordenadas, sin espacios superfluos, UTF-8/LF)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _canonical_sha(obj) -> str:
    return _sha(_canonical_bytes(obj))


def _code_fingerprint() -> str:
    """Huella de ESTE generador (normalizada a LF: misma huella en cualquier OS)."""
    raw = Path(__file__).resolve().read_bytes().replace(b"\r\n", b"\n")
    return _sha(raw)


def _frame_fingerprint(df: pd.DataFrame, cols: list[str], date_col: str = "date") -> str:
    """Huella de los DATOS realmente usados: fechas + matriz de features.

    No es un resumen (n filas, rango): son los bytes float64 de las columnas
    usadas, así que una corrección retroactiva de una sola barra cambia la
    huella aunque el último día siga siendo el mismo.
    """
    dates = pd.to_datetime(df[date_col]).astype("int64").to_numpy()
    values = np.ascontiguousarray(df[cols].to_numpy(dtype=float))
    h = hashlib.sha256()
    h.update(_canonical_bytes({"columns": list(cols), "n_rows": int(len(df))}))
    h.update(dates.tobytes())
    h.update(values.tobytes())
    return "sha256:" + h.hexdigest()


def _file_fingerprint(path: Path) -> str:
    """Huella de un fichero de config (LF-normalizada); ``absent`` si no existe."""
    try:
        return _sha(path.read_bytes().replace(b"\r\n", b"\n"))
    except OSError:
        return _sha(b"<absent>")


def _train_size_summary(fold_meta: list[dict], distinct_train_rows: int) -> dict:
    """N de entrenamiento NO ambiguo para un esquema EXPANDING.

    ``sum(n_train por fold)`` cuenta las MISMAS filas una vez por fold (los
    trains son anidados), así que comunicaba un N inflado — 4959 "filas de
    train" sobre un dataset de ~1.6k. Se publica el N del ÚLTIMO fit, el número
    de filas DISTINTAS vistas por algún fit, y el detalle por fold. La suma no
    se publica en ningún campo.
    """
    per_fold = [int(m["n_train"]) for m in fold_meta]
    return {
        "n_train_last_fit": per_fold[-1] if per_fold else 0,
        "n_train_distinct_rows": int(distinct_train_rows),
        "n_train_by_fold": per_fold,
        "n_train_note": ("esquema expanding: los trains son ANIDADOS, la suma por folds "
                         "contaria las mismas filas varias veces y no es un N — se publica "
                         "el N del ultimo fit, las filas distintas y el detalle por fold"),
    }


def _artifact_identity(payload: dict) -> str:
    """sha256 de TODO el contenido no volátil (huellas + eslabón ``supersedes``).

    Es el ``artifact_id``: cambiar UN valor, UNA huella o el eslabón de
    sustitución cambia la identidad.
    """
    core = {k: v for k, v in payload.items() if k not in VOLATILE_FIELDS}
    return _canonical_sha(core)


def _content_identity(payload: dict) -> str:
    """sha256 de la EVIDENCIA (identidad sin el eslabón de la cadena).

    Responde "¿es la misma ciencia?", que es la pregunta de la idempotencia:
    regenerar el mismo resultado sobre un artefacto que sustituyó a otro no
    puede ser un conflicto eterno solo porque el publicado lleva ``supersedes``.
    """
    core = {k: v for k, v in payload.items()
            if k not in VOLATILE_FIELDS and k not in CHAIN_FIELDS}
    return _canonical_sha(core)


def _read_published(out: Path) -> dict:
    """Lee el artefacto publicado y VERIFICA su integridad antes de usarlo.

    El defecto que esto cierra: ``_write`` comparaba el ``artifact_id``
    *declarado* por el fichero almacenado, es decir, confiaba en el mismo campo
    que un manipulador controlaría. Alterar un valor conservando el id se
    reportaba como idempotencia silenciosa. Aquí la identidad se RECOMPUTA sobre
    el contenido realmente almacenado y se compara en tiempo constante.
    """
    try:
        previous = json.loads(out.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactConflictError(
            f"{out}: existe pero no es JSON legible ({exc}) — se rehusa a pisarlo"
        ) from exc
    if not isinstance(previous, dict):
        raise ArtifactConflictError(f"{out}: el artefacto publicado no es un objeto JSON")

    stored = previous.get("artifact_id")
    if not isinstance(stored, str) or not stored:
        # Era del pre-artifact_id: no hay identidad que verificar. Fail-closed;
        # solo un --supersede humano explícito puede seguir adelante.
        raise ArtifactConflictError(
            f"{out}: el artefacto publicado no declara artifact_id (formato previo al "
            "remedio de identidad). No se puede verificar su integridad — usa "
            "--supersede para sustituirlo de forma explicita."
        )
    recomputed = _artifact_identity(previous)
    if not hmac.compare_digest(stored, recomputed):
        raise ArtifactConflictError(
            f"{out}: FALLO DE INTEGRIDAD — el artefacto declara artifact_id={stored!r} "
            f"pero el hash de su contenido almacenado es {recomputed!r}. El fichero fue "
            "modificado despues de publicarse (o se escribio con otro esquema de "
            "identidad: ejecuta --migrate-identity). No se publica nada ni se reporta "
            "idempotencia: la identidad se recomputa, no se cree lo que el fichero dice "
            "de si mismo."
        )
    return previous


def _publish_cas(staged: Path, out: Path) -> bool:
    """Publica ``staged`` en ``out`` SOLO si ``out`` no existe (compare-and-swap).

    ``os.replace`` es atómico pero NO exclusivo: dos primeros publicadores
    divergentes lo llaman ambos y el último pisa al primero — ambos creían haber
    ganado. La creación por enlace duro es atómica **y** falla si el destino ya
    existe, así que exactamente un escritor gana la carrera y el otro se entera.

    Devuelve ``True`` si ganó, ``False`` si otro publicó primero.
    """
    try:
        os.link(staged, out)
        return True
    except FileExistsError:
        return False
    except (OSError, NotImplementedError, AttributeError):
        # Sistema de ficheros sin enlaces duros: se degrada a una creación
        # exclusiva O_EXCL, que sigue dando exactamente-un-ganador. Ventana
        # declarada: entre el create y el write un lector podría ver el fichero
        # vacío (con enlace duro no existe esa ventana).
        try:
            fd = os.open(str(out), os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0))
        except FileExistsError:
            return False
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(staged.read_bytes())
                fh.flush()
                os.fsync(fh.fileno())
        except BaseException:
            out.unlink(missing_ok=True)
            raise
        return True


def _decide_against_published(out: Path, payload: dict, *, supersede: bool) -> dict | None:
    """Compara el payload nuevo contra el artefacto YA publicado en ``out``.

    Devuelve ``None`` si es idempotente (no hay nada que escribir) o el
    ``previous`` verificado cuando toca sustituir. Lanza si hay conflicto.
    """
    previous = _read_published(out)          # verifica integridad ANTES de decidir
    if hmac.compare_digest(_content_identity(previous), _content_identity(payload)):
        return None                          # misma evidencia: NADA se reescribe
    if not supersede:
        raise ArtifactConflictError(
            f"{out}: ya hay un artefacto publicado con artifact_id="
            f"{previous['artifact_id']!r} y el nuevo es {payload['artifact_id']!r}. "
            "Misma (surface/asset/model/version), contenido DISTINTO: datos corregidos, "
            "codigo cambiado, config o modelo distintos, o el calculo no es reproducible. "
            "La evidencia publicada no se sobrescribe — usa --supersede (queda anotado en "
            "'supersedes') o publica bajo otra version."
        )
    return previous


def _write(surface: str, asset: str, model_id: str, version: str, payload: dict,
           *, supersede: bool = False) -> Path:
    """Escritura ATÓMICA, EXCLUSIVA e INMUTABLE del artefacto.

    - **Identidad**: ``artifact_id`` = sha256 del contenido no volátil (huellas
      de datos/código/config/modelo **y** el eslabón ``supersedes``), así que la
      ``version`` deja de ser la única clave y la cadena de sustitución queda
      criptográficamente ligada.
    - **Verificada**: la identidad del artefacto ya publicado se RECOMPUTA sobre
      sus bytes (comparación en tiempo constante) antes de decidir nada. Un
      fichero alterado que conserve su ``artifact_id`` es un fallo de integridad,
      nunca una idempotencia silenciosa.
    - **Inmutable**: si la evidencia publicada es la misma no se reescribe nada
      (``generated_at`` conserva el de la primera publicación). Si difiere, se
      lanza :class:`ArtifactConflictError` — salvo ``supersede=True``, acto humano
      explícito que queda anotado en ``supersedes``.
    - **Atómica + exclusiva**: se materializa en un temporal del MISMO directorio
      (``os.replace``, precedente BL-15 ``write_csv``) y la publicación es un
      compare-and-swap por enlace duro: **exactamente un** escritor concurrente
      gana; el divergente recibe ``ArtifactConflictError`` y los bytes del ganador
      quedan intactos.
    """
    out = OUT_ROOT / surface / asset / model_id / version / "summary.json"
    payload = {k: v for k, v in payload.items() if k != "artifact_id"}
    payload["artifact_id"] = _artifact_identity(payload)

    if out.exists():
        previous = _decide_against_published(out, payload, supersede=supersede)
        if previous is None:
            return out                       # idempotente: NADA se reescribe
        # Qué se sustituye queda EN el artefacto nuevo: sustituir sin dejar rastro
        # de QUÉ se sustituyó volvería a ser una mutación silenciosa.
        payload["supersedes"] = previous["artifact_id"]
        payload["artifact_id"] = _artifact_identity(payload)

    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(out.parent), prefix=".summary-", suffix=".tmp")
    tmp, staged = Path(tmp_name), None
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            safe_json_dump(payload, f)      # sin NaN/Inf (A.7)
            f.flush()
            os.fsync(f.fileno())
        # Materialización atómica: los bytes COMPLETOS bajo un nombre estable.
        staged_fd, staged_name = tempfile.mkstemp(
            dir=str(out.parent), prefix=".publish-", suffix=".staged")
        os.close(staged_fd)
        staged = Path(staged_name)
        os.replace(tmp, staged)
        if supersede:
            os.replace(staged, out)          # sustitución deliberada (acto humano)
            staged = None
        elif not _publish_cas(staged, out):
            # Perdimos la carrera: otro publicó primero. Se decide contra SU
            # contenido (verificado), jamás pisándolo.
            if _decide_against_published(out, payload, supersede=False) is not None:
                raise AssertionError("unreachable: sin supersede el conflicto siempre lanza")
    except BaseException:
        tmp.unlink(missing_ok=True)         # ni parcial ni temporal huérfano
        if staged is not None:
            staged.unlink(missing_ok=True)
        raise
    if staged is not None:
        staged.unlink(missing_ok=True)      # el contenido ya vive en `out`
    return out


def migrate_identity(root: Path | None = None, *, apply: bool = False) -> list[dict]:
    """Migración DELIBERADA del esquema de identidad (``supersedes`` entra al hash).

    Antes el ``artifact_id`` era el hash del contenido EXCLUYENDO ``supersedes``;
    ahora lo incluye. Los artefactos ya publicados que llevan un eslabón traen,
    por tanto, un id del esquema viejo: sin migrarlos, la verificación de
    integridad los marcaría (con razón) como no verificables.

    Es una re-derivación PURA: el único campo que cambia es ``artifact_id``, y
    solo se toca un fichero cuyo id almacenado coincide EXACTAMENTE con su
    identidad de contenido bajo el esquema viejo — es decir, del que se puede
    demostrar que no fue manipulado. Cualquier otra cosa aborta (fail-closed).
    """
    root = Path(root) if root is not None else OUT_ROOT
    report: list[dict] = []
    for path in sorted(root.glob("*/*/*/*/summary.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        stored = doc.get("artifact_id")
        target = _artifact_identity(doc)
        if isinstance(stored, str) and hmac.compare_digest(stored, target):
            report.append({"path": _rel(path), "status": "already_current"})
            continue
        if not (isinstance(stored, str) and stored
                and hmac.compare_digest(stored, _content_identity(doc))):
            raise ArtifactConflictError(
                f"{path}: su artifact_id no coincide ni con el esquema nuevo ni con el "
                "viejo — no es una migracion de esquema, es un fichero alterado. Abortada "
                "la migracion completa sin tocar nada."
            )
        entry = {"path": _rel(path), "status": "migrated" if apply else "would_migrate",
                 "artifact_id_before": stored, "artifact_id_after": target,
                 "content_identity": _content_identity(doc)}
        if apply:
            doc["artifact_id"] = target      # ÚNICO campo que cambia
            fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".migrate-",
                                            suffix=".tmp")
            tmp = Path(tmp_name)
            try:
                with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
                    safe_json_dump(doc, f)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, path)
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
        report.append(entry)
    return report


# ---------------------------------------------------------------------------
# MAQUINARIA COMPARTIDA por las dos rutas del zoo (lineal y árbol)
# ---------------------------------------------------------------------------
#
# El esquema de folds vive AQUÍ, en UNA sola función, en vez de duplicado por
# ruta. Motivo (defecto BL-20 de honestidad, 2026-07-28): la ruta de árbol hacía
# walk-forward anual y la LINEAL atribuía con un único fit global sobre todo el
# histórico — 1649 de 1654 filas atribuidas eran filas de su propio train — pero
# ambos artefactos llevaban el mismo header ``solo test-folds``. Un banner
# constitucional que contradice al campo de al lado es honestidad decorativa.
# Con una sola implementación las dos rutas son comparables por construcción y
# no pueden volver a divergir sin que el diff lo enseñe.

def _annual_expanding_folds(df: pd.DataFrame, feat_cols: list[str]) -> list[dict]:
    """Folds expanding ANUALES: train = filas < 1-ene-Y (purgadas), test = filas de Y.

    Una fila NUNCA se atribuye con un modelo que la vio en su train (invariante A.7).
    La purga descarta las últimas ``HORIZON`` filas del tramo previo: sus targets a 5
    días miran DENTRO del año de test, así que entrenarlas sería look-ahead.

    Requiere ``df`` ordenado por fecha, con índice 0..n-1, y las columnas ``date``,
    ``y5`` y ``feat_cols``. Cada fold publica ``train_idx``/``test_idx`` (índices del
    ``df``) para que un tercero pueda comprobar la disjunción, no solo creérsela.
    """
    folds: list[dict] = []
    for yr in sorted({int(y) for y in df["date"].dt.year.unique()}):
        cut = pd.Timestamp(year=yr, month=1, day=1)
        prev = df[df["date"] < cut]
        if len(prev) <= HORIZON:
            continue
        train = prev.iloc[:-HORIZON]                       # purga: targets 5d no realizados
        m_ok = train[feat_cols].notna().all(axis=1) & train["y5"].notna()
        Xtr = train.loc[m_ok, feat_cols].to_numpy(float)
        ytr = train.loc[m_ok, "y5"].to_numpy(float)
        test = df[(df["date"] >= cut) & (df["date"] < cut.replace(year=yr + 1))]
        test = test[test[feat_cols].notna().all(axis=1)]
        if len(Xtr) < MIN_TRAIN or test.empty:
            continue
        folds.append({"year": yr, "Xtr": Xtr, "ytr": ytr, "test": test,
                      "train_idx": train.loc[m_ok].index.to_numpy(),
                      "test_idx": test.index.to_numpy(),
                      "train_start": pd.Timestamp(train.loc[m_ok, "date"].iloc[0]),
                      "train_end": pd.Timestamp(train.loc[m_ok, "date"].iloc[-1])})
    return folds


def _fold_meta(fold: dict, n_test: int, base_value: float) -> dict:
    """Bitácora publicable de un fold (idéntica en las dos rutas).

    ``fold_fingerprint`` es la huella del train EXACTO: dos corridas con la misma
    huella entrenaron sobre las mismas filas, así que un tercero puede verificar
    QUÉ vio el modelo en vez de fiarse de la prosa del ``scheme``.
    """
    return {"year": int(fold["year"]), "n_train": int(len(fold["Xtr"])),
            "n_test": int(n_test),
            "train_start": fold["train_start"].date().isoformat(),
            "train_end": fold["train_end"].date().isoformat(),
            "base_value": float(base_value),
            "fold_fingerprint": _canonical_sha({
                "year": int(fold["year"]),
                "train_end": fold["train_end"].date().isoformat(),
                "n_train": int(len(fold["Xtr"])),
                "X": _sha(np.ascontiguousarray(fold["Xtr"]).tobytes()),
                "y": _sha(np.ascontiguousarray(fold["ytr"]).tobytes()),
            })}


def _distinct_train_rows(folds: list[dict]) -> int:
    """Filas DISTINTAS que algún fit vio (los trains expanding son ANIDADOS: la
    unión no es la suma, y contar la unión es lo único que da un N real)."""
    return len(set().union(*(set(f["train_idx"].tolist()) for f in folds)))


def _regime_labels(df: pd.DataFrame) -> pd.Series:
    """Etiqueta de régimen por fila con el gate Hurst CONGELADO de smart_simple_v1.yaml.

    Se evalúa con los retornos <= la propia fila (misma información que tendría el gate
    en vivo en esa barra): sin look-ahead. Devuelve strings del RegimeState.
    """
    import yaml
    from src.forecasting.regime_gate import RegimeGateConfig, classify_regime

    raw = {}
    cfg_path = REPO / "config" / "execution" / "smart_simple_v1.yaml"
    try:
        raw = (yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}).get("regime_gate", {})
    except OSError:
        raw = {}
    gcfg = RegimeGateConfig(
        hurst_lookback=int(raw.get("hurst_lookback", RegimeGateConfig.hurst_lookback)),
        hurst_trending=float(raw.get("hurst_trending", RegimeGateConfig.hurst_trending)),
        hurst_mean_rev=float(raw.get("hurst_mean_rev", RegimeGateConfig.hurst_mean_rev)),
    )
    rets = df["close"].pct_change().fillna(0.0).to_numpy(float)
    look = gcfg.hurst_lookback
    out = []
    for i in range(len(rets)):
        window = rets[max(0, i - look + 1): i + 1]      # <= la propia fila
        out.append(classify_regime(list(window), gcfg).state.value)
    return pd.Series(out, index=df.index, name="regime")


def _agg_rows(phi: np.ndarray, feat_cols: list[str], sel: np.ndarray) -> list[dict]:
    """mean|phi| y mean(phi) por feature sobre el subconjunto `sel`, ordenado por magnitud."""
    sub = phi[sel]
    if not len(sub):
        return []
    ma, mp = np.nanmean(np.abs(sub), axis=0), np.nanmean(sub, axis=0)
    return [{"feature": feat_cols[j], "mean_abs_shap": float(ma[j]), "mean_shap": float(mp[j])}
            for j in np.argsort(-ma)]


def _sign_change_flags(groups: dict[str, list[dict]], feat_cols: list[str],
                       scale: float) -> list[str]:
    """Kill-flag A.7: mean(phi) que cambia de signo entre grupos con magnitud material.

    MISMA regla en las dos rutas (eps = 10% de la magnitud media global). Solo FLAG
    diagnóstico: la decisión de rechazar el modelo es humana.
    """
    eps = 0.1 * scale if np.isfinite(scale) else 0.0
    per_feat: dict[str, list[float]] = {c: [] for c in feat_cols}
    for rows in groups.values():
        for r in rows:
            per_feat[r["feature"]].append(float(r["mean_shap"]))
    return sorted(c for c, v in per_feat.items() if v and min(v) < -eps and max(v) > eps)


def _linear_contributions(mdl, Z: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """SHAP lineal EXACTO en las coordenadas que el modelo realmente usa.

    Para un modelo lineal sobre features independientes, el valor SHAP exacto es
    ``phi_j = coef_j * (z_j - mu_j)``. El matiz que hay que respetar: ``ard``
    reescala INTERNAMENTE dentro de su ``fit``/``predict``
    (``src/forecasting/models/ard.py:86``), así que sus ``coef_`` viven en OTRA
    base que la ``Z`` que se le pasa. Componer ese scaler interno es lo que hace
    que ``sum(phi) + intercept`` siga siendo la predicción del wrapper también
    para ARD; ignorarlo daría una atribución que no explica al modelo.
    """
    inner = getattr(mdl, "_scaler", None)          # ARD: StandardScaler interno
    Zi = np.asarray(inner.transform(Z), float) if inner is not None else np.asarray(Z, float)
    coefs = np.asarray(mdl._model.coef_, dtype=float).ravel()
    intercept = float(np.asarray(mdl._model.intercept_).ravel()[0])
    return Zi * coefs, intercept, coefs


# ---------------------------------------------------------------------------
# (a) Zoo COP — SHAP lineal cerrado (ridge / bayesian_ridge / ard), SOLO test-folds
# ---------------------------------------------------------------------------

def generate_zoo_linear(model_ids: tuple[str, ...] | None = None,
                        *, supersede: bool = False, asset: str = "usdcop") -> list[Path]:
    from sklearn.preprocessing import StandardScaler
    from src.forecasting.models.factory import ModelFactory
    from src.forecasting.ssot_config import ForecastingSSOTConfig
    from src.forecasting.dataset_loader import ForecastingDatasetLoader

    if model_ids is None:
        model_ids = _models_for_asset(asset, "linear")
    cfg = ForecastingSSOTConfig.load(ASSET_CONFIGS[asset])
    loader = ForecastingDatasetLoader(cfg, project_root=REPO)
    df, _ = loader.load_dataset()
    feat_cols = [c for c in cfg.get_feature_columns() if c in df.columns]
    df = df.sort_values("date").reset_index(drop=True)
    df["y5"] = df["close"].shift(-HORIZON) / df["close"] - 1.0
    df["regime"] = _regime_labels(df)

    version = pd.Timestamp(df["date"].iloc[-1]).date().isoformat()

    # MISMOS folds que la ruta de árbol (una sola implementación compartida): fit
    # con filas < 1-ene-Y purgadas, atribución SOLO sobre las filas del año Y.
    folds = _annual_expanding_folds(df, feat_cols)
    if not folds:
        raise RuntimeError("zoo lineal: ningún fold anual cumple la guarda de train mínimo")
    distinct_train_rows = _distinct_train_rows(folds)

    # Huella de los DATOS de entrada (misma para todos los modelos de esta corrida).
    data_fp = _frame_fingerprint(df, feat_cols + ["close"])
    code_fp = _code_fingerprint()
    paths: list[Path] = []
    for mid in model_ids:
        phi_parts, base_parts, coef_parts, dates, fold_meta = [], [], [], [], []
        intercepts: list[float] = []
        add_err = 0.0
        for f in folds:
            sc = StandardScaler().fit(f["Xtr"])            # train-only POR FOLD, sin fuga
            Xte = sc.transform(f["test"][feat_cols].to_numpy(float))
            mdl = ModelFactory.create(mid)                 # hiperparámetros CONGELADOS (defaults)
            mdl.fit(sc.transform(f["Xtr"]), f["ytr"])
            phi_f, intercept, coefs = _linear_contributions(mdl, Xte)
            if phi_f.shape[1] != len(feat_cols):
                raise RuntimeError(
                    f"{mid}: {phi_f.shape[1]} contribuciones vs {len(feat_cols)} features")
            # Aditividad de la forma cerrada: sum(phi) + intercept == prediccion del
            # wrapper. Se persiste (no se asume): en lineal exacto ⇒ ~1e-17.
            raw_pred = np.asarray(mdl.predict(Xte), dtype=float).ravel()
            add_err = max(add_err,
                          float(np.nanmax(np.abs(phi_f.sum(axis=1) + intercept - raw_pred))))
            phi_parts.append(phi_f)
            base_parts.append(np.full(len(Xte), intercept, dtype=float))
            coef_parts.append(coefs)
            intercepts.append(intercept)
            dates.append(f["test"][["date", "regime"]])
            fold_meta.append(_fold_meta(f, len(Xte), intercept))

        phi = np.vstack(phi_parts)                         # SOLO filas de test-fold
        meta = pd.concat(dates, ignore_index=True)
        base_value = float(np.nanmean(np.concatenate(base_parts)))
        # El coeficiente ya no es único (un fit por fold): se publica la MEDIA por
        # fold, y el detalle exacto queda comprometido en model_fingerprint.
        coef_mean = np.mean(np.vstack(coef_parts), axis=0)

        col = {c: j for j, c in enumerate(feat_cols)}
        global_rows = _agg_rows(phi, feat_cols, np.ones(len(phi), dtype=bool))
        top_features = [
            {"rank": i + 1, "feature": r["feature"],
             "coef": float(coef_mean[col[r["feature"]]]),
             "mean_abs_shap": r["mean_abs_shap"], "mean_shap": r["mean_shap"]}
            for i, r in enumerate(global_rows)
        ]
        scale = float(np.nanmean([r["mean_abs_shap"] for r in global_rows]))

        yr_arr = meta["date"].dt.year.to_numpy()
        by_year = {str(int(y)): _agg_rows(phi, feat_cols, yr_arr == y)
                   for y in sorted(set(yr_arr))}
        reg_arr = meta["regime"].to_numpy()
        by_regime = {str(r): _agg_rows(phi, feat_cols, reg_arr == r)
                     for r in sorted(set(reg_arr))}

        params = {k: v for k, v in (ModelFactory.create(mid).get_params() or {}).items()
                  if isinstance(v, (int, float, str, bool, type(None)))}
        config_fp = _canonical_sha({
            "horizon": HORIZON, "purge_days": HORIZON, "min_train": MIN_TRAIN,
            "features": feat_cols, "scaler": "StandardScaler train-only por fold",
            "model_id": mid, "params": params,
            "forecasting_ssot": _file_fingerprint(Path(cfg._config_path)),
            "regime_gate_config": _file_fingerprint(
                REPO / "config" / "execution" / "smart_simple_v1.yaml"),
        })
        # Modelo LINEAL: la huella son los coeficientes ajustados DE CADA FOLD —
        # compromete los modelos EXACTOS que produjeron estas atribuciones, no una
        # receta para obtenerlos.
        model_fp = _canonical_sha({
            "basis": "fitted_linear_coefficients_per_fold",
            "model_id": mid, "params": params,
            "folds": [m["fold_fingerprint"] for m in fold_meta],
            "coef": [[float(c) for c in row] for row in coef_parts],
            "intercept": [float(b) for b in intercepts],
        })

        payload = {
            "nota": NOTA,
            "surface": "zoo",
            "asset": asset,
            "model_id": mid,
            "model_type": "linear",
            "method": "linear_shap_closed_form",
            "attribution_not_shap": False,
            "version": version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "provenance": {
                "data_fingerprint": data_fp,
                "code_fingerprint": code_fp,
                "config_fingerprint": config_fp,
                "model_fingerprint": model_fp,
                "model_fingerprint_basis": "fitted_linear_coefficients_per_fold",
                "nota": ("la 'version' (ultimo dia del dataset) NO identifica la evidencia: "
                         "el artifact_id se deriva de estas cuatro huellas mas todo el "
                         "contenido, y republicar contenido distinto bajo la misma version "
                         "es un error, no un overwrite"),
            },
            "additivity_max_abs_err": add_err,
            "fit": {
                "scheme": ("walk-forward EXPANDING ANUAL: fit con filas < 1-ene-Y menos purga "
                           f"de {HORIZON}d; atribucion SOLO sobre filas del año Y (test-fold)"),
                "origin": version,
                "n_fits": len(fold_meta),
                **_train_size_summary(fold_meta, distinct_train_rows),
                "horizon": HORIZON,
                "purge_days": HORIZON,
                "scaler": "StandardScaler train-only por fold",
                "params": params,
            },
            "folds": fold_meta,
            "scope": ("phi_j = coef_j*(z_j-mu_j) EXACTO (forma cerrada) sobre filas OOS: "
                      "ninguna fila fue vista por el fit que la atribuye. Un fit por fold "
                      "anual; el 'coef' publicado es la MEDIA por fold (el detalle exacto "
                      "esta comprometido en model_fingerprint). Hiperparametros = defaults "
                      "CONGELADOS del ModelFactory, sin tuning ni seleccion; ninguna metrica "
                      "de acierto computada (0 trials) — diagnostico, no claim de edge."),
            "base_value": base_value,
            "n_rows": int(len(phi)),
            "n_features": len(feat_cols),
            "n_folds": len(fold_meta),
            "top_features": top_features,
            "by_year": by_year,
            "by_regime": by_regime,
            "regime_gate": ("gate Hurst congelado de config/execution/smart_simple_v1.yaml "
                            "evaluado con retornos <= la propia fila (sin look-ahead)"),
            "kill_flags_sign_change_by_year": _sign_change_flags(by_year, feat_cols, scale),
            "kill_flags_sign_change_by_regime": _sign_change_flags(by_regime, feat_cols, scale),
        }
        p = _write("zoo", asset, mid, version, payload, supersede=supersede)
        paths.append(p)
        print(f"[zoo] {mid}: {_rel(p)} (n_rows={len(phi)}, folds={len(fold_meta)}, "
              f"add_err={add_err:.2e})", flush=True)
    return paths


# ---------------------------------------------------------------------------
# (c) Zoo COP — TreeSHAP exacto (xgboost / lightgbm / catboost), SOLO test-folds
# ---------------------------------------------------------------------------

def _tree_shap_backend(model_id: str):
    """Devuelve (nombre_backend, fn(booster_wrapper, X) -> (phi, bias)) o lanza ImportError.

    Los tres boosters implementan TreeSHAP EXACTO (Lundberg et al.) de forma nativa; la
    última columna que devuelven es el valor base. No se requiere el paquete `shap`
    (que aquí ni siquiera importa: su `_tree.py` arrastra pyspark, roto en py3.12).
    """
    # Gold/BTC declaran los MISMOS boosters con el sufijo `_pure`
    # (`config/assets/*_forecasting.yaml`), y el `ModelFactory` ya los registra bajo
    # ambos nombres. El backend TreeSHAP es el del booster, no el del alias: sin esta
    # normalizacion los tres arboles de cada activo salian `tree_shap_unavailable`
    # — degradacion honesta, pero cobertura CERO por un detalle de vocabulario.
    # Se normaliza SOLO el sufijo declarado; cualquier otro nombre sigue cayendo al
    # `raise` del final, que es lo que impide inventar un backend.
    if model_id.endswith("_pure"):
        model_id = model_id[: -len("_pure")]

    if model_id == "xgboost":
        import xgboost as xgb  # noqa: F401 — falla ⇒ backend no disponible

        def fn(mdl, X):
            raw = np.asarray(
                mdl._model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True), float)
            return raw[:, :-1], raw[:, -1]

        return "xgboost.Booster.predict(pred_contribs=True)", fn

    if model_id == "lightgbm":
        import lightgbm  # noqa: F401

        def fn(mdl, X):
            raw = np.asarray(mdl._model.predict(X, pred_contrib=True), float)
            return raw[:, :-1], raw[:, -1]

        return "lightgbm.Booster.predict(pred_contrib=True)", fn

    if model_id == "catboost":
        from catboost import Pool

        def fn(mdl, X):
            raw = np.asarray(
                mdl._model.get_feature_importance(Pool(X), type="ShapValues"), float)
            return raw[:, :-1], raw[:, -1]

        return "catboost.get_feature_importance(type='ShapValues')", fn

    raise ImportError(f"{model_id}: sin backend TreeSHAP nativo registrado")


def _unavailable_payload(model_id: str, version: str, reason: str, detail: str,
                         *, provenance: dict | None = None,
                         asset: str = "usdcop") -> dict:
    """Estado TIPADO de degradación — jamás valores de atribución fabricados.

    Lleva las MISMAS huellas que un artefacto con datos (el modelo se declara
    ``unavailable``, no se omite): un hueco tambien es evidencia y tambien tiene
    que ser identificable e inmutable.
    """
    return {
        "nota": NOTA,
        "surface": "zoo",
        "asset": asset,
        "model_id": model_id,
        "model_type": "tree",
        "method": "tree_shap_unavailable",
        "attribution_not_shap": False,
        "version": version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance or {
            "data_fingerprint": _sha(b"<unavailable>"),
            "code_fingerprint": _code_fingerprint(),
            "config_fingerprint": _sha(b"<unavailable>"),
            "model_fingerprint": _sha(b"<unavailable>"),
            "model_fingerprint_basis": "unavailable",
        },
        "scope": ("sin atribuciones: el backend TreeSHAP no estuvo disponible en esta "
                  "corrida. NO se emiten valores — un artefacto inventado es peor que un "
                  "hueco declarado."),
        "status": "tree_shap_unavailable",
        "reason": reason,
        "detail": detail,
    }


def generate_zoo_tree(model_ids: tuple[str, ...] | None = None,
                      *, supersede: bool = False, asset: str = "usdcop") -> list[Path]:
    from sklearn.preprocessing import StandardScaler
    from src.forecasting.models.factory import ModelFactory
    from src.forecasting.ssot_config import ForecastingSSOTConfig
    from src.forecasting.dataset_loader import ForecastingDatasetLoader

    if model_ids is None:
        model_ids = _models_for_asset(asset, "tree")
    cfg = ForecastingSSOTConfig.load(ASSET_CONFIGS[asset])
    loader = ForecastingDatasetLoader(cfg, project_root=REPO)
    df, _ = loader.load_dataset()
    feat_cols = [c for c in cfg.get_feature_columns() if c in df.columns]
    df = df.sort_values("date").reset_index(drop=True)
    df["y5"] = df["close"].shift(-HORIZON) / df["close"] - 1.0
    df["regime"] = _regime_labels(df)

    version = pd.Timestamp(df["date"].iloc[-1]).date().isoformat()

    # MISMOS folds que la ruta lineal (una sola implementación compartida).
    folds = _annual_expanding_folds(df, feat_cols)
    if not folds:
        raise RuntimeError("zoo tree: ningún fold anual cumple la guarda de train mínimo")
    distinct_train_rows = _distinct_train_rows(folds)

    data_fp = _frame_fingerprint(df, feat_cols + ["close"])
    code_fp = _code_fingerprint()
    paths: list[Path] = []
    for mid in model_ids:
        try:
            backend_name, shap_fn = _tree_shap_backend(mid)
        except Exception as exc:   # backend/librería ausente ⇒ degradación explícita
            p = _write("zoo", asset, mid, version, _unavailable_payload(
                mid, version, "backend_import_failed", f"{type(exc).__name__}: {exc}",
                provenance={"data_fingerprint": data_fp, "code_fingerprint": code_fp,
                            "config_fingerprint": _sha(b"<no backend>"),
                            "model_fingerprint": _sha(b"<no backend>"),
                            "model_fingerprint_basis": "unavailable"}, asset=asset),
                supersede=supersede)
            paths.append(p)
            print(f"[tree] {mid}: DEGRADADO (backend_import_failed) -> {_rel(p)}",
                  flush=True)
            continue

        try:
            phi_parts, base_parts, dates, fold_meta = [], [], [], []
            add_err = 0.0
            for f in folds:
                # Scaler train-only por fold. Los árboles no lo necesitan, pero es el MISMO
                # preproceso del zoo (meta01) — se mantiene para no cambiar el modelo.
                sc = StandardScaler().fit(f["Xtr"])
                Xte = sc.transform(f["test"][feat_cols].to_numpy(float))
                mdl = ModelFactory.create(mid)              # hiperparámetros CONGELADOS (defaults)
                mdl.fit(sc.transform(f["Xtr"]), f["ytr"])
                phi, bias = shap_fn(mdl, Xte)
                if phi.shape[1] != len(feat_cols):
                    raise RuntimeError(
                        f"{mid}: {phi.shape[1]} contribuciones vs {len(feat_cols)} features")
                # Chequeo de aditividad TreeSHAP: sum(phi) + bias == prediccion CRUDA.
                raw_pred = np.asarray(mdl._model.predict(Xte), float).ravel()
                add_err = max(add_err, float(np.nanmax(np.abs(phi.sum(axis=1) + bias - raw_pred))))
                phi_parts.append(phi)
                base_parts.append(bias)
                dates.append(f["test"][["date", "regime"]])
                fold_meta.append(_fold_meta(f, len(Xte), float(np.nanmean(bias))))

            phi = np.vstack(phi_parts)
            meta = pd.concat(dates, ignore_index=True)
            base_value = float(np.nanmean(np.concatenate(base_parts)))
        except Exception as exc:   # fit/predict/SHAP falló ⇒ degradación explícita
            p = _write("zoo", asset, mid, version, _unavailable_payload(
                mid, version, "shap_computation_failed", f"{type(exc).__name__}: {exc}",
                provenance={"data_fingerprint": data_fp, "code_fingerprint": code_fp,
                            "config_fingerprint": _sha(b"<shap failed>"),
                            "model_fingerprint": _sha(b"<shap failed>"),
                            "model_fingerprint_basis": "unavailable"}, asset=asset),
                supersede=supersede)
            paths.append(p)
            print(f"[tree] {mid}: DEGRADADO (shap_computation_failed) -> {_rel(p)}",
                  flush=True)
            continue

        all_rows = np.ones(len(phi), dtype=bool)
        global_rows = _agg_rows(phi, feat_cols, all_rows)
        top_features = [{"rank": i + 1, **r} for i, r in enumerate(global_rows)]
        scale = float(np.nanmean([r["mean_abs_shap"] for r in global_rows]))

        yr_arr = meta["date"].dt.year.to_numpy()
        by_year = {str(int(y)): _agg_rows(phi, feat_cols, yr_arr == y)
                   for y in sorted(set(yr_arr))}
        reg_arr = meta["regime"].to_numpy()
        by_regime = {str(r): _agg_rows(phi, feat_cols, reg_arr == r)
                     for r in sorted(set(reg_arr))}

        params = {k: v for k, v in (ModelFactory.create(mid).get_params() or {}).items()
                  if isinstance(v, (int, float, str, bool, type(None)))}
        config_fp = _canonical_sha({
            "horizon": HORIZON, "purge_days": HORIZON, "min_train": MIN_TRAIN,
            "features": feat_cols, "scaler": "StandardScaler train-only por fold",
            "model_id": mid, "params": params, "shap_backend": backend_name,
            "forecasting_ssot": _file_fingerprint(Path(cfg._config_path)),
            "regime_gate_config": _file_fingerprint(
                REPO / "config" / "execution" / "smart_simple_v1.yaml"),
        })
        # Modelo de ARBOL: el booster no se serializa aqui, asi que la huella
        # compromete la RECETA EXACTA (id + hiperparametros + el train de cada fold,
        # huella incluida). Se declara la base para no aparentar mas de lo que cubre.
        model_fp = _canonical_sha({
            "basis": "frozen_recipe_plus_fold_train_fingerprints",
            "model_id": mid, "params": params,
            "folds": [m["fold_fingerprint"] for m in fold_meta],
        })

        payload = {
            "nota": NOTA,
            "surface": "zoo",
            "asset": asset,
            "model_id": mid,
            "model_type": "tree",
            "method": "tree_shap",
            "attribution_not_shap": False,
            "version": version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "provenance": {
                "data_fingerprint": data_fp,
                "code_fingerprint": code_fp,
                "config_fingerprint": config_fp,
                "model_fingerprint": model_fp,
                "model_fingerprint_basis": "frozen_recipe_plus_fold_train_fingerprints",
                "nota": ("la 'version' (ultimo dia del dataset) NO identifica la evidencia: "
                         "el artifact_id se deriva de estas cuatro huellas mas todo el "
                         "contenido, y republicar contenido distinto bajo la misma version "
                         "es un error, no un overwrite"),
            },
            "shap_backend": backend_name,
            "shap_package_available": _shap_package_available(),
            "additivity_max_abs_err": add_err,
            "fit": {
                "scheme": ("walk-forward EXPANDING ANUAL: fit con filas < 1-ene-Y menos purga "
                           f"de {HORIZON}d; atribucion SOLO sobre filas del año Y (test-fold)"),
                "origin": version,
                **_train_size_summary(fold_meta, distinct_train_rows),
                "horizon": HORIZON,
                "purge_days": HORIZON,
                "scaler": "StandardScaler train-only por fold",
                "params": params,
            },
            "folds": fold_meta,
            "scope": ("TreeSHAP EXACTO del booster nativo sobre filas OOS (ninguna fila fue "
                      "vista por el modelo que la atribuye). Hiperparametros = defaults "
                      "CONGELADOS del ModelFactory, sin tuning ni seleccion; ninguna metrica "
                      "de acierto computada (0 trials). La atribucion es sobre la salida CRUDA "
                      "del booster: el reescalado de varianza del wrapper predict() es un "
                      "post-proceso afin y NO se atribuye."),
            "base_value": base_value,
            "n_rows": int(len(phi)),
            "n_features": len(feat_cols),
            "n_folds": len(fold_meta),
            "top_features": top_features,
            "by_year": by_year,
            "by_regime": by_regime,
            "regime_gate": ("gate Hurst congelado de config/execution/smart_simple_v1.yaml "
                            "evaluado con retornos <= la propia fila (sin look-ahead)"),
            "kill_flags_sign_change_by_year": _sign_change_flags(by_year, feat_cols, scale),
            "kill_flags_sign_change_by_regime": _sign_change_flags(by_regime, feat_cols, scale),
        }
        p = _write("zoo", asset, mid, version, payload, supersede=supersede)
        paths.append(p)
        print(f"[tree] {mid}: {_rel(p)} (n_rows={len(phi)}, folds={len(fold_meta)}, "
              f"add_err={add_err:.2e})", flush=True)
    return paths


def generate_zoo_hybrid(model_ids: tuple[str, ...] | None = None,
                        *, supersede: bool = False, asset: str = "usdcop") -> list[Path]:
    """SHAP EXACTO de los hibridos por DESCOMPOSICION, no TreeSHAP sobre el conjunto.

    El hibrido NO es un arbol: `HybridBaseModel.predict` es una combinacion CONVEXA

        pred(X) = (1-a) * boost.predict(X)  +  a * ridge.predict(scaler.transform(X))

    Aplicarle TreeSHAP puro seria incorrecto —solo veria el booster y se comeria el
    termino lineal—, y esa es la razon por la que la ficha de BL-20 los dejo fuera.
    Pero la atribucion correcta NO es dificil: **SHAP es aditivo y lineal en la salida
    del modelo**, asi que una combinacion lineal de modelos tiene por valores SHAP la
    misma combinacion lineal de sus valores SHAP:

        phi  = (1-a) * phi_tree            + a * phi_linear
        base = (1-a) * base_tree           + a * intercept

    Es EXACTO, no una aproximacion, y no hay que creerselo: `sum(phi) + base` tiene que
    reproducir `hybrid.predict(X)` hasta precision de coma flotante. Ese es
    `additivity_max_abs_err` — si no cuadra, el artefacto NO se publica y sale la
    degradacion tipada. La prueba viaja con el dato.

    Las dos mitades son a su vez exactas: TreeSHAP nativo del booster (mismo backend que
    la ruta de arbol) y la forma cerrada del lineal. El `Ridge` interno se ajusta sobre
    `scaler.transform(X)`, cuya media de train es 0, luego su baseline es el intercept y
    `phi_j = coef_j * z_j` — sin centrado adicional que inventar.
    """
    from sklearn.preprocessing import StandardScaler
    from src.forecasting.models.factory import ModelFactory
    from src.forecasting.ssot_config import ForecastingSSOTConfig
    from src.forecasting.dataset_loader import ForecastingDatasetLoader

    if model_ids is None:
        model_ids = _models_for_asset(asset, "hybrid")
    if not model_ids:
        return []
    cfg = ForecastingSSOTConfig.load(ASSET_CONFIGS[asset])
    loader = ForecastingDatasetLoader(cfg, project_root=REPO)
    df, _ = loader.load_dataset()
    feat_cols = [c for c in cfg.get_feature_columns() if c in df.columns]
    df = df.sort_values("date").reset_index(drop=True)
    df["y5"] = df["close"].shift(-HORIZON) / df["close"] - 1.0
    df["regime"] = _regime_labels(df)
    version = pd.Timestamp(df["date"].iloc[-1]).date().isoformat()

    folds = _annual_expanding_folds(df, feat_cols)
    if not folds:
        raise RuntimeError("zoo hibrido: ningun fold anual cumple la guarda de train minimo")
    distinct_train_rows = _distinct_train_rows(folds)
    data_fp = _frame_fingerprint(df, feat_cols + ["close"])
    code_fp = _code_fingerprint()

    paths: list[Path] = []
    for mid in model_ids:
        booster = mid.replace("hybrid_", "")        # hybrid_xgboost -> xgboost
        try:
            backend_name, shap_fn = _tree_shap_backend(booster)
        except Exception as exc:
            p = _write("zoo", asset, mid, version, _unavailable_payload(
                mid, version, "backend_import_failed", f"{type(exc).__name__}: {exc}",
                provenance={"data_fingerprint": data_fp, "code_fingerprint": code_fp,
                            "config_fingerprint": _sha(b"<no backend>"),
                            "model_fingerprint": _sha(b"<no backend>"),
                            "model_fingerprint_basis": "unavailable"}, asset=asset),
                supersede=supersede)
            paths.append(p)
            print(f"[hybrid] {mid}: DEGRADADO (backend_import_failed) -> {_rel(p)}", flush=True)
            continue

        try:
            phi_parts, base_parts, dates, fold_meta = [], [], [], []
            add_err = 0.0
            alpha_used = None
            for f in folds:
                sc = StandardScaler().fit(f["Xtr"])
                Xte = sc.transform(f["test"][feat_cols].to_numpy(float))
                mdl = ModelFactory.create(mid)      # hiperparametros CONGELADOS (defaults)
                mdl.fit(sc.transform(f["Xtr"]), f["ytr"])

                a = float(mdl.params.get("alpha", 0.3))
                alpha_used = a if alpha_used is None else alpha_used

                # (i) parte ARBOL: TreeSHAP nativo explica la salida CRUDA del booster,
                # pero el wrapper aplica DESPUES un reescalado de varianza
                # (`xgboost.py::predict`: y = mean + s*(raw - mean)) y es ESA la que entra
                # en la combinacion del hibrido. Ignorarlo fue mi primer intento y la
                # aditividad lo delato: 1e-2, no 1e-16. El reescalado es AFIN, asi que se
                # compone exacto — phi' = s*phi, base' = s*base + t — y `s`,`t` se DERIVAN
                # de la pareja (crudo, reescalado) en vez de reimplementar la formula.
                phi_t, base_t = shap_fn(mdl._boosting_model, Xte)
                raw = np.asarray(mdl._boosting_model._model.predict(Xte), float).ravel()
                scaled = np.asarray(mdl._boosting_model.predict(Xte), float).ravel()
                if np.ptp(raw) > 0:
                    s_aff, t_aff = np.polyfit(raw, scaled, 1)
                    resid = float(np.nanmax(np.abs(s_aff * raw + t_aff - scaled)))
                    # Tolerancia anclada a la PRECISION REAL del booster, no a un numero
                    # bonito: XGBoost predice en float32 (~1e-7 relativo), asi que un
                    # residuo de 1e-9 sobre valores de orden 1e-2 es redondeo, no
                    # no-afinidad. Con 1e-9 absoluto el guard rechazaba una composicion
                    # CORRECTA (medido: 1.013e-09 vs 1.000e-09). Esto NO afloja el
                    # criterio: quien decide de verdad es el candado de aditividad de
                    # abajo, que compara contra la prediccion del hibrido COMPLETO.
                    tol = max(1e-8, 1e-6 * float(np.nanmax(np.abs(scaled))))
                    if resid > tol:
                        raise RuntimeError(
                            f"{mid}: el post-proceso del booster NO es afin (residuo "
                            f"{resid:.3e} > {tol:.3e}); la descomposicion exacta no aplica")
                else:                              # booster degenerado (constante)
                    s_aff, t_aff = 1.0, float(np.nanmean(scaled - raw))
                phi_t = s_aff * np.asarray(phi_t, float)
                base_t = s_aff * np.asarray(base_t, float) + t_aff
                # (ii) parte LINEAL: forma cerrada sobre las coordenadas del Ridge interno
                Z = np.asarray(mdl._scaler.transform(Xte), float)
                coefs = np.asarray(mdl._linear_model.coef_, float).ravel()
                phi_l = Z * coefs
                base_l = float(np.asarray(mdl._linear_model.intercept_).ravel()[0])

                phi = (1.0 - a) * np.asarray(phi_t, float) + a * phi_l
                bias = (1.0 - a) * np.asarray(base_t, float) + a * base_l
                if phi.shape[1] != len(feat_cols):
                    raise RuntimeError(
                        f"{mid}: {phi.shape[1]} contribuciones vs {len(feat_cols)} features")

                # LA PRUEBA: contra la prediccion del HIBRIDO COMPLETO, no de una mitad.
                pred = np.asarray(mdl.predict(Xte), float).ravel()
                add_err = max(add_err,
                              float(np.nanmax(np.abs(phi.sum(axis=1) + bias - pred))))
                phi_parts.append(phi)
                base_parts.append(np.asarray(bias, float).ravel())
                dates.append(f["test"][["date", "regime"]])
                fold_meta.append(_fold_meta(f, len(Xte), float(np.nanmean(bias))))

            # CANDADO DURO DE ADITIVIDAD — mas estricto que la ruta de arbol a proposito.
            # Alli TreeSHAP es exacto por construccion del backend; aqui la descomposicion
            # es MIA, asi que su exactitud hay que DEMOSTRARLA y no publicarla si falla.
            # Mi primer intento daba 1e-2 (olvidaba el reescalado del wrapper) y habria
            # publicado una atribucion que no explica al modelo.
            # Umbral anclado a la PRECISION ALCANZABLE del backend, no a un numero redondo.
            # XGBoost computa en float32 (~1e-7 relativo): su TreeSHAP nativo ya deja
            # residuos de 1e-9..1e-8 sobre predicciones de orden 1e-2, y la ruta de arbol
            # PURO publica 2.72e-08 sin objecion. Un 1e-9 absoluto rechazaba
            # descomposiciones CORRECTAS por redondeo (medido: 3.9e-09 / 1.8e-08 / 6.4e-09).
            # Esto NO es aflojar hasta que pase: mi version equivocada —la que olvidaba el
            # reescalado del wrapper— daba 1e-2, que este umbral sigue rechazando por SEIS
            # ordenes de magnitud.
            pred_scale = float(np.nanmax(np.abs(np.concatenate(base_parts)))) or 1.0
            tol_add = max(1e-9, 1e-6 * pred_scale)
            if not np.isfinite(add_err) or add_err > tol_add:
                raise RuntimeError(
                    f"{mid}: la descomposicion NO reproduce la prediccion del hibrido "
                    f"(additivity_max_abs_err={add_err:.3e} > {tol_add:.3e}). No se publica: "
                    f"una atribucion que no suma a la prediccion no explica al modelo.")

            phi = np.vstack(phi_parts)
            meta = pd.concat(dates, ignore_index=True)
            base_value = float(np.nanmean(np.concatenate(base_parts)))
        except Exception as exc:
            p = _write("zoo", asset, mid, version, _unavailable_payload(
                mid, version, "shap_computation_failed", f"{type(exc).__name__}: {exc}",
                provenance={"data_fingerprint": data_fp, "code_fingerprint": code_fp,
                            "config_fingerprint": _sha(b"<shap failed>"),
                            "model_fingerprint": _sha(b"<shap failed>"),
                            "model_fingerprint_basis": "unavailable"}, asset=asset),
                supersede=supersede)
            paths.append(p)
            print(f"[hybrid] {mid}: DEGRADADO (shap_computation_failed) -> {_rel(p)}", flush=True)
            continue

        global_rows = _agg_rows(phi, feat_cols, np.ones(len(phi), dtype=bool))
        top_features = [{"rank": i + 1, **r} for i, r in enumerate(global_rows)]
        scale = float(np.nanmean([r["mean_abs_shap"] for r in global_rows]))
        yr_arr = meta["date"].dt.year.to_numpy()
        by_year = {str(int(y)): _agg_rows(phi, feat_cols, yr_arr == y)
                   for y in sorted(set(yr_arr))}
        reg_arr = meta["regime"].to_numpy()
        by_regime = {str(r): _agg_rows(phi, feat_cols, reg_arr == r)
                     for r in sorted(set(reg_arr))}

        params = {k: v for k, v in (ModelFactory.create(mid).get_params() or {}).items()
                  if isinstance(v, (int, float, str, bool, type(None)))}
        config_fp = _canonical_sha({
            "horizon": HORIZON, "purge_days": HORIZON, "min_train": MIN_TRAIN,
            "features": feat_cols, "scaler": "StandardScaler train-only por fold",
            "model_id": mid, "params": params, "shap_backend": backend_name,
            "decomposition": "convex_alpha_weighted_tree_plus_linear",
            "forecasting_ssot": _file_fingerprint(Path(cfg._config_path)),
            "regime_gate_config": _file_fingerprint(
                REPO / "config" / "execution" / "smart_simple_v1.yaml"),
        })
        model_fp = _canonical_sha({
            "basis": "frozen_recipe_plus_fold_train_fingerprints",
            "model_id": mid, "params": params,
            "folds": [m["fold_fingerprint"] for m in fold_meta],
        })

        payload = {
            "nota": NOTA,
            "surface": "zoo",
            "asset": asset,
            "model_id": mid,
            "model_type": "hybrid",
            "method": "hybrid_shap_convex_decomposition",
            "attribution_not_shap": False,
            "version": version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "provenance": {
                "data_fingerprint": data_fp,
                "code_fingerprint": code_fp,
                "config_fingerprint": config_fp,
                "model_fingerprint": model_fp,
                "model_fingerprint_basis": "frozen_recipe_plus_fold_train_fingerprints",
            },
            "shap_backend": f"{backend_name} (parte arbol) + forma cerrada Ridge (parte lineal)",
            "shap_package_available": _shap_package_available(),
            "additivity_max_abs_err": add_err,
            "hybrid_alpha": alpha_used,
            "fit": {
                "scheme": ("walk-forward EXPANDING ANUAL: fit con filas < 1-ene-Y menos purga "
                           f"de {HORIZON}d; atribucion SOLO sobre filas del año Y (test-fold)"),
                "origin": version,
                **_train_size_summary(fold_meta, distinct_train_rows),
                "horizon": HORIZON,
                "purge_days": HORIZON,
                "scaler": "StandardScaler train-only por fold",
                "params": params,
            },
            "folds": fold_meta,
            "scope": ("SHAP EXACTO por DESCOMPOSICION del hibrido, no TreeSHAP sobre el "
                      "conjunto: pred = (1-a)*boost(X) + a*ridge(scaler(X)), y SHAP es "
                      "aditivo y lineal en la salida, luego phi = (1-a)*phi_tree + a*phi_lin "
                      "y base = (1-a)*base_tree + a*intercept. Aplicar TreeSHAP puro seria "
                      "incorrecto (ignoraria el termino lineal). La exactitud NO se afirma: "
                      "se COMPRUEBA contra la prediccion del hibrido COMPLETO y se publica "
                      "como additivity_max_abs_err. Filas OOS unicamente; hiperparametros "
                      "defaults CONGELADOS, ninguna metrica de acierto computada (0 trials)."),
            "base_value": base_value,
            "n_rows": int(len(phi)),
            "n_features": len(feat_cols),
            "n_folds": len(fold_meta),
            "top_features": top_features,
            "by_year": by_year,
            "by_regime": by_regime,
            "regime_gate": ("gate Hurst congelado de config/execution/smart_simple_v1.yaml "
                            "evaluado con retornos <= la propia fila (sin look-ahead)"),
            "kill_flags_sign_change_by_year": _sign_change_flags(by_year, feat_cols, scale),
            "kill_flags_sign_change_by_regime": _sign_change_flags(by_regime, feat_cols, scale),
        }
        p = _write("zoo", asset, mid, version, payload, supersede=supersede)
        paths.append(p)
        print(f"[hybrid] {mid}: {_rel(p)} (n_rows={len(phi)}, folds={len(fold_meta)}, "
              f"alpha={alpha_used}, add_err={add_err:.2e})", flush=True)
    return paths


def _shap_package_available() -> bool:
    """¿Importa el paquete `shap`? Informativo: el TreeSHAP usado es el NATIVO del booster."""
    try:
        import shap  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# (b) Rule-based SPX500 — atribución de reglas (NO SHAP)
# ---------------------------------------------------------------------------

def _decompose(pos: np.ndarray, ret: np.ndarray, cost: np.ndarray, swap: np.ndarray) -> dict:
    """PnL bruto = beta + timing.  timing = n*cov(pos, ret); beta = n*mean(pos)*mean(ret)."""
    n = len(pos)
    gross = float(np.nansum(pos * ret))
    beta = float(n * np.nanmean(pos) * np.nanmean(ret)) if n else 0.0
    timing = gross - beta                                 # identidad exacta: cov muestral sesgada
    costs = float(np.nansum(cost) + np.nansum(swap))
    return {"n_days": int(n), "pnl_gross": gross, "pnl_beta": beta,
            "pnl_timing_cov_pos_ret": timing, "costs": costs, "pnl_net": gross - costs}


def generate_rule_attribution(rule_ids: tuple[str, ...] = ("spx500",),
                              *, supersede: bool = False) -> list[Path]:
    from scripts.analysis.profitability_adapters import ADAPTERS

    paths: list[Path] = []
    for rid in rule_ids:
        sleeve = ADAPTERS[rid]()
        idx = pd.DatetimeIndex(pd.to_datetime(sleeve.index))
        pos, ret = sleeve.position, sleeve.asset_ret
        cost, swap = sleeve.cost, sleeve.swap
        trend_on = sleeve.dumb_position                   # spx500: ma200_always_on = la regla

        by_year = {}
        for yr in sorted(set(idx.year)):
            sel = (idx.year == yr)
            d = _decompose(pos[sel], ret[sel], cost[sel], swap[sel])
            d["pct_days_trend_on"] = float(np.nanmean(trend_on[sel] > 0))
            d["pct_days_position_active"] = float(np.nanmean(np.abs(pos[sel]) > 1e-9))
            d["avg_exposure"] = float(np.nanmean(pos[sel]))
            by_year[str(int(yr))] = d

        total = _decompose(pos, ret, cost, swap)
        version = idx[-1].date().isoformat()
        # Huella de la SERIE del adapter publicado (posiciones, retornos, costes):
        # una re-derivacion distinta del bundle cambia la identidad del artefacto.
        series_df = pd.DataFrame({
            "date": idx, "position": np.asarray(pos, float), "asset_ret": np.asarray(ret, float),
            "cost": np.asarray(cost, float), "swap": np.asarray(swap, float),
            "dumb_position": np.asarray(trend_on, float),
        })
        data_fp = _frame_fingerprint(
            series_df, ["position", "asset_ret", "cost", "swap", "dumb_position"])
        # Una politica de reglas NO tiene pesos: su "modelo" ES la receta congelada
        # (adapter + estrategia + reloj). Se declara asi, sin aparentar un fit.
        model_fp = _canonical_sha({
            "basis": "rule_based_frozen_recipe",
            "adapter": rid, "strategy_id": sleeve.strategy_id, "asset": sleeve.asset,
            "clock_label": sleeve.clock_label, "n_trades": sleeve.n_trades,
        })
        payload = {
            "nota": NOTA,
            "surface": "rule_based",
            "asset": sleeve.asset,
            "model_id": sleeve.strategy_id,
            "model_type": "rule_based",
            "method": "rule_attribution",
            "attribution_not_shap": True,               # ATRIBUCION, no SHAP (BL-20 punto 2)
            "version": version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "provenance": {
                "data_fingerprint": data_fp,
                "code_fingerprint": _code_fingerprint(),
                "config_fingerprint": _canonical_sha({
                    "adapter": rid, "adapters_module": _file_fingerprint(
                        REPO / "scripts" / "analysis" / "profitability_adapters.py"),
                    "clock_label": sleeve.clock_label,
                }),
                "model_fingerprint": model_fp,
                "model_fingerprint_basis": "rule_based_frozen_recipe",
                "nota": ("la 'version' (ultimo dia de la serie) NO identifica la evidencia: "
                         "el artifact_id se deriva de estas cuatro huellas mas todo el "
                         "contenido, y republicar contenido distinto bajo la misma version "
                         "es un error, no un overwrite"),
            },
            "scope": ("atribucion de reglas sobre la misma serie del adapter publicado "
                      f"({sleeve.clock_label}); descomposicion pnl_gross = beta + timing, "
                      "timing = n*cov(pos,ret) — diagnostico, no claim de edge"),
            "rules": {
                "gate": "trend_on = close > MA200 (dumb baseline ma200_always_on del sleeve)",
                "pct_days_trend_on": float(np.nanmean(trend_on > 0)),
                "pct_days_position_active": float(np.nanmean(np.abs(pos) > 1e-9)),
                "avg_exposure": float(np.nanmean(pos)),
                "n_trades": sleeve.n_trades,
            },
            "pnl_decomposition": total,
            "by_year": by_year,
        }
        p = _write("rule_based", sleeve.asset, sleeve.strategy_id, version, payload,
                   supersede=supersede)
        paths.append(p)
        print(f"[rule] {rid}: {_rel(p)}", flush=True)
    return paths


# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--asset", default="usdcop", choices=sorted(ASSET_CONFIGS),
                    help="activo del zoo. Sin este flag: usdcop, comportamiento previo "
                         "intacto. Con xauusd/btcusdt se carga SU config y SUS model_id "
                         "declarados (los hybrid_* quedan fuera: TreeSHAP no es correcto "
                         "sobre un modelo mitad lineal mitad arbol)")
    ap.add_argument("--zoo-models", default=None,
                    help="modelos LINEALES del zoo (SHAP cerrado). Por defecto, los que "
                         "DECLARA el activo — nunca una lista fija")
    ap.add_argument("--tree-models", default=None,
                    help="modelos de ARBOL del zoo (TreeSHAP nativo). Por defecto, los que "
                         "DECLARA el activo — nunca una lista fija")
    ap.add_argument("--rules", default="spx500",
                    help="adapters rule-based (profitability_adapters.ADAPTERS)")
    ap.add_argument("--skip-zoo", action="store_true")
    ap.add_argument("--skip-trees", action="store_true")
    ap.add_argument("--skip-rules", action="store_true")
    ap.add_argument("--supersede", action="store_true",
                    help="ACTO EXPLICITO: republica sobre una identidad ya publicada. "
                         "Sin este flag, contenido distinto bajo la misma "
                         "(surface/asset/model/version) es ERROR, no overwrite. El "
                         "artefacto nuevo anota el artifact_id que sustituye "
                         "('supersedes'), asi que la sustitucion queda en el registro.")
    ap.add_argument("--migrate-identity", action="store_true",
                    help="MIGRACION DE ESQUEMA (no publica ciencia): re-deriva el "
                         "artifact_id de los artefactos ya publicados ahora que "
                         "'supersedes' entra en el hash. Solo toca ficheros cuyo id "
                         "coincide exactamente con el esquema viejo (prueba de que no "
                         "fueron alterados); cualquier otra cosa aborta. Dry-run salvo "
                         "que se pase --apply.")
    ap.add_argument("--apply", action="store_true",
                    help="ejecuta de verdad la migracion de --migrate-identity")
    args = ap.parse_args(argv)

    if args.migrate_identity:
        report = migrate_identity(apply=args.apply)
        for row in report:
            print(json.dumps(row, ensure_ascii=False))
        pend = sum(1 for r in report if r["status"] != "already_current")
        print(f"{'MIGRADOS' if args.apply else 'PENDIENTES (dry-run)'}: {pend}/{len(report)}")
        return 0

    paths: list[Path] = []
    # Los model_id permitidos los DECLARA el activo: validar contra una lista fija
    # rechazaria `xgboost_pure` (el nombre real de Gold/BTC) y aceptaria `xgboost`
    # para un activo que no lo tiene.
    allowed_lin = _models_for_asset(args.asset, "linear")
    allowed_tree = _models_for_asset(args.asset, "tree")
    if not args.skip_zoo:
        mids = (tuple(m.strip() for m in args.zoo_models.split(",") if m.strip())
                if args.zoo_models else None)
        bad = [m for m in (mids or ()) if m not in allowed_lin]
        if bad:
            raise SystemExit(f"--zoo-models: {bad} no son lineales declarados por "
                             f"{args.asset!r} (declarados: {list(allowed_lin)})")
        paths += generate_zoo_linear(mids, supersede=args.supersede, asset=args.asset)
    if not args.skip_trees:
        tids = (tuple(m.strip() for m in args.tree_models.split(",") if m.strip())
                if args.tree_models else None)
        bad = [m for m in (tids or ()) if m not in allowed_tree]
        if bad:
            raise SystemExit(f"--tree-models: {bad} no son arboles declarados por "
                             f"{args.asset!r} (declarados: {list(allowed_tree)})")
        paths += generate_zoo_tree(tids, supersede=args.supersede, asset=args.asset)
    if not args.skip_rules:
        rids = tuple(r.strip() for r in args.rules.split(",") if r.strip())
        paths += generate_rule_attribution(rids, supersede=args.supersede)

    print(f"OK: {len(paths)} artefactos")
    for p in paths:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
