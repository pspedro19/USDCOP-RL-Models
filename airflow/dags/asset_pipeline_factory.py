"""
DAG FACTORY: asset_<asset_id>_pipeline_weekly
=============================================
Emits ONE data-science-lifecycle pipeline DAG per asset declared in
`config/assets/pipelines.yaml`, making every tradeable index/pair (Gold, BTC,
future additions) fully **DAG-driven** — not just a manual script-runner.

Each generated DAG's tasks map to the DS-cycle:

    l0_ingest            ->  L0  Data ingestion (daily OHLCV -> seed / DB)
    l4_backtest_publish  ->  L2+L4+L5  features -> regime -> backtest (honest gate) -> publish bundle
    l6_verify_registry   ->  L6  Verify/Monitor (registry.json has the asset + fresh strategy bundles)

USD/COP is intentionally NOT here — it runs the richer bespoke H5 weekly chain
(forecast_h5_l3..l7). Adding an asset = ONE entry in the SSOT yaml (no code here).

Graceful degradation: a stage marked `graceful: true` (the ingest refresh) does
not block the science stage — if the live feed is down, the backtest still runs
on the last good seed. The failed ingest task remains visible in the run so the
staleness is surfaced, never hidden (`trigger_rule=ALL_DONE` on the next stage).

Contract: CTR-ASSET-PIPELINE-001
Version: 1.0.0
Date: 2026-07-05
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import yaml
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.utils.trigger_rule import TriggerRule

from src.orchestration.dataset_uri import DatasetContractError, validate_dataset_edges

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/opt/airflow")
CONFIG_PATH = PROJECT_ROOT / "config" / "assets" / "pipelines.yaml"

DEFAULT_ARGS = {
    "owner": "asset-pipeline",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "execution_timeout": timedelta(minutes=45),
}


def _load_config() -> dict:
    """Load the SSOT, degrading only on availability/serialization failures."""
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        validate_dataset_edges(config.get("dataset_edges") or [])
        return config
    except DatasetContractError:
        raise
    except Exception as e:  # noqa: BLE001
        logger.warning("asset_pipeline_factory: could not read %s: %s", CONFIG_PATH, e)
        return {}


def _run_stage(script: str, args: list[str], stage_name: str, **context) -> None:
    """Run a pipeline stage as a subprocess from the repo root."""
    script_path = PROJECT_ROOT / script
    if not script_path.exists():
        raise FileNotFoundError(f"[{stage_name}] script not found: {script_path}")

    cmd = [sys.executable, str(script_path), *args]
    logger.info("[%s] running: %s", stage_name, " ".join(cmd))
    result = subprocess.run(
        cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=40 * 60,
    )
    for line in (result.stdout or "").splitlines()[-80:]:
        logger.info("[%s] %s", stage_name, line)
    for line in (result.stderr or "").splitlines()[-30:]:
        logger.warning("[%s:err] %s", stage_name, line)

    if result.returncode != 0:
        raise RuntimeError(f"[{stage_name}] exited {result.returncode}")
    logger.info("[%s] DONE", stage_name)


def _make_verify(registry_root: str, registry_asset: str, strategy_ids: list[str]):
    """Build the L6 verify callable: registry has the asset + all strategy bundles."""

    def _verify(**context) -> None:
        registry_path = PROJECT_ROOT / registry_root / "registry.json"
        if not registry_path.exists():
            raise FileNotFoundError(f"registry.json not found: {registry_path}")
        with open(registry_path, encoding="utf-8") as f:
            reg = json.load(f)

        asset_ids = {a.get("asset_id") for a in reg.get("assets", [])}
        if registry_asset not in asset_ids:
            raise RuntimeError(f"registry missing asset '{registry_asset}' (have: {sorted(asset_ids)})")

        published = {s.get("strategy_id") for s in reg.get("strategies", [])
                     if s.get("asset_id") == registry_asset}
        missing = [sid for sid in strategy_ids if sid not in published]
        if missing:
            raise RuntimeError(
                f"registry missing strategies for {registry_asset}: {missing} "
                f"(published: {sorted(published)})"
            )

        # Confirm each strategy's manifest bundle is present on disk.
        for sid in strategy_ids:
            manifest = PROJECT_ROOT / registry_root / "strategies" / sid / "manifest.json"
            if not manifest.exists():
                raise RuntimeError(f"bundle manifest missing for {sid}: {manifest}")

        logger.info("[verify] %s OK — %d strategies published + bundles present",
                    registry_asset, len(strategy_ids))

    return _verify



# --- C-010 R3: cadena gobernada de politica (aditiva, fail-closed) -----------
#
# Solo se emite para referencias declaradas en `policy_runs` cuyo
# `migration.status` sea PARITY_GREEN o CUTOVER. `SPEC_ONLY`/`PARITY_PENDING`
# producen CERO tareas -- no un skip verde.
#
# ESTADO REAL (actualizado 2026-08-06, CLD-553/CXD-593): ya NO es cierto que no
# haya entradas declaradas. `spx500` declara `policy_runs: [spx500_daily_ma200_v1]`
# desde `04fa2dd2` y esa policy esta en PARITY_GREEN, o sea que ES elegible y SI
# emite cadena. El comentario anterior seguia afirmando "cero entradas declaradas"
# y su candado no podia desmentirlo porque leia un CONFIG_PATH inexistente en host.
#
# Promover un `migration.status` es acto EXCLUSIVO del operador (C-010 amendment):
# este codigo solo lee el estado, nunca lo escribe ni lo infiere.
ELIGIBLE_MIGRATION_STATES = frozenset({"PARITY_GREEN", "CUTOVER"})

#: Ramificacion permitida: por `engine.type`, jamas por `strategy_id`
#: (`strategy-engines.md` invariante 1).
SUPPORTED_ENGINE_TYPES = frozenset({"rule_based"})


class PolicyRunConfigError(RuntimeError):
    """Declaracion de `policy_runs` invalida: falla al parsear el DAG, no en runtime."""


def resolve_policy_runs(spec: dict) -> list[dict]:
    """Resolver `policy_runs` a las referencias ELEGIBLES, o fallar cerrado.

    Devuelve `[]` sin importar nada cuando no hay `policy_runs`, para que el
    grafo actual no cambie ni adquiera dependencias nuevas.
    """
    declared = spec.get("policy_runs") or []
    if not declared:
        return []

    seen: set[str] = set()
    wanted: list[str] = []
    for index, entry in enumerate(declared):
        if not isinstance(entry, dict) or not isinstance(entry.get("policy_id"), str):
            raise PolicyRunConfigError(
                f"policy_runs[{index}] debe ser un mapping con `policy_id` de texto"
            )
        policy_id = entry["policy_id"].strip()
        if not policy_id:
            raise PolicyRunConfigError(f"policy_runs[{index}]: `policy_id` vacio")
        if policy_id in seen:
            raise PolicyRunConfigError(f"policy_runs: `policy_id` duplicado: {policy_id}")
        seen.add(policy_id)
        wanted.append(policy_id)

    from src.strategies.policies.loader import load_all_policy_specs

    by_id = {str(item["id"]): item for item in load_all_policy_specs()}
    unknown = [pid for pid in wanted if pid not in by_id]
    if unknown:
        raise PolicyRunConfigError(
            f"policy_runs referencia policies que el loader SSOT no conoce: {unknown}"
        )

    eligible: list[dict] = []
    for policy_id in wanted:
        policy_spec = by_id[policy_id]
        status = (policy_spec.get("migration") or {}).get("status")
        if status not in ELIGIBLE_MIGRATION_STATES:
            continue  # inerte: cero tareas, nunca un skip verde
        engine_type = (policy_spec.get("engine") or {}).get("type")
        if engine_type not in SUPPORTED_ENGINE_TYPES:
            raise PolicyRunConfigError(
                f"{policy_id}: engine.type {engine_type!r} elegible pero no soportado "
                f"por la cadena gobernada (soportados: {sorted(SUPPORTED_ENGINE_TYPES)})"
            )
        # `retrain` decide si la cadena necesita una tarea de ENTRENAMIENTO. Hoy solo se
        # sabe generar la cadena sin train (`never`), asi que cualquier otro valor **falla
        # cerrado** en vez de omitirlo en silencio: omitir el train de una policy que SI lo
        # necesita produciria decisiones con un modelo caducado y el grafo se veria sano.
        # No entrenar es una decision declarada, no un hueco.
        # OJO: `retrain` vive bajo `engine`, no en la raiz del spec (medido en los 4
        # specs vigentes). Leerlo de la raiz daba None para TODOS y habria disparado un
        # fail-closed espurio en cada policy elegible: una guarda que se activa siempre
        # no protege, bloquea.
        retrain = (policy_spec.get("engine") or {}).get("retrain")
        if retrain != "never":
            raise PolicyRunConfigError(
                f"{policy_id}: retrain={retrain!r}; la cadena gobernada solo sabe generar "
                f"policies con `retrain: never` (sin tarea de entrenamiento). Omitirlo "
                f"silenciosamente entregaria decisiones con un modelo sin reentrenar"
            )
        eligible.append(
            {"policy_id": policy_id, "engine_type": engine_type, "retrain": retrain}
        )
    return eligible


#: Clave XCom del HECHO de frescura. No es un default: si no esta, la cadena
#: falla cerrado (ver `_declared_max_snapshot_age` y `_policy_context`).
STALENESS_XCOM_KEY = "snapshot_is_stale"


def _declared_max_snapshot_age(spec: dict):
    """La edad maxima DECLARADA por la policy, o `None` si no la declara.

    Se lee de `inputs.max_snapshot_age` como duracion ISO-8601 (`P1D`, `PT4H`).
    **Ninguno de los cuatro specs vigentes la declara** (medido), y por eso la
    cadena falla cerrado: un umbral de frescura decide CUANDO opera la estrategia,
    asi que inventarlo aqui seria fijar un parametro economico por conveniencia de
    fontaneria. Es una declaracion que le toca al spec, no al orquestador
    (`quant-constitution.md` §1: priors declarados ex-ante, nunca deducidos para
    que algo pase).
    """
    declarado = (spec.get("inputs") or {}).get("max_snapshot_age")
    if declarado is None:
        return None
    if not isinstance(declarado, str) or not declarado.startswith("P"):
        raise PolicyRunConfigError(
            f"{spec.get('id')}: `inputs.max_snapshot_age` debe ser una duracion "
            f"ISO-8601 (p.ej. P1D), no {declarado!r}"
        )
    from datetime import timedelta

    texto = declarado[1:]
    dias = horas = minutos = 0
    if "T" in texto:
        fecha, _, hora = texto.partition("T")
    else:
        fecha, hora = texto, ""
    if not fecha and not hora:
        # "P" y "PT" no declaran NADA. Antes devolvian timedelta(0) en silencio: una
        # declaracion malformada pasaba por un umbral valido de cero. La direccion
        # era fail-safe (todo stale), pero aceptar basura callando es como se cuela
        # una configuracion sin sentido creyendo que hay criterio. `P0D` SI es
        # legitimo -- declara "mismo instante" -- asi que se rechaza el vacio, no el cero.
        raise PolicyRunConfigError(
            f"{spec.get('id')}: `max_snapshot_age` = {declarado!r} no declara ninguna "
            f"magnitud (usa P1D, PT4H, PT30M...)"
        )
    if fecha.endswith("D"):
        dias = int(fecha[:-1])
    elif fecha:
        raise PolicyRunConfigError(
            f"{spec.get('id')}: solo se soportan dias/horas/minutos en "
            f"`max_snapshot_age`, no {declarado!r}"
        )
    if hora.endswith("H"):
        horas = int(hora[:-1])
    elif hora.endswith("M"):
        minutos = int(hora[:-1])
    elif hora:
        raise PolicyRunConfigError(
            f"{spec.get('id')}: componente horario no soportado en {declarado!r}"
        )
    return timedelta(days=dias, hours=horas, minutes=minutos)


def _derive_staleness(
    policy_id: str, observations: dict, decision_cutoff, spec: dict
) -> bool | None:
    """DERIVAR el hecho de frescura de la evidencia, o fallar cerrado.

    Vive en la frontera de LECTURA porque es el unico sitio donde existe: 
    `resolve_feature_snapshot` proyecta solo `{feature: valor}` y **descarta la
    metadata a proposito** ("Metadata stays at the read boundary"), asi que
    aguas abajo la frescura ya no es derivable — solo inventable. Eso es
    exactamente lo que hacia R4 con `snapshot_is_stale=False` por defecto, y
    CXD-600 tenia razon en que no es un limite pasivo: **fabrica** un hecho de
    frescura y deja evaluar un snapshot viejo como si fuera nuevo.

    Regla: stale := (decision_cutoff - max(available_at)) > max_snapshot_age.
    Sin `max_snapshot_age` declarado no hay criterio, y sin criterio no se decide.
    """
    from src.orchestration.feature_snapshot import _aware_datetime

    limite = _declared_max_snapshot_age(spec)
    if limite is None:
        raise PolicyRunConfigError(
            f"{policy_id}: la policy no declara `inputs.max_snapshot_age`, asi que la "
            f"frescura del snapshot NO es derivable. La cadena falla cerrado en vez de "
            f"asumir que el dato esta fresco: `stale_input_policy` "
            f"({(spec.get('policy') or {}).get('stale_input_policy')!r}) seria "
            f"inalcanzable y se evaluaria un snapshot viejo sin saberlo. Declarar el "
            f"umbral es decision de la policy, no del orquestador"
        )
    # Sobre las features REQUERIDAS, no sobre todo lo que llegue. La regla es "stale
    # si cualquier observacion REQUERIDA excede el umbral": una feature declarada
    # `optional` no puede bloquear una decision que la policy dice saber tomar sin
    # ella.
    #
    # CORRECCION (CXD-607): aqui decia "hoy los cuatro specs declaran
    # `optional_features: []`, asi que el comportamiento observable no cambia".
    # ERA FALSO — medido: `gold_trend_simple` y `btc_hodl_b1` declaran
    # `[regime_risk_mult]`. Se miro solo `spx500` y se generalizo a cuatro, y sobre
    # esa frase se argumento que el cambio no afectaba a nadie. Afectaba a esos dos,
    # que es justo donde CXD-605 encontro el defecto.
    #
    # Solo features REQUERIDAS presentes, y JAMAS opcionales (CXD-606 §1).
    # Si el conjunto requerido esta incompleto la frescura NO es medible: se
    # devuelve `None`, nunca `False`. Fabricar un "fresco" para poder seguir es lo
    # que ya costo dos rechazos; y medir una opcional como sustituto dejaba que la
    # EDAD de un dato que la policy dice no necesitar reclasificara la ausencia del
    # nucleo requerido (CXD-605). Con `missing` resuelto ANTES que `stale` en el
    # runner, este `None` no puede alcanzar el chequeo de frescura.
    requeridas = (spec.get("inputs") or {}).get("required_features") or []
    considerar = {n: o for n, o in observations.items() if n in requeridas}
    if not requeridas or len(considerar) < len(requeridas):
        return None
    cutoff = _aware_datetime(decision_cutoff, field="decision_cutoff")
    # `min`, NO `max`: el snapshot esta stale si CUALQUIER observacion requerida
    # excede el umbral, asi que manda la MAS VIEJA. R5 usaba `max` -- la mas nueva --
    # y con eso una feature reciente blanqueaba a otra de hace seis dias: el
    # snapshot se declaraba fresco y la policy operaba con un input caducado
    # (CXD-603, probado con cutoff 24-jul, `vieja` del 18 y `fresca` del 24 -> False
    # cuando la respuesta segura era True). Un agregado mal elegido no es un detalle
    # de estilo: convierte el peor caso en el mejor.
    mas_vieja = min(
        _aware_datetime(obs["available_at"], field=f"feature {name!r} available_at")
        for name, obs in considerar.items()
    )
    return (cutoff - mas_vieja) > limite


def make_produce_observations(policy_id: str):
    """Tarea 0: MATERIALIZAR el snapshot. El eslabon que faltaba (BL-45 C2).

    Hasta ahora la cadena empezaba en `resolve_snapshot`, que hace `xcom_pull` de
    `observations::<policy_id>` y `decision_cutoff::<policy_id>` — y **ninguna tarea
    productiva los ponia**. El grafo mostraba cuatro eslabones que ninguna corrida
    podia atravesar: el mecanismo existia y no tenia productor, igual que
    `resolve_feature_snapshot` antes de C-010.

    Trae las barras canonicas y delega en `build_observations`, que es funcion PURA
    sobre un DataFrame: una implementacion, dos consumidores (esta tarea y los tests),
    el mismo patron que `validate_policy_inputs`. La consulta vive aqui porque es
    efecto de borde; la logica de QUE materializar vive en el feature-set y el
    catalogo, nunca en este fichero.
    """

    def _produce(**context):
        import pandas as pd

        from src.features.observations import build_observations

        spec = _spec_for(policy_id)
        asset_id = spec.get("asset")
        cutoff = _policy_context(policy_id, context).as_of

        from utils.dag_common import get_db_connection

        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT canonical_symbol FROM reference.instrument WHERE asset_id = %s",
                    (asset_id,),
                )
                fila = cur.fetchone()
                if not fila:
                    raise PolicyRunConfigError(
                        f"{asset_id}: sin fila en reference.instrument; sin simbolo "
                        f"canonico no se sabe QUE serie materializar"
                    )
                # Serie COMPLETA hasta el cutoff: las features con ventana no se
                # pueden calcular sobre un recorte. Un LIMIT aqui daria un numero
                # plausible y silenciosamente distinto -- el peor error, porque no falla.
                cur.execute(
                    "SELECT time, close FROM public.asset_daily_ohlcv "
                    "WHERE symbol = %s ORDER BY time",
                    (fila[0],),
                )
                filas = cur.fetchall()
        finally:
            conn.close()

        if not filas:
            raise PolicyRunConfigError(
                f"{asset_id}: cero barras para {fila[0]!r}; no se decide sin evidencia"
            )
        bars = pd.DataFrame(filas, columns=["time", "close"])
        observations = build_observations(spec, bars, decision_cutoff=cutoff)

        ti = context["ti"]
        ti.xcom_push(key=f"observations::{policy_id}", value=observations)
        ti.xcom_push(key=f"decision_cutoff::{policy_id}", value=cutoff)
        return {"features": sorted(observations), "decision_cutoff": cutoff}

    return _produce


def make_resolve_snapshot(policy_id: str):
    """Tarea 1: materializar el snapshot causal. Aqui viven el cutoff Y la frescura."""

    def _resolve(**context):
        from src.orchestration.feature_snapshot import resolve_feature_snapshot

        ti = context["ti"]
        observations = ti.xcom_pull(key=f"observations::{policy_id}")
        decision_cutoff = ti.xcom_pull(key=f"decision_cutoff::{policy_id}")
        if not observations or not decision_cutoff:
            raise PolicyRunConfigError(
                f"{policy_id}: faltan observations/decision_cutoff; no se evalua a ciegas"
            )
        resuelto = resolve_feature_snapshot(observations, decision_cutoff=decision_cutoff)
        # El hecho de frescura se PRODUCE aqui, con la evidencia delante, y viaja por
        # XCom. `_policy_context` lo consume; si no llega, falla cerrado.
        ti.xcom_push(
            key=f"{STALENESS_XCOM_KEY}::{policy_id}",
            value=_derive_staleness(
                policy_id, observations, decision_cutoff, _spec_for(policy_id)
            ),
        )
        return resuelto

    return _resolve


def _spec_for(policy_id: str) -> dict:
    """Resolver `policy_id` -> spec. UNA implementacion para los tres eslabones.

    Los tres lo hacian mal y de dos formas distintas (CXD-598 encontro dos, la
    tercera es de la misma raiz):
      * validate y evaluate llamaban `build_policy(policy_id)`, pero `build_policy`
        recibe el **spec** (`Mapping`), no un id: `AttributeError: 'str' object has
        no attribute 'get'`;
      * publish llamaba `load_policy_spec(policy_id)`, pero ese recibe una **ruta**:
        habria muerto con FileNotFoundError.
    Es decir: la cadena se veia entera en el grafo y **ninguno de sus tres ultimos
    eslabones podia ejecutarse**. Estructura verde, ejecucion imposible — justo lo
    que R3 decia cerrar.

    El loader no expone lookup por id, asi que se indexa aqui, en un solo sitio.
    """
    from src.strategies.policies.loader import load_all_policy_specs

    by_id = {str(spec["id"]): spec for spec in load_all_policy_specs()}
    if policy_id not in by_id:
        raise PolicyRunConfigError(
            f"{policy_id}: el loader SSOT no conoce esta policy (conocidas: "
            f"{sorted(by_id)})"
        )
    return by_id[policy_id]


def _declared_fallbacks(spec: dict) -> dict:
    """Los fallbacks DECLARADOS por la policy, no los defaults del runner.

    `spx500_daily_ma200_v1` declara `stale_input_policy: FLAT`; la cadena no pasaba
    ninguno y el runner aplicaba su default `FAIL_CLOSED`. O sea: un snapshot stale
    habria **cerrado la tarea** en vez de emitir el FLAT explicito que la policy
    declara. El invariante 9 dice "sin default, sin freeze" y la cadena estaba
    ejecutando el default. Ausente en el spec => FAIL_CLOSED, que es la direccion
    segura y ademas la que el propio runner ya toma.
    """
    bloque = spec.get("policy") or {}
    return {
        "missing_input_policy": bloque.get("missing_input_policy", "FAIL_CLOSED"),
        "stale_input_policy": bloque.get("stale_input_policy", "FAIL_CLOSED"),
    }


def _policy_context(policy_id: str, context: dict):
    """`PolicyContext` DETERMINISTA, compartido por validate y evaluate.

    Antes se leia `context.get("ctx")`, una clave que **Airflow no inyecta jamas**
    y que ningun `op_kwargs` producia: siempre `None`, y `None.extras` reventaba.
    Ahora se construye del intervalo logico de la corrida — no de `now()` — para que
    dos re-ejecuciones de la misma fecha produzcan el mismo contexto; si no, el
    replay dejaria de ser replay.

    `snapshot_is_stale` es un hecho que el llamador DECLARA (`extras`, bool estricto).
    La cadena aun no tiene un medidor de staleness propio, asi que declara `False`
    explicitamente en vez de omitirlo: omitirlo dejaria el fallback stale sin
    ejercicio posible y el eslabon `FLAT` seria inalcanzable por construccion.
    """
    from src.contracts.policy import PolicyContext

    inicio = context.get("data_interval_end") or context.get("data_interval_start")
    if inicio is None:
        raise PolicyRunConfigError(
            f"{policy_id}: la corrida no expone `data_interval_*`; un contexto sin "
            f"`as_of` publicaria una decision sin fecha logica"
        )
    as_of = inicio.isoformat() if hasattr(inicio, "isoformat") else str(inicio)
    # El hecho de frescura se CONSUME, no se inventa. R4 hacia
    # `context.get("snapshot_is_stale", False)`: ningun productor entregaba esa
    # clave, asi que toda corrida productiva declaraba "fresco" sin medir nada y
    # el fallback FLAT era inalcanzable (CXD-600). Ahora lo produce `resolve` con
    # la evidencia delante y su ausencia es error, no un "no".
    ti = context.get("ti")
    stale = (
        ti.xcom_pull(key=f"{STALENESS_XCOM_KEY}::{policy_id}") if ti is not None else None
    )
    # `None` se TRANSPORTA (CXD-606 §2): significa "no medible porque el conjunto
    # requerido esta incompleto", y quien decide entonces es el `missing_input_policy`
    # declarado, que el runner resuelve ANTES. Abortar aqui convertiria una ausencia
    # con `missing: FLAT` declarado en un error duro -- ignorar otra vez un fallback
    # declarado, que es el pecado original de toda esta serie. Lo que NO se acepta es
    # cualquier otra cosa: un string o un int ahi es un productor roto, no un matiz.
    if stale is not None and not isinstance(stale, bool):
        raise PolicyRunConfigError(
            f"{policy_id}: hecho de frescura invalido ({stale!r}); se espera bool o "
            f"None (no medible)"
        )
    return PolicyContext(as_of=as_of, mode="DECISION", extras={"snapshot_is_stale": stale})


def make_validate_inputs(policy_id: str):
    """Tarea 2: aplicar los fallbacks DECLARADOS a los inputs, ANTES de evaluar.

    Vive como tarea propia porque el pipeline declarado es
    `resolve -> validate -> evaluate -> publish` y hasta ahora el segundo eslabon estaba
    DENTRO del tercero: una validacion fallida no era observable: se veia como "evaluate
    fallo". Devuelve `None` si los inputs valen (sigue la cadena) o la decision FLAT
    degradada, que `evaluate` respeta en vez de recalcular.
    """

    def _validate(**context):
        from src.policy_engine import validate_policy_inputs
        from src.strategies.policies.loader import build_policy

        ti = context["ti"]
        snapshot = ti.xcom_pull(task_ids=f"policy_{policy_id}_resolve_snapshot")
        if snapshot is None:
            raise PolicyRunConfigError(f"{policy_id}: sin snapshot resuelto; no se valida")
        spec = _spec_for(policy_id)
        return validate_policy_inputs(
            build_policy(spec),
            snapshot,
            _policy_context(policy_id, context),
            **_declared_fallbacks(spec),
        )

    return _validate


def make_evaluate_policy(policy_id: str):
    """Tarea 3: evaluar la politica sobre el snapshot ya acotado por cutoff."""

    def _evaluate(**context):
        from src.policy_engine import evaluate_policy
        from src.strategies.policies.loader import build_policy

        ti = context["ti"]
        snapshot = ti.xcom_pull(task_ids=f"policy_{policy_id}_resolve_snapshot")
        if snapshot is None:
            raise PolicyRunConfigError(f"{policy_id}: sin snapshot resuelto; no se evalua")
        # Si la validacion ya degrado a FLAT, esa ES la decision: recalcularla aqui haria
        # que el eslabon de validacion fuera decorativo.
        degraded = ti.xcom_pull(task_ids=f"policy_{policy_id}_validate_inputs")
        if degraded is not None:
            return degraded
        spec = _spec_for(policy_id)
        return evaluate_policy(
            build_policy(spec),
            snapshot,
            _policy_context(policy_id, context),
            **_declared_fallbacks(spec),
        )

    return _evaluate


def _canonical_instrument_id(spec: dict) -> str:
    """`instrument_id` canonico del activo que la policy declara.

    Se resuelve contra `reference.instrument` --la espina de BL-37-- y NO por convencion
    de nombres: `asset_id` y `canonical_symbol` son cosas distintas, y adivinar cual toca
    es como se rompieron los joins que BL-37 existe para arreglar.
    """
    # `asset` es una CADENA en los cuatro specs vigentes (medido). La version
    # anterior hacia `(spec.get("asset") or {}).get("id") or spec.get("asset")`:
    # el primer termino levanta `AttributeError: 'str' object has no attribute
    # 'get'` para el 100% de los specs reales, y el fallback tras el `or` era
    # codigo muerto que no se alcanzaba nunca. Cuarto crash de la misma familia
    # que los tres de CXD-598: la cadena entera se escribio contra formas
    # supuestas en vez de contra los specs que existen.
    declarado = spec.get("asset")
    asset_id = declarado.get("id") if isinstance(declarado, dict) else declarado
    if not isinstance(asset_id, str) or not asset_id:
        raise PolicyRunConfigError(
            f"{spec.get('id')}: no declara `asset`; sin activo no hay instrumento canonico"
        )

    from utils.dag_common import get_db_connection

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT instrument_id::text FROM reference.instrument WHERE asset_id = %s",
                (asset_id,),
            )
            fila = cur.fetchone()
    finally:
        conn.close()

    if not fila:
        raise PolicyRunConfigError(
            f"{asset_id}: sin fila en reference.instrument. La espina debe estar poblada "
            "antes de publicar señales: una señal sin instrumento canonico no se puede "
            "unir a nada"
        )
    return fila[0]


def make_publish_signal(policy_id: str):
    """Tarea 3: publicar la decision. Sin decision no se publica nada."""

    def _publish(**context):
        """Publicar la decision como `StrategySignalRecord`.

        `publish_signal` exige cinco keyword-args obligatorios --`policy_version_id`,
        `instrument_id`, `valid_from`, `valid_until`, `created_at`--. La primera version
        de esta tarea llamaba `publish_signal(decision)` a secas: emitia la tarea y
        **crasheaba con TypeError al ejecutarla**. La cadena existia y no podia
        atravesarse, que es peor que no tenerla, porque el grafo la mostraba.

        Los cinco salen de fuentes declaradas, ninguno se inventa:
          * `policy_version_id` y `instrument_id` del propio spec y de la espina
            canonica (`reference.instrument`), no de una convencion de nombres;
          * la ventana de validez del intervalo de datos del DAG, que es la ventana
            que la decision gobierna;
          * `created_at` del instante logico de la corrida, NO de `now()`: dos
            re-ejecuciones de la misma fecha deben producir el mismo registro, o el
            replay dejaria de ser replay.
        """
        from src.policy_engine import publish_signal

        ti = context["ti"]
        decision = ti.xcom_pull(task_ids=f"policy_{policy_id}_evaluate")
        if decision is None:
            raise PolicyRunConfigError(f"{policy_id}: sin decision; no se publica")

        # `load_policy_spec` recibe una RUTA, no un id: llamarlo con el id habria
        # muerto con FileNotFoundError. Mismo defecto que CXD-598 encontro en
        # validate/evaluate, tercer eslabon incluido. Un solo resolver: `_spec_for`.
        spec = _spec_for(policy_id)
        version_id = (spec.get("governance") or {}).get("policy_hash") or spec.get("version")
        if not version_id:
            raise PolicyRunConfigError(
                f"{policy_id}: sin `policy_hash` ni `version` declarados; una señal sin "
                "identidad de politica no es auditable"
            )

        instrument_id = _canonical_instrument_id(spec)
        inicio = context["data_interval_start"]
        fin = context["data_interval_end"]

        return publish_signal(
            decision,
            policy_version_id=str(version_id),
            instrument_id=instrument_id,
            valid_from=inicio.isoformat(),
            valid_until=fin.isoformat(),
            # Instante LOGICO, no `now()`: dos re-ejecuciones de la misma fecha deben
            # producir el mismo registro.
            created_at=fin.isoformat(),
        )

    return _publish


def _build_asset_dag(asset_id: str, spec: dict, registry_root: str) -> DAG:
    """Build a single DS-cycle pipeline DAG for one asset."""
    stages = spec.get("stages") or []
    verify_spec = spec.get("verify") or {}

    dag = DAG(
        dag_id=f"asset_{asset_id}_pipeline_weekly",
        default_args=DEFAULT_ARGS,
        description=f"DS-cycle pipeline for {spec.get('display_name', asset_id)} "
                    f"(ingest -> backtest/publish -> verify)",
        schedule=spec.get("schedule"),
        start_date=days_ago(1),
        catchup=False,
        tags=["asset-pipeline", "multi-asset", asset_id, "weekly",
              "ds-cycle", "ctr-asset-pipeline-001"],
        max_active_runs=1,
    )

    with dag:
        prev = None
        for stage in stages:
            graceful = bool(stage.get("graceful", False))
            task = PythonOperator(
                task_id=stage["id"],
                python_callable=_run_stage,
                op_kwargs={
                    "script": stage["script"],
                    "args": stage.get("args", []),
                    "stage_name": f"{asset_id}:{stage['id']}",
                },
                # If the *previous* stage was graceful, still run this one so a
                # stale-feed ingest failure never blocks the science stage.
                trigger_rule=(TriggerRule.ALL_DONE if (prev is not None and prev.get("graceful"))
                              else TriggerRule.ALL_SUCCESS),
                # A graceful stage (e.g. best-effort ingest refresh) fails fast:
                # retrying it only stalls the downstream science stage, which runs
                # on the last good seed anyway.
                retries=(0 if graceful else DEFAULT_ARGS["retries"]),
            )
            if prev is not None:
                dag.get_task(prev["id"]) >> task
            prev = {"id": stage["id"], "graceful": graceful}

        verify = PythonOperator(
            task_id="l6_verify_registry",
            python_callable=_make_verify(
                registry_root=registry_root,
                registry_asset=verify_spec.get("registry_asset", asset_id),
                strategy_ids=verify_spec.get("strategy_ids", []),
            ),
            # Verify only after the (non-graceful) publish stage actually succeeds.
            trigger_rule=TriggerRule.ALL_SUCCESS,
        )
        if prev is not None:
            dag.get_task(prev["id"]) >> verify

        # C-010 R3: cadena gobernada por `engine.type`, solo para elegibles.
        # Sin `policy_runs` declarados esto es un bucle vacio y el grafo no cambia.
        for run in resolve_policy_runs(spec):
            policy_id = run["policy_id"]
            chain = [
                PythonOperator(
                    task_id=f"policy_{policy_id}_produce_observations",
                    python_callable=make_produce_observations(policy_id),
                ),
                PythonOperator(
                    task_id=f"policy_{policy_id}_resolve_snapshot",
                    python_callable=make_resolve_snapshot(policy_id),
                ),
                PythonOperator(
                    task_id=f"policy_{policy_id}_validate_inputs",
                    python_callable=make_validate_inputs(policy_id),
                ),
                PythonOperator(
                    task_id=f"policy_{policy_id}_evaluate",
                    python_callable=make_evaluate_policy(policy_id),
                ),
                PythonOperator(
                    task_id=f"policy_{policy_id}_publish",
                    python_callable=make_publish_signal(policy_id),
                ),
            ]
            verify >> chain[0] >> chain[1] >> chain[2] >> chain[3] >> chain[4]

    return dag


# --- Factory: register one DAG per enabled asset in the module globals --------
_config = _load_config()
_registry_root = _config.get("registry_root", "usdcop-trading-dashboard/public/data")

for _asset_id, _spec in (_config.get("assets") or {}).items():
    if not _spec.get("enabled", False):
        continue
    _dag = _build_asset_dag(_asset_id, _spec, _registry_root)
    globals()[f"asset_{_asset_id}_pipeline_weekly"] = _dag
