# AUDIT-CLAUDE — la brecha de cableado (medida, no narrada)

> Autor: CLAUDE `claude-root-152c263e-r2` · 2026-08-03 · snapshot `6d3a123c`
> Motivo: en `CLD-315` afirmé que hay mecanismos correctos, con tests, que **nadie invoca**.
> Lo había encontrado **a mano** (tres casos). Una afirmación así no puede quedarse en anécdota:
> o se mide o no se dice. Esto la mide.

## Qué se midió y cómo

Para cada símbolo público *top-level* de los módulos de garantía (`src/{identity,orchestration,
portfolio,data_quality,research,governance,metrics,market,policy_engine,contracts}`) se cuenta
si alguien lo referencia desde **producción** (`airflow/`, `scripts/`, `services/`, `src/`,
`app/`) y desde **tests**, excluyendo siempre el fichero que lo define. Las líneas de docstring y
comentario **no cuentan** como uso: se descartan por AST. Script:
`scratchpad/wiring_audit.py` (temporal, no versionado).

**Corrección de método aplicada durante la propia medición.** La primera pasada daba
`src/data_quality/rules.py` como cableado. Era falso: lo único que lo referenciaba era
`src/data_quality/__init__.py` re-exportándolo en `__all__`. **Un re-export no es un llamador**,
así que se excluyeron los `__init__.py` de "producción". Con eso, los módulos muertos pasaron de
4 a 9. Se deja escrito porque es exactamente el tipo de falso verde que esta auditoría busca.

## Resultado

| Medida | Valor |
|---|---:|
| Símbolos públicos analizados | 243 |
| Con tests y **cero** llamadores productivos | **61** |
| **Módulos con superficie pública 100% sin cablear** | **9 de 41 (22%)** |

El número que vale es el de **módulos**, no el de símbolos. 61 símbolos exagera: mezcla
mecanismos realmente muertos con superficie no usada de módulos que sí están vivos —
`write_approval` no tiene llamador, pero `approval_store` está cableado en ocho scripts vía
`approval_path`/`commit_approval_transition`. Un módulo entero sin un solo llamador no admite esa
lectura benigna.

### Los nueve

| Módulo | Públicos | Con tests | BL | ¿Lo explica `fabric-v1`? |
|---|---:|---:|---|---|
| `src/portfolio/allocator.py` | 8 | 3 | BL-27 | **Sí** |
| `src/portfolio/snapshot.py` | 7 | 6 | BL-26 | **Sí** |
| `src/governance/declaration.py` | 6 | 5 | BL-17 | **Sí** |
| `src/contracts/news_engine_schema.py` | 5 | 5 | — | **NO** ← ver abajo |
| `src/market/resampling.py` | 4 | 3 | BL-38 | **Sí** |
| `src/data_quality/rules.py` | 3 | 1 | BL-40 | **Sí** (072/073) |
| `src/metrics/persistence.py` | 3 | 1 | BL-18 | **Sí** (070) |
| `src/governance/synthetic_isolation.py` | 2 | 2 | BL-43 | **Sí** |
| `src/orchestration/feature_snapshot.py` | 2 | 2 | BL-45 #6 | Parcial (es R3) |

## La lectura honesta: ocho de nueve NO son un defecto de nadie

**Esto no acusa a nadie de escribir código muerto.** Ocho de los nueve son módulos de la fábrica
de control cuyo esquema (`fabric-v1`, migraciones 070–081) **no está aplicado**. No es que nadie
los haya cableado: es que **no se pueden cablear todavía**, porque las tablas contra las que
operan no existen. Verificado en la DB viva: de los diez esquemas de fabric sólo existe `demo`.

Lo que sí hace esta medición es **poner precio a la decisión pendiente**. El pin de `fabric-v1`
no bloquea "BL-18 integration" como una casilla suelta: mantiene **ocho módulos completos** en
estado de no proteger nada en ejecución, todos con tests verdes. Ése es el coste real de la
decisión, y hasta ahora se estaba contando como una línea de tablero.

## La excepción, que sí es deriva y no dependencia

`src/contracts/news_engine_schema.py` **no lo explica `fabric-v1`.** El News Engine está
**operativo** (78 + 276 artículos en DB, adapters corriendo, DAGs 3×/día) y `CLAUDE.md:140` lo
declara como su contrato Python:

> \| `news_engine_schema.py` \| `ArticleRecord`, `DigestRecord`, `FeatureSnapshotRecord`, `CrossReferenceRecord` \|

**`src/news_engine/` no importa ese módulo en ninguna parte.** Define sus propios `@dataclass`
(`config.py` y compañía) y nunca toca los cuatro records declarados. Es decir: el contrato que la
documentación presenta como la interfaz del subsistema **no es el que el subsistema usa**.

Consecuencia práctica, y es la de siempre con un contrato paralelo: los cinco candados que
prueban `ArticleRecord` prueban una forma que **ningún dato de producción atraviesa**. Si mañana
el adapter real cambia de forma, esos tests siguen verdes. La regla DRY del propio `CLAUDE.md`
—"same feature code for training and inference; never duplicate"— está incumplida aquí, con la
agravante de que la copia no usada es la **declarada**.

No se corrige en esta auditoría: decidir cuál de los dos es el contrato bueno es cambio de SSOT
y toca `CLAUDE.md`, que no se edita sin instrucción del operador.

## Qué propongo que cambie (y qué no)

1. **"Cableado" como requisito explícito de DONE.** Un candado verde sobre una función que nadie
   llama prueba que la función es correcta, **no** que la garantía esté vigente. Hoy tres fichas
   podrían haber sumado DONE sin que el sistema estuviera un gramo más protegido.
2. **No** proponer que se retiren estos módulos ni que se relajen sus tests: ocho de nueve están
   esperando una autorización, no un arreglo. Borrarlos sería el error opuesto.
3. Para los `dependency-blocked`, que la ficha **nombre la dependencia concreta** (072/073/070)
   en vez de decir "PARTIAL". Ya se hizo en BL-40 tras `CLD-312`; conviene para los otros siete.

## Reproducir

```
python scratchpad/wiring_audit.py     # imprime las tres tablas de arriba
```

Medido contra `6d3a123c`, árbol limpio salvo canales runtime.

## Correccion (2026-08-04): eran **10 de 41**, no 9 — falso negativo por colision de nombre

`src/policy_engine/runner.py` **debio aparecer en la tabla y no aparecio**. Su superficie publica
—`evaluate_policy`, `publish_signal`, `write_policy_version_index`, `_flat_decision`— tiene **cero
llamadores productivos**. Lo comprobado, fichero a fichero:

| Aparicion de `evaluate_policy` fuera de `policy_engine/` | Que es |
|---|---|
| `src/contracts/policy.py:72` | docstring |
| `src/strategies/policies/loader.py:53` | comentario (`#:`) |
| `src/experiments/experiment_runner.py:391` | **otra funcion**: `stable_baselines3.common.evaluation.evaluate_policy` |
| `src/ml_workflow/training_callbacks.py:259` | **otra funcion**: la misma de `stable_baselines3` |

Y `publish_signal` fuera del modulo es un **metodo de otra clase**
(`services/common/redis_streams_manager.py:455`), no esta funcion.

**Por que mi propio script no lo vio:** cuenta apariciones del **nombre**. Dos librerias distintas
exportan un simbolo llamado `evaluate_policy`, asi que el homonimo de `stable_baselines3` se conto
como "llamador productivo" del nuestro. **La medicion heredo el defecto que la propia auditoria
denuncia**: una busqueda por nombre no es una medicion de capacidad.

**Consecuencia sobre el numero:** el conteo pasa de **9/41 a 10/41 (24%)**, y el sesgo del metodo
es **optimista** — puede haber mas falsos negativos por la misma causa. El numero publicado debe
leerse como **cota inferior**, no como censo.

**Consecuencia sobre BL-45 R3, que es lo que lo destapo:** no existe "el menor slice productivo"
que encadene `resolve_feature_snapshot` antes de evaluar, porque **`evaluate_policy` tampoco se
invoca desde produccion**. Encadenar el cutoff ahi uniria dos mecanismos que nadie llama. El
cableado real de R3 empieza mas arriba: alguien tiene que invocar el motor de politicas.
