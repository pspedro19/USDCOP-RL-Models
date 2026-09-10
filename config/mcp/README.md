# MCP — configuración local de desarrollo

`claude_desktop_config.json` es un **ejemplo local**, no configuración desplegada.

El servidor MCP de noticias (`src/news_engine/mcp_server.py`) es **development-only y no está
soportado en producción** (decisión del operador, 2026-08-04). No es una restricción nueva: no
aparece en ningún `docker-compose`, `Dockerfile`, script de arranque ni DAG, y la librería `mcp`
no está instalada en los contenedores del stack.

## El `cwd` del JSON está obsoleto — a propósito no se corrige aquí

Apunta a un árbol de usuario distinto del de este repositorio. **Se deja tal cual** porque adivinar
la ruta local de quien lo use sería peor que declararlo: si vas a usar la herramienta, ajusta el
`cwd` a tu propio checkout. Que esté obsoleto es, además, la evidencia más clara de que no hay
consumidor operativo.

## Qué NO es esta herramienta

Su tabla `news_articles_search` **no es el almacén canónico de noticias**. El canónico es
`news_articles` (migración `045`), que escriben los DAGs productivos. La tabla de búsqueda:

- se crea **fuera de todo plan de migración revisado**, por `mcp_server.py --init-db` (opt-in
  explícito; el arranque por defecto no toca el esquema) o por `scripts/ops/migrate_csv_to_pg.py`;
- pierde todo lo que produce el enriquecimiento (`content`, `summary`, `keywords`, `entities`,
  `raw_json`, `sentiment_*`, …);
- es **más estricta en unicidad** (`url_hash` global frente a `(source_id, url_hash)`), así que
  copiar el canónico hacia ella no truncaría columnas: colapsaría filas.

`tests/unit/test_mcp_dev_only.py` impide que entre en un plan gobernado por deriva. Gobernarla
exigiría una decisión explícita del operador, no un commit.

## Deuda conocida

Los filtros de fecha del camino PostgreSQL están rotos (`str` contra un parámetro
`$n::timestamptz`; asyncpg exige `datetime`). Afecta a `_pg_search`, `_pg_top_headlines`,
`by_category` y `_pg_source_stats`. **No se arregla** por ser dev-only; queda declarado para no
volver a descubrirlo.
