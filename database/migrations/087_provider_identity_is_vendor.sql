-- 087_provider_identity_is_vendor.sql
-- C038 · reference.provider = VENDEDOR, no ruta de ingesta.
--
-- QUE ARREGLA
-- -----------
-- `reference.provider` tenia 14 filas de las que solo 3 eran proveedores reales (binance,
-- investing, twelvedata: las unicas con `authoritative_for` no vacio). Las otras 11 eran
-- RUTAS DE INGESTA ascendidas a identidad de primera clase, con FK entrante desde
-- `market.raw_bar`: twelvedata_backfill/_gap_fill/_daily/_daily_deep/_multi/_xauusd,
-- binance_daily/_btcusdt, investing_daily, y -- la que lo delata -- twelvedata_manual_test.
-- Una prueba manual era un proveedor de primera clase.
--
-- La propia metadata lo confirma: las 11 rutas llevan `observed_writer=true` y
-- `declared_authority=false` (se derivaron de QUIEN ESCRIBIO filas), mientras los 3
-- vendedores llevan `declared_authority=true`. La dimension mezclaba lo observado con lo
-- declarado y presentaba ambos como identidad.
--
-- POR QUE AHORA Y NO DESPUES
-- --------------------------
-- `market.raw_bar` y `market.canonical_bar` tienen 0 filas (medido en preflight). Hoy esto
-- es renombrar filas de una dimension. En cuanto los writers de BL-38 pueblen millones de
-- barras con provider_id apuntando a rutas, seria una migracion de datos.
--
-- QUE NO HACE, Y POR QUE (declarado, no omitido)
-- ----------------------------------------------
-- 1. NO toca `daily_native`. Etiqueta el origen del seed diario de USD/COP
--    (scripts/ops/seed_session_calendar.py:146) y ese parquet no lleva columna de
--    procedencia -- sus columnas son time/open/high/low/close. El vendedor detras de esa
--    serie NO ES MEDIBLE desde el repo, y asignarle uno seria inventar provenance.
--    Queda como pregunta abierta para el operador.
-- 2. NO crea USD/BRL. Tiene barras vivas y no tiene fila en provider_symbol, pero tampoco
--    tiene AssetProfile (no existe config/assets/usdbrl.yaml): se ingiere como serie
--    correlacionada de COP y nunca se onboardeo. Insertar asset+instrument exigiria
--    declarar asset_class, quote_currency, annualization y calendar de un activo que nadie
--    onboardeo. Es deuda de ONBOARDING, no de identidad, y se declara en vez de taparse.
--
-- INTEGRIDAD
-- ----------
-- El mapa ruta->vendedor se DERIVA (prefijo mas largo que sea vendedor con
-- authoritative_for no vacio), no se escribe a mano: asi `twelvedata_daily_deep` colapsa a
-- `twelvedata` y no a `twelvedata_daily`, que es otra ruta.
-- La metadata se FUSIONA, no se reescribe: cada entrada de `evidence` conserva sus filas
-- medidas y gana `via` con la ruta que la produjo. Las postcondiciones ABORTAN la
-- transaccion si se pierde una sola entrada de evidencia o un instrument_id.
-- Idempotente: en la segunda ejecucion el mapa sale vacio y todo afecta a 0 filas.

BEGIN;

-- ---------------------------------------------------------------- mapa derivado
CREATE TEMP TABLE c038_route_map ON COMMIT DROP AS
SELECT p.provider_id AS route,
       (SELECT v.provider_id
          FROM reference.provider v
         WHERE p.provider_id LIKE v.provider_id || '\_%'
           AND COALESCE(array_length(v.authoritative_for, 1), 0) > 0
         ORDER BY length(v.provider_id) DESC
         LIMIT 1) AS vendor
  FROM reference.provider p;

DELETE FROM c038_route_map WHERE vendor IS NULL;

-- ------------------------------------- cada fila resuelta a su destino (sin agrupar aun)
-- Se resuelve ANTES de agrupar a proposito: intentar correlacionar una subconsulta con
-- `rm.vendor` dentro del GROUP BY es un error de SQL, y lo caza el dry-run, no la revision.
CREATE TEMP TABLE c038_resolved ON COMMIT DROP AS
SELECT COALESCE(rm.vendor, ps.provider_id) AS provider_id,
       ps.provider_symbol,
       ps.instrument_id,
       ps.valid_from,
       ps.valid_until,
       ps.metadata,
       ps.provider_id                      AS origen
  FROM reference.provider_symbol ps
  LEFT JOIN c038_route_map rm ON rm.route = ps.provider_id;

-- --------------------------------------------------- evidencia con su procedencia
CREATE TEMP TABLE c038_evidence ON COMMIT DROP AS
SELECT r.provider_id,
       r.provider_symbol,
       (e || jsonb_build_object('via', r.origen)) AS entry
  FROM c038_resolved r
  CROSS JOIN LATERAL jsonb_array_elements(COALESCE(r.metadata -> 'evidence', '[]'::jsonb)) e;

CREATE TEMP TABLE c038_declared ON COMMIT DROP AS
SELECT r.provider_id, r.provider_symbol, d AS valor
  FROM c038_resolved r
  CROSS JOIN LATERAL jsonb_array_elements_text(COALESCE(r.metadata -> 'declared_for', '[]'::jsonb)) d;

-- ------------------------------------------------------------- destino fusionado
CREATE TEMP TABLE c038_merged ON COMMIT DROP AS
SELECT r.provider_id,
       r.provider_symbol,
       (array_agg(r.instrument_id))[1]                           AS instrument_id,
       min(r.valid_from)                                         AS valid_from,
       max(r.valid_until)                                        AS valid_until,
       jsonb_build_object(
         'declared_authority', bool_or(COALESCE((r.metadata ->> 'declared_authority')::boolean, false)),
         'observed_writer',    bool_or(COALESCE((r.metadata ->> 'observed_writer')::boolean, false)),
         'declared_for',       COALESCE((SELECT jsonb_agg(DISTINCT dd.valor)
                                           FROM c038_declared dd
                                          WHERE dd.provider_id = r.provider_id
                                            AND dd.provider_symbol = r.provider_symbol),
                                        '[]'::jsonb),
         'evidence',           COALESCE((SELECT jsonb_agg(ev.entry)
                                           FROM c038_evidence ev
                                          WHERE ev.provider_id = r.provider_id
                                            AND ev.provider_symbol = r.provider_symbol),
                                        '[]'::jsonb),
         'ingestion_routes',   jsonb_agg(DISTINCT r.origen),
         'c038_note',          'provider = vendedor; la ruta de ingesta vive en ingestion_routes'
       ) AS metadata
  FROM c038_resolved r
 GROUP BY r.provider_id, r.provider_symbol;

-- --------------------------------------------------------- postcondiciones PREVIAS
DO $$
DECLARE
    ambiguos int;
    ev_antes int;
    ev_despues int;
BEGIN
    SELECT count(*) INTO ambiguos FROM (
        SELECT 1 FROM reference.provider_symbol ps
          LEFT JOIN c038_route_map rm ON rm.route = ps.provider_id
         GROUP BY COALESCE(rm.vendor, ps.provider_id), ps.provider_symbol
        HAVING count(DISTINCT ps.instrument_id) > 1) x;
    IF ambiguos > 0 THEN
        RAISE EXCEPTION 'C038 ABORTA: % pares (vendedor,simbolo) apuntan a mas de un instrumento; colapsar exigiria ELEGIR identidad', ambiguos;
    END IF;

    SELECT count(*) INTO ev_antes
      FROM reference.provider_symbol ps
      CROSS JOIN LATERAL jsonb_array_elements(COALESCE(ps.metadata -> 'evidence', '[]'::jsonb)) e;
    SELECT coalesce(sum(jsonb_array_length(metadata -> 'evidence')), 0) INTO ev_despues FROM c038_merged;
    IF ev_antes <> ev_despues THEN
        RAISE EXCEPTION 'C038 ABORTA: la fusion pierde evidencia (% entradas antes, % despues)', ev_antes, ev_despues;
    END IF;
END $$;

-- ------------------------------------------------------------------- aplicacion
DELETE FROM reference.provider_symbol;

INSERT INTO reference.provider_symbol
       (provider_id, provider_symbol, instrument_id, valid_from, valid_until, metadata)
SELECT provider_id, provider_symbol, instrument_id, valid_from, valid_until, metadata
  FROM c038_merged;

DELETE FROM reference.provider
 WHERE provider_id IN (SELECT route FROM c038_route_map);

-- ------------------------------------------------------- postcondiciones FINALES
DO $$
DECLARE
    rutas int;
    huerfanas int;
BEGIN
    SELECT count(*) INTO rutas
      FROM reference.provider a
      JOIN reference.provider b ON a.provider_id LIKE b.provider_id || '\_%'
     WHERE a.provider_id <> b.provider_id;
    IF rutas > 0 THEN
        RAISE EXCEPTION 'C038 ABORTA: quedan % proveedores con forma de ruta', rutas;
    END IF;

    SELECT count(*) INTO huerfanas
      FROM reference.provider_symbol ps
      LEFT JOIN reference.provider p ON p.provider_id = ps.provider_id
     WHERE p.provider_id IS NULL;
    IF huerfanas > 0 THEN
        RAISE EXCEPTION 'C038 ABORTA: % provider_symbol sin proveedor', huerfanas;
    END IF;
END $$;

COMMIT;
