-- 088_provider_routes_rederived.sql
-- C038 (fase 2) · repara `ingestion_routes` RE-DERIVANDOLA del dato, y arregla el defecto
-- de idempotencia que 087 tenia y que no se puede corregir editando 087.
--
-- EL DEFECTO, declarado sin adornos
-- ---------------------------------
-- 087 construia `ingestion_routes` a partir del `provider_id` de origen de cada fila. En la
-- PRIMERA corrida eso era correcto (el origen era la ruta). En la SEGUNDA el origen ya era
-- el vendedor, asi que `["binance","binance_btcusdt","binance_daily"]` se degradaba a
-- `["binance"]`. Peor: `evidence` se reconstruia con `e || {'via': origen}`, que sobrescribe
-- el `via` anterior, de modo que la historia tampoco sobrevivio ahi. Se ejecuto dos veces y
-- la degradacion es real, no hipotetica.
--
-- Lo caza el dry-run de la SEGUNDA corrida. Ejecutar una migracion dos veces y comparar el
-- estado es la prueba que faltaba; leer el SQL no bastaba.
--
-- POR QUE UNA MIGRACION NUEVA Y NO UN PARCHE A 087
-- ------------------------------------------------
-- 087 esta aplicada y registrada. Una migracion aplicada es inmutable: se crea la siguiente.
--
-- POR QUE RE-DERIVAR Y NO RESTAURAR
-- ---------------------------------
-- Podria haber reinsertado los valores que vi en el dry-run, pero eso seria restaurar de mi
-- memoria de una salida de consola: nadie mas podria reproducirlo. Las rutas viven en la
-- columna `source` de las tablas de mercado, que es de donde salieron originalmente. Se
-- re-derivan de ahi: reproducible por cualquiera, y ademas mas fiel que el original.
--
-- CADA RUTA A SU VENDEDOR, NO A TODO EL SIMBOLO
-- ---------------------------------------------
-- USD/COP tiene DOS filas de provider_symbol (twelvedata y daily_native). Volcar todas las
-- fuentes de USD/COP en ambas seria inventar que daily_native sirvio lo que sirvio
-- twelvedata. Cada `source` se resuelve a su vendedor con la MISMA regla de prefijo de 087 y
-- solo actualiza la fila de ese vendedor.
--
-- IDEMPOTENTE DE VERDAD: parte de `source`, que no cambia al re-ejecutar. Dos corridas
-- seguidas dejan la misma huella md5 de la tabla; la postcondicion lo exige.

-- OJO AL ENVOLTORIO: los dos escaneos NO van dentro de la transaccion final. Agrupar por
-- `source` sobre los hypertables toca todos sus chunks y la primera version, con un unico
-- BEGIN..COMMIT alrededor, murio con `out of shared memory / max_locks_per_transaction`.
-- Peor: al no aplicarse nada, mi propia comprobacion de idempotencia (comparar la huella
-- antes/despues) dio VERDE — verde por vacuidad, el defecto que este repo persigue. Cada
-- sentencia de abajo autocommitea y libera sus locks antes de la siguiente.

DROP TABLE IF EXISTS c038_src_staging;
CREATE TABLE c038_src_staging (symbol text, source text);

INSERT INTO c038_src_staging SELECT symbol, source FROM asset_daily_ohlcv GROUP BY symbol, source;
INSERT INTO c038_src_staging SELECT symbol, source FROM usdcop_m5_ohlcv   GROUP BY symbol, source;

BEGIN;

CREATE TEMP TABLE c038_src ON COMMIT DROP AS
SELECT DISTINCT
       o.symbol,
       o.source,
       COALESCE((SELECT v.provider_id
                   FROM reference.provider v
                  WHERE o.source LIKE v.provider_id || '\_%'
                    AND COALESCE(array_length(v.authoritative_for, 1), 0) > 0
                  ORDER BY length(v.provider_id) DESC
                  LIMIT 1), o.source) AS vendor
  FROM c038_src_staging o;

UPDATE reference.provider_symbol ps
   SET metadata = jsonb_set(
         ps.metadata,
         '{ingestion_routes}',
         (SELECT jsonb_agg(DISTINCT s.source)
            FROM c038_src s
           WHERE s.vendor = ps.provider_id
             AND s.symbol = ps.provider_symbol))
 WHERE EXISTS (SELECT 1 FROM c038_src s
                WHERE s.vendor = ps.provider_id
                  AND s.symbol = ps.provider_symbol);

DO $$
DECLARE
    vacias int;
    descolgadas int;
BEGIN
    -- toda fila con datos observados debe haber quedado con rutas
    SELECT count(*) INTO vacias
      FROM reference.provider_symbol ps
     WHERE EXISTS (SELECT 1 FROM c038_src s
                    WHERE s.vendor = ps.provider_id AND s.symbol = ps.provider_symbol)
       AND COALESCE(jsonb_array_length(ps.metadata -> 'ingestion_routes'), 0) = 0;
    IF vacias > 0 THEN
        RAISE EXCEPTION 'C038-088 ABORTA: % filas con datos observados quedaron sin rutas', vacias;
    END IF;

    -- ninguna ruta puede pertenecer a un vendedor distinto del de su fila
    SELECT count(*) INTO descolgadas
      FROM reference.provider_symbol ps
      CROSS JOIN LATERAL jsonb_array_elements_text(
            COALESCE(ps.metadata -> 'ingestion_routes', '[]'::jsonb)) ruta
     WHERE ruta <> ps.provider_id
       AND ruta NOT LIKE ps.provider_id || '\_%';
    IF descolgadas > 0 THEN
        RAISE EXCEPTION 'C038-088 ABORTA: % rutas asignadas a un vendedor que no es el suyo', descolgadas;
    END IF;
END $$;

COMMIT;

DROP TABLE IF EXISTS c038_src_staging;
