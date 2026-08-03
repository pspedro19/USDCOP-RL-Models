-- Migration 080 / BL-38 + BL-44
-- Governed Timescale physical profile.  This migration records and exposes the
-- policy; it deliberately does not convert an empty/small table during deploy.
--
-- DESTRUCTIVE RETENTION IS OUT OF SCOPE.  Cold archive restore evidence must
-- exist before a later, separately reviewed retention migration may be proposed.

CREATE SCHEMA IF NOT EXISTS control;
CREATE SCHEMA IF NOT EXISTS market;

CREATE TABLE IF NOT EXISTS control.timescale_physical_profile (
    relation_name REGCLASS PRIMARY KEY,
    time_column NAME NOT NULL,
    chunk_time_interval INTERVAL NOT NULL CHECK (chunk_time_interval > INTERVAL '0'),
    compress_segmentby NAME NOT NULL,
    minimum_rows BIGINT NOT NULL CHECK (minimum_rows > 0),
    classification TEXT NOT NULL CHECK (
        classification IN ('SSOT', 'PROJECTION', 'CACHE', 'DEPRECATED')
    ),
    archive_restore_evidence TEXT,
    applied_at TIMESTAMPTZ,
    applied_by TEXT,
    CHECK (
        archive_restore_evidence IS NULL
        OR archive_restore_evidence ~ '^sha256:[0-9a-f]{64}$'
    )
);

INSERT INTO control.timescale_physical_profile (
    relation_name,
    time_column,
    chunk_time_interval,
    compress_segmentby,
    minimum_rows,
    classification
)
SELECT relation_name, time_column, chunk_interval, segment_column, minimum_rows,
       classification
FROM (
    VALUES
        (
            'market.raw_bar'::REGCLASS,
            'event_time'::NAME,
            INTERVAL '7 days',
            'instrument_id'::NAME,
            100000::BIGINT,
            'SSOT'::TEXT
        ),
        (
            'market.canonical_bar'::REGCLASS,
            'event_time'::NAME,
            INTERVAL '7 days',
            'instrument_id'::NAME,
            100000::BIGINT,
            'PROJECTION'::TEXT
        )
) AS desired(
    relation_name,
    time_column,
    chunk_interval,
    segment_column,
    minimum_rows,
    classification
)
ON CONFLICT (relation_name) DO UPDATE
SET time_column = EXCLUDED.time_column,
    chunk_time_interval = EXCLUDED.chunk_time_interval,
    compress_segmentby = EXCLUDED.compress_segmentby,
    minimum_rows = EXCLUDED.minimum_rows,
    classification = EXCLUDED.classification;

CREATE OR REPLACE FUNCTION control.assert_timescale_unique_keys(
    p_relation REGCLASS,
    p_time_column NAME
)
RETURNS VOID AS $$
DECLARE
    invalid_indexes TEXT[];
BEGIN
    SELECT ARRAY_AGG(index_name ORDER BY index_name)
    INTO invalid_indexes
    FROM (
        SELECT ci.relname AS index_name
        FROM pg_index i
        JOIN pg_class ci ON ci.oid = i.indexrelid
        WHERE i.indrelid = p_relation
          AND i.indisunique
          AND NOT EXISTS (
              SELECT 1
              FROM unnest(i.indkey) AS key(attnum)
              JOIN pg_attribute a
                ON a.attrelid = p_relation
               AND a.attnum = key.attnum
              WHERE a.attname = p_time_column
          )
    ) AS incompatible;

    IF COALESCE(cardinality(invalid_indexes), 0) > 0 THEN
        RAISE EXCEPTION
            'cannot convert %: unique indexes must include % (incompatible=%)',
            p_relation::TEXT,
            p_time_column,
            invalid_indexes;
    END IF;
END;
$$ LANGUAGE plpgsql STABLE;

CREATE OR REPLACE FUNCTION control.apply_timescale_physical_profile(
    p_relation REGCLASS,
    p_archive_restore_evidence TEXT,
    p_operator TEXT
)
RETURNS JSONB AS $$
DECLARE
    profile control.timescale_physical_profile%ROWTYPE;
    row_count BIGINT;
    is_hypertable BOOLEAN;
BEGIN
    IF p_archive_restore_evidence IS NULL
       OR p_archive_restore_evidence !~ '^sha256:[0-9a-f]{64}$' THEN
        RAISE EXCEPTION
            'cold archive restore evidence must be a sha256 fingerprint';
    END IF;
    IF NULLIF(BTRIM(p_operator), '') IS NULL THEN
        RAISE EXCEPTION 'operator identity is required';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'
    ) THEN
        RAISE EXCEPTION 'timescaledb extension is not installed';
    END IF;

    SELECT *
    INTO profile
    FROM control.timescale_physical_profile
    WHERE relation_name = p_relation
    FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'no governed Timescale profile for %', p_relation::TEXT;
    END IF;

    EXECUTE format('SELECT count(*) FROM %s', p_relation) INTO row_count;
    IF row_count < profile.minimum_rows THEN
        RAISE EXCEPTION
            'refusing premature hypertable for %: % rows < governed minimum %',
            p_relation::TEXT,
            row_count,
            profile.minimum_rows;
    END IF;

    PERFORM control.assert_timescale_unique_keys(
        p_relation, profile.time_column
    );

    SELECT EXISTS (
        SELECT 1
        FROM timescaledb_information.hypertables h
        WHERE format('%I.%I', h.hypertable_schema, h.hypertable_name)
              = p_relation::TEXT
    )
    INTO is_hypertable;

    IF NOT is_hypertable THEN
        -- The call is operator-triggered after the row-count, key-shape and
        -- archive checks above; deployment itself never takes this lock.
        PERFORM create_hypertable(
            p_relation::TEXT,
            profile.time_column::TEXT,
            chunk_time_interval => profile.chunk_time_interval,
            if_not_exists => TRUE,
            migrate_data => TRUE
        );
    END IF;

    EXECUTE format(
        'ALTER TABLE %s SET (timescaledb.compress, '
        'timescaledb.compress_segmentby = %L)',
        p_relation,
        profile.compress_segmentby
    );

    UPDATE control.timescale_physical_profile
    SET archive_restore_evidence = p_archive_restore_evidence,
        applied_at = NOW(),
        applied_by = p_operator
    WHERE relation_name = p_relation;

    RETURN jsonb_build_object(
        'relation', p_relation::TEXT,
        'rows', row_count,
        'chunk_time_interval', profile.chunk_time_interval,
        'compress_segmentby', profile.compress_segmentby,
        'archive_restore_evidence', p_archive_restore_evidence
    );
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION market.install_canonical_continuous_aggregates(
    p_operator TEXT
)
RETURNS VOID AS $$
DECLARE
    interval_spec RECORD;
BEGIN
    IF NULLIF(BTRIM(p_operator), '') IS NULL THEN
        RAISE EXCEPTION 'operator identity is required';
    END IF;
    IF NOT EXISTS (
        SELECT 1
        FROM timescaledb_information.hypertables
        WHERE hypertable_schema = 'market'
          AND hypertable_name = 'canonical_bar'
    ) THEN
        RAISE EXCEPTION
            'market.canonical_bar must pass the governed hypertable preflight first';
    END IF;

    FOR interval_spec IN
        SELECT *
        FROM (
            VALUES
                ('1h'::TEXT, INTERVAL '1 hour'),
                ('4h'::TEXT, INTERVAL '4 hours'),
                ('1d'::TEXT, INTERVAL '1 day')
        ) AS intervals(view_suffix, bucket_width)
    LOOP
        EXECUTE format(
            'CREATE MATERIALIZED VIEW IF NOT EXISTS market.canonical_bar_%I '
            'WITH (timescaledb.continuous) AS '
            'SELECT instrument_id, bar_method, '
            'time_bucket(%L::interval, event_time) AS event_time, '
            'first(open, event_time) AS open, '
            'max(high) AS high, min(low) AS low, '
            'last(close, event_time) AS close, '
            'sum(volume) AS volume, max(available_at) AS available_at '
            'FROM market.canonical_bar '
            'WHERE interval_id = ''PT5M'' '
            'GROUP BY instrument_id, bar_method, '
            'time_bucket(%L::interval, event_time) '
            'WITH NO DATA',
            interval_spec.view_suffix,
            interval_spec.bucket_width,
            interval_spec.bucket_width
        );
    END LOOP;
END;
$$ LANGUAGE plpgsql;

COMMENT ON TABLE control.timescale_physical_profile IS
    'BL-44 governed physical intent; no table is converted merely by deployment.';
COMMENT ON FUNCTION control.apply_timescale_physical_profile(
    REGCLASS, TEXT, TEXT
) IS
    'Operator-only conversion after scale, unique-key, and cold archive restore evidence checks.';
