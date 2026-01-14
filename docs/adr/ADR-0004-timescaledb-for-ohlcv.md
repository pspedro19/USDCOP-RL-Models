# ADR-0004: Use TimescaleDB for OHLCV Time-Series Storage

## Status

**Accepted** (2026-01-14)

## Context

The USDCOP RL Trading system requires efficient storage and querying of OHLCV (Open, High, Low, Close, Volume) time-series data for:

- Real-time feature computation
- Historical backtesting
- Training dataset generation
- Analytics and visualization

Requirements:
- Efficient time-range queries (e.g., last 1 hour, specific date range)
- Fast aggregation (e.g., 5-min to 1-hour rollups)
- SQL compatibility for existing tools
- Compression for historical data
- Continuous aggregates for dashboards

Options considered:
1. Plain PostgreSQL
2. TimescaleDB (PostgreSQL extension)
3. InfluxDB
4. QuestDB
5. ClickHouse

## Decision

We chose **TimescaleDB** (PostgreSQL extension) for OHLCV storage.

## Rationale

### Why TimescaleDB over alternatives:

| Criterion | TimescaleDB | PostgreSQL | InfluxDB | QuestDB | ClickHouse |
|-----------|-------------|------------|----------|---------|------------|
| SQL Support | Full | Full | InfluxQL/Flux | SQL | SQL |
| Time-series optimized | Yes | No | Yes | Yes | Yes |
| PostgreSQL compatible | Yes | N/A | No | Partial | No |
| Compression | 90%+ | Limited | Good | Good | Excellent |
| JOINs | Full | Full | Limited | Limited | Limited |
| Ecosystem | PostgreSQL | PostgreSQL | Custom | New | Custom |

### Key advantages:

1. **PostgreSQL compatible** - Existing queries, tools, ORMs work
2. **Hypertables** - Automatic partitioning by time
3. **Compression** - 90%+ compression for historical data
4. **Continuous aggregates** - Materialized views that auto-update
5. **Same database** - No need for separate time-series DB

### Why not InfluxDB?

- Different query language (InfluxQL/Flux)
- Limited JOIN support
- Another system to manage
- Less mature Python ecosystem

### Why not QuestDB/ClickHouse?

- Limited PostgreSQL compatibility
- Additional operational overhead
- Overkill for our data volume (~100K rows/year)

## Consequences

### Positive

- Single database for all data (OHLCV, features, signals)
- Full SQL support with time-series optimizations
- Excellent compression for historical data
- Continuous aggregates for real-time dashboards
- Easy integration with existing tools (pgAdmin, DBeaver, etc.)

### Negative

- Requires TimescaleDB extension installation
- Some advanced features require TimescaleDB license
- Slightly more complex than plain PostgreSQL

### Mitigations

- Use official TimescaleDB Docker image
- Document extension installation in deployment guide
- Monitor query performance and add indexes as needed

## Implementation

### Schema

```sql
-- Create hypertable
CREATE TABLE usdcop_m5_ohlcv (
    time TIMESTAMPTZ NOT NULL,
    open NUMERIC(12,4) NOT NULL,
    high NUMERIC(12,4) NOT NULL,
    low NUMERIC(12,4) NOT NULL,
    close NUMERIC(12,4) NOT NULL,
    volume NUMERIC(18,2)
);

SELECT create_hypertable('usdcop_m5_ohlcv', 'time');

-- Add compression policy
ALTER TABLE usdcop_m5_ohlcv SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = ''
);

SELECT add_compression_policy('usdcop_m5_ohlcv', INTERVAL '7 days');
```

### Continuous Aggregate

```sql
CREATE MATERIALIZED VIEW ohlcv_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    first(open, time) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, time) AS close,
    sum(volume) AS volume
FROM usdcop_m5_ohlcv
GROUP BY bucket;
```

## References

- [TimescaleDB Documentation](https://docs.timescale.com/)
- [TimescaleDB vs PostgreSQL Benchmarks](https://www.timescale.com/blog/timescaledb-vs-6a696248104e/)
- [Continuous Aggregates](https://docs.timescale.com/use-timescale/latest/continuous-aggregates/)
