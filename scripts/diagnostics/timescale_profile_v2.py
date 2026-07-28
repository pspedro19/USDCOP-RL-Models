"""Read-only TimescaleDB physical profile query contract (no database startup)."""

TIMESCALE_PROFILE_QUERY = """
SELECT h.hypertable_schema, h.hypertable_name, h.chunk_time_interval,
       c.view_schema, c.view_name
FROM timescaledb_information.hypertables h
LEFT JOIN timescaledb_information.continuous_aggregates c ON TRUE
"""


def profile_query() -> str:
    return TIMESCALE_PROFILE_QUERY
