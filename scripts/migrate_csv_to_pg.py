"""
Migrate GDELT historical CSV to PostgreSQL news_articles_search table.

Reads data/news/gdelt_articles_historical.csv, creates the search table if it
does not exist, and UPSERTs rows by url_hash (SHA256 of URL).

Usage:
    python scripts/migrate_csv_to_pg.py              # Full migration
    python scripts/migrate_csv_to_pg.py --dry-run    # Preview without writing

Part of Phase 4: MCP News Server implementation.
"""

import argparse
import hashlib
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "news" / "gdelt_articles_historical.csv"

# ---------------------------------------------------------------------------
# USDCOP relevance scoring (mirrors mcp_data_layer.py)
# ---------------------------------------------------------------------------

RELEVANCE_KEYWORDS = {
    3.0: [
        "usdcop", "dólar", "peso colombiano", "banrep",
        "tasa de cambio", "devaluación", "revaluación",
    ],
    2.0: [
        "petróleo", "wti", "brent", "fed", "fomc", "powell",
        "tasas de interés", "embi", "inflación", "pib colombia",
    ],
    1.0: [
        "commodities", "oro", "dxy", "emergentes", "latam",
        "brasil", "méxico", "opep", "china", "aranceles",
    ],
}


def compute_relevance(title: str, language: str = "en") -> float:
    if not title:
        return 0.0
    title_lower = str(title).lower()
    score = 0.0
    for weight, keywords in RELEVANCE_KEYWORDS.items():
        for kw in keywords:
            if kw in title_lower:
                score += weight
    if language and str(language).lower().startswith("es"):
        score += 0.5
    return round(score, 2)


def url_hash(url: str) -> str:
    return hashlib.sha256(str(url).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS news_articles_search (
    id SERIAL PRIMARY KEY,
    title TEXT,
    source VARCHAR(200),
    domain VARCHAR(200),
    date TIMESTAMPTZ,
    language VARCHAR(10),
    url TEXT,
    tone FLOAT,
    url_hash VARCHAR(64) UNIQUE,
    category VARCHAR(50),
    relevance FLOAT
);
"""

CREATE_INDEXES_SQL = [
    """
    CREATE INDEX IF NOT EXISTS idx_news_search_title_gin
    ON news_articles_search
    USING GIN (to_tsvector('simple', title));
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_news_search_date
    ON news_articles_search (date DESC);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_news_search_category
    ON news_articles_search (category);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_news_search_relevance
    ON news_articles_search (relevance DESC);
    """,
]

UPSERT_SQL = """
INSERT INTO news_articles_search (title, source, domain, date, language, url, tone, url_hash, category, relevance)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (url_hash) DO UPDATE SET
    title = EXCLUDED.title,
    source = EXCLUDED.source,
    domain = EXCLUDED.domain,
    date = EXCLUDED.date,
    language = EXCLUDED.language,
    tone = EXCLUDED.tone,
    category = EXCLUDED.category,
    relevance = EXCLUDED.relevance;
"""


def main():
    parser = argparse.ArgumentParser(
        description="Migrate GDELT CSV to PostgreSQL news_articles_search"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview migration without writing to database",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of rows per INSERT batch (default: 1000)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # ---------------------------------------------------------------
    # 1. Load CSV
    # ---------------------------------------------------------------
    if not CSV_PATH.exists():
        logger.error("CSV not found at %s", CSV_PATH)
        sys.exit(1)

    logger.info("Loading CSV from %s ...", CSV_PATH)
    df = pd.read_csv(CSV_PATH, low_memory=False)
    logger.info("Loaded %d rows from CSV", len(df))

    # ---------------------------------------------------------------
    # 2. Normalise columns
    # ---------------------------------------------------------------
    # Detect date column
    date_col = None
    for candidate in ("date", "published_at", "seendate"):
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col:
        df["_date"] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    else:
        logger.warning("No date column found; dates will be NULL")
        df["_date"] = pd.NaT

    # Ensure required columns exist with defaults
    if "title" not in df.columns:
        df["title"] = ""
    if "source" not in df.columns:
        df["source"] = df.get("domain", "unknown")
    if "domain" not in df.columns:
        df["domain"] = df.get("source", "unknown")
    if "language" not in df.columns:
        df["language"] = "en"
    if "url" not in df.columns:
        # Generate a synthetic URL from title hash if missing
        df["url"] = df["title"].apply(lambda t: f"https://synthetic/{hashlib.md5(str(t).encode()).hexdigest()}")
    if "tone" not in df.columns:
        df["tone"] = 0.0
    if "category" not in df.columns:
        df["category"] = "general"

    # Compute relevance
    logger.info("Computing relevance scores ...")
    df["_relevance"] = df.apply(
        lambda r: compute_relevance(str(r.get("title", "")), str(r.get("language", "en"))),
        axis=1,
    )

    # Compute url_hash
    df["_url_hash"] = df["url"].apply(lambda u: url_hash(str(u)))

    # Drop duplicates by url_hash (keep first)
    before = len(df)
    df = df.drop_duplicates(subset=["_url_hash"], keep="first")
    dupes = before - len(df)
    if dupes > 0:
        logger.info("Dropped %d duplicate URLs", dupes)

    logger.info("Prepared %d rows for migration", len(df))

    # ---------------------------------------------------------------
    # 3. Dry run summary
    # ---------------------------------------------------------------
    if args.dry_run:
        logger.info("=== DRY RUN ===")
        logger.info("Rows to upsert: %d", len(df))
        logger.info("Date range: %s to %s", df["_date"].min(), df["_date"].max())
        logger.info("Sources: %s", df["source"].nunique())
        logger.info("Languages: %s", df["language"].value_counts().to_dict())
        logger.info("Relevance stats: mean=%.2f, max=%.2f, >0: %d",
                     df["_relevance"].mean(), df["_relevance"].max(),
                     (df["_relevance"] > 0).sum())
        logger.info("Sample rows:")
        for _, row in df.head(5).iterrows():
            logger.info("  [%s] %s (rel=%.1f)", row["source"], row["title"][:80], row["_relevance"])
        logger.info("=== DRY RUN COMPLETE (no data written) ===")
        return

    # ---------------------------------------------------------------
    # 4. Connect to PostgreSQL
    # ---------------------------------------------------------------
    try:
        import psycopg2
    except ImportError:
        logger.error("psycopg2 not installed. Run: pip install psycopg2-binary")
        sys.exit(1)

    host = os.getenv("USDCOP_DB_HOST", "localhost")
    port = int(os.getenv("USDCOP_DB_PORT", "5432"))
    dbname = os.getenv("USDCOP_DB_NAME", "usdcop")
    user = os.getenv("USDCOP_DB_USER", "postgres")
    password = os.getenv("USDCOP_DB_PASSWORD", "")

    logger.info("Connecting to PostgreSQL %s@%s:%s/%s ...", user, host, port, dbname)
    conn = psycopg2.connect(
        host=host, port=port, dbname=dbname, user=user, password=password,
    )
    conn.autocommit = False
    cur = conn.cursor()

    # ---------------------------------------------------------------
    # 5. Create table and indexes
    # ---------------------------------------------------------------
    logger.info("Creating table if not exists ...")
    cur.execute(CREATE_TABLE_SQL)
    for idx_sql in CREATE_INDEXES_SQL:
        cur.execute(idx_sql)
    conn.commit()
    logger.info("Schema ready")

    # ---------------------------------------------------------------
    # 6. Batch upsert
    # ---------------------------------------------------------------
    total_upserted = 0
    batch_size = args.batch_size
    t_start = time.time()

    for batch_start in range(0, len(df), batch_size):
        batch = df.iloc[batch_start:batch_start + batch_size]
        rows_data = []
        for _, row in batch.iterrows():
            date_val = row["_date"]
            if pd.isna(date_val):
                date_val = None
            else:
                date_val = date_val.isoformat()

            tone_val = row.get("tone", 0.0)
            if pd.isna(tone_val):
                tone_val = None

            rows_data.append((
                str(row.get("title", ""))[:10000],  # Cap title length
                str(row.get("source", ""))[:200],
                str(row.get("domain", ""))[:200],
                date_val,
                str(row.get("language", "en"))[:10],
                str(row.get("url", ""))[:5000],
                float(tone_val) if tone_val is not None else None,
                row["_url_hash"],
                str(row.get("category", "general"))[:50],
                float(row["_relevance"]),
            ))

        try:
            cur.executemany(UPSERT_SQL, rows_data)
            conn.commit()
            total_upserted += len(rows_data)

            if (batch_start // batch_size) % 10 == 0:
                elapsed = time.time() - t_start
                rate = total_upserted / elapsed if elapsed > 0 else 0
                logger.info(
                    "Progress: %d/%d rows (%.0f rows/sec)",
                    total_upserted, len(df), rate,
                )
        except Exception as e:
            conn.rollback()
            logger.error("Batch error at offset %d: %s", batch_start, e)
            # Continue with next batch
            continue

    elapsed = time.time() - t_start

    # ---------------------------------------------------------------
    # 7. Verify
    # ---------------------------------------------------------------
    cur.execute("SELECT COUNT(*) FROM news_articles_search")
    final_count = cur.fetchone()[0]

    cur.close()
    conn.close()

    logger.info("=== Migration Complete ===")
    logger.info("Rows processed: %d", total_upserted)
    logger.info("Total rows in table: %d", final_count)
    logger.info("Time elapsed: %.1f seconds", elapsed)
    logger.info(
        "Rate: %.0f rows/sec",
        total_upserted / elapsed if elapsed > 0 else 0,
    )


if __name__ == "__main__":
    main()
