"""
MCP News Server for USDCOP Trading System.

Exposes 7 read-only tools for querying USDCOP-relevant news articles via the
Model Context Protocol (MCP). Supports stdio (Claude Desktop) and HTTP modes.

Usage:
    python src/news_engine/mcp_server.py              # stdio mode (Claude Desktop)
    python src/news_engine/mcp_server.py --http 8080   # HTTP mode on port 8080
    python src/news_engine/mcp_server.py --init-db     # Create PG schema

Part of Phase 4: MCP News Server implementation.
"""

import asyncio
import json
import logging

from mcp.server.fastmcp import FastMCP

from src.news_engine.mcp_data_layer import VALID_CATEGORIES, NewsDataLayer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------

mcp = FastMCP("usdcop-news")
data_layer = NewsDataLayer()


def _format_articles(articles: list, header: str = "") -> str:
    """Format a list of article dicts into readable text output."""
    if not articles:
        return header + "\nNo articles found." if header else "No articles found."

    lines = []
    if header:
        lines.append(header)
        lines.append("")

    for i, art in enumerate(articles, 1):
        title = art.get("title", "No title")
        source = art.get("source") or art.get("domain") or "unknown"
        date = art.get("date", "")
        tone = art.get("tone")
        category = art.get("category", "")
        relevance = art.get("relevance") or art.get("_relevance")
        url = art.get("url", "")

        line = f"{i}. [{source}] {title}"
        meta_parts = []
        if date:
            meta_parts.append(f"date={date}")
        if tone is not None:
            meta_parts.append(f"tone={tone}")
        if category:
            meta_parts.append(f"cat={category}")
        if relevance is not None:
            meta_parts.append(f"rel={relevance}")
        if meta_parts:
            line += f"\n   ({', '.join(meta_parts)})"
        if url:
            line += f"\n   {url}"
        lines.append(line)

    lines.append(f"\n--- {len(articles)} article(s) returned ---")
    return "\n".join(lines)


def _format_stats(stats: dict) -> str:
    """Format source statistics into readable text."""
    lines = [
        f"Total articles: {stats.get('total_articles', 0)}",
        f"Average tone: {stats.get('avg_tone', 'N/A')}",
    ]

    dr = stats.get("date_range", {})
    if dr.get("earliest"):
        lines.append(f"Date range: {dr['earliest']} to {dr.get('latest', '?')}")

    lines.append("")
    lines.append("By Source:")
    for src in stats.get("by_source", []):
        tone_str = f", avg_tone={src['avg_tone']}" if "avg_tone" in src else ""
        lines.append(f"  - {src.get('source', '?')}: {src.get('article_count', 0)} articles{tone_str}")

    lines.append("")
    lines.append("By Category:")
    for cat in stats.get("by_category", []):
        lines.append(f"  - {cat.get('category', '?')}: {cat.get('article_count', 0)} articles")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 1: news_search
# ---------------------------------------------------------------------------

@mcp.tool(
    description=(
        "Search USDCOP-relevant news articles by keyword. Supports date range, "
        "source, language, and category filters. Uses full-text search on PostgreSQL "
        "or keyword matching on CSV fallback."
    ),
)
async def news_search(
    query: str,
    date_from: str | None = None,
    date_to: str | None = None,
    source: str | None = None,
    language: str | None = None,
    category: str | None = None,
    limit: int = 20,
) -> str:
    """Search news articles by keyword with optional filters."""
    await _ensure_connected()
    articles = await data_layer.search(
        query=query,
        date_from=date_from,
        date_to=date_to,
        source=source,
        language=language,
        category=category,
        limit=limit,
    )
    return _format_articles(articles, f"Search results for: '{query}'")


# ---------------------------------------------------------------------------
# Tool 2: news_top_headlines
# ---------------------------------------------------------------------------

@mcp.tool(
    description=(
        "Get top USDCOP-relevant headlines sorted by relevance score. "
        "Optionally diversifies across sources to avoid single-source dominance. "
        "Relevance is computed from 3-tier USDCOP keyword matching."
    ),
)
async def news_top_headlines(
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 10,
    diversify_sources: bool = True,
    min_relevance: float = 0.3,
) -> str:
    """Get top USDCOP-relevant headlines."""
    await _ensure_connected()
    articles = await data_layer.top_headlines(
        date_from=date_from,
        date_to=date_to,
        limit=limit,
        diversify_sources=diversify_sources,
        min_relevance=min_relevance,
    )
    return _format_articles(articles, "Top USDCOP Headlines")


# ---------------------------------------------------------------------------
# Tool 3: news_by_category
# ---------------------------------------------------------------------------

@mcp.tool(
    description=(
        "Filter news articles by financial category. "
        f"Valid categories: {', '.join(VALID_CATEGORIES)}. "
        "Categories come from the 9-class enrichment pipeline."
    ),
)
async def news_by_category(
    category: str,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 20,
) -> str:
    """Get articles filtered by enrichment category."""
    if category not in VALID_CATEGORIES:
        return (
            f"Invalid category: '{category}'. "
            f"Valid categories: {', '.join(VALID_CATEGORIES)}"
        )
    await _ensure_connected()
    articles = await data_layer.by_category(
        category=category,
        date_from=date_from,
        date_to=date_to,
        limit=limit,
    )
    return _format_articles(articles, f"Articles in category: {category}")


# ---------------------------------------------------------------------------
# Tool 4: news_source_stats
# ---------------------------------------------------------------------------

@mcp.tool(
    description=(
        "Get volume, sentiment tone, and category breakdown statistics "
        "across all news sources. Useful for understanding coverage and bias."
    ),
)
async def news_source_stats(
    date_from: str | None = None,
    date_to: str | None = None,
) -> str:
    """Get source-level statistics for articles."""
    await _ensure_connected()
    stats = await data_layer.source_stats(
        date_from=date_from,
        date_to=date_to,
    )
    return _format_stats(stats)


# ---------------------------------------------------------------------------
# Tool 5: news_daily_briefing
# ---------------------------------------------------------------------------

@mcp.tool(
    description=(
        "Get a structured daily news briefing for a specific date, "
        "grouped by financial category. Includes top headlines and "
        "category-level breakdown."
    ),
)
async def news_daily_briefing(
    date: str,
    max_headlines: int = 10,
) -> str:
    """Get a daily news briefing grouped by category."""
    await _ensure_connected()
    briefing = await data_layer.daily_briefing(
        date_str=date,
        max_headlines=max_headlines,
    )

    lines = [
        f"Daily Briefing: {briefing['date']}",
        f"Total articles: {briefing['total_articles']}",
        "",
        "=== Top Headlines ===",
    ]

    for i, art in enumerate(briefing.get("top_headlines", []), 1):
        title = art.get("title", "No title")
        source = art.get("source") or art.get("domain") or "?"
        lines.append(f"  {i}. [{source}] {title}")

    for cat, arts in briefing.get("by_category", {}).items():
        lines.append(f"\n=== {cat.upper()} ({len(arts)} articles) ===")
        for art in arts[:5]:
            title = art.get("title", "No title")
            source = art.get("source") or art.get("domain") or "?"
            lines.append(f"  - [{source}] {title}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 6: news_weekly_summary
# ---------------------------------------------------------------------------

@mcp.tool(
    description=(
        "Get a weekly news summary with per-day breakdown and a "
        "prompt_injection_text field ready for LLM context injection. "
        "Covers the full week with key headlines per day."
    ),
)
async def news_weekly_summary(
    week_start: str,
    week_end: str | None = None,
    headlines_per_day: int = 5,
) -> str:
    """Get a weekly news summary with prompt-ready text."""
    if week_end is None:
        # Default to 6 days after start (Mon-Sun)
        from datetime import datetime, timedelta

        try:
            start_dt = datetime.fromisoformat(week_start)
        except ValueError:
            start_dt = datetime.strptime(week_start, "%Y-%m-%d")
        week_end = (start_dt + timedelta(days=6)).strftime("%Y-%m-%d")

    await _ensure_connected()
    summary = await data_layer.weekly_summary(
        week_start=week_start,
        week_end=week_end,
        headlines_per_day=headlines_per_day,
    )

    # Return the prompt injection text as the primary output
    output_lines = [
        summary.get("prompt_injection_text", ""),
        "",
        "---",
        f"Week: {summary['week_start']} to {summary['week_end']}",
        f"Total headlines collected: {summary.get('total_headlines', 0)}",
        f"Days covered: {len(summary.get('daily_summaries', []))}",
    ]

    return "\n".join(output_lines)


# ---------------------------------------------------------------------------
# Tool 7: db_query_raw
# ---------------------------------------------------------------------------

@mcp.tool(
    description=(
        "Execute a raw read-only SQL query against the PostgreSQL news database. "
        "Only SELECT statements are allowed (max 2000 chars). "
        "Not available when running in CSV fallback mode."
    ),
)
async def db_query_raw(sql: str) -> str:
    """Execute a raw read-only SQL query."""
    if len(sql) > 2000:
        return f"Query too long ({len(sql)} chars). Maximum is 2000 characters."

    await _ensure_connected()
    try:
        rows = await data_layer.raw_query(sql)
    except RuntimeError as e:
        return str(e)
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Query error: {e}"

    if not rows:
        return "Query returned 0 rows."

    # Format as a simple table
    lines = [f"Query returned {len(rows)} row(s):", ""]
    for i, row in enumerate(rows[:50]):  # Cap display at 50 rows
        lines.append(f"Row {i + 1}: {json.dumps({k: _safe_val(v) for k, v in row.items()}, default=str)}")

    if len(rows) > 50:
        lines.append(f"... ({len(rows) - 50} more rows not shown)")

    return "\n".join(lines)


def _safe_val(v):
    """Make a value JSON-serialisable."""
    if v is None:
        return None
    if isinstance(v, (int, float, str, bool)):
        return v
    return str(v)


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

_connected = False


async def _ensure_connected():
    """Lazy-connect the data layer on first tool call."""
    global _connected
    if not _connected:
        backend = await data_layer.connect()
        logger.info("MCP News Server connected to: %s", backend)
        _connected = True


# ---------------------------------------------------------------------------
# Schema initialisation (--init-db)
# ---------------------------------------------------------------------------

async def init_db_schema():
    """Create the news_articles_search table and indexes in PostgreSQL."""
    import os

    import asyncpg

    host = os.getenv("USDCOP_DB_HOST", "localhost")
    port = int(os.getenv("USDCOP_DB_PORT", "5432"))
    dbname = os.getenv("USDCOP_DB_NAME", "usdcop")
    user = os.getenv("USDCOP_DB_USER", "postgres")
    password = os.getenv("USDCOP_DB_PASSWORD", "")

    conn = await asyncpg.connect(
        host=host, port=port, database=dbname, user=user, password=password,
    )

    try:
        await conn.execute("""
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
        """)
        print("Created table: news_articles_search")

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_news_search_title_gin
            ON news_articles_search
            USING GIN (to_tsvector('simple', title));
        """)
        print("Created GIN index on title")

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_news_search_date
            ON news_articles_search (date DESC);
        """)
        print("Created index on date")

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_news_search_category
            ON news_articles_search (category);
        """)
        print("Created index on category")

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_news_search_relevance
            ON news_articles_search (relevance DESC);
        """)
        print("Created index on relevance")

        count = await conn.fetchval("SELECT COUNT(*) FROM news_articles_search")
        print(f"Table has {count} rows. Schema ready.")

    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="USDCOP News MCP Server")
    parser.add_argument(
        "--http",
        type=int,
        default=None,
        metavar="PORT",
        help="Run in HTTP mode on the specified port",
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Create PostgreSQL schema and exit",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.init_db:
        asyncio.run(init_db_schema())
        return

    if args.http is not None:
        # HTTP/SSE transport
        mcp.run(transport="sse", port=args.http)
    else:
        # Default: stdio transport (Claude Desktop)
        mcp.run()


if __name__ == "__main__":
    main()
