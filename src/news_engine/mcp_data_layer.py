"""
MCP News Data Layer - Dual PostgreSQL + CSV backend for USDCOP news intelligence.

Provides structured access to news articles with full-text search, relevance scoring,
and aggregation capabilities. Falls back to CSV when PostgreSQL is unavailable.

Part of Phase 4: MCP News Server implementation.
"""

import hashlib
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# USDCOP relevance keywords (3-tier weighting)
# ---------------------------------------------------------------------------

RELEVANCE_KEYWORDS: Dict[float, List[str]] = {
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

SPANISH_BOOST = 0.5

VALID_CATEGORIES = [
    "monetary_policy", "fx_market", "commodities", "inflation",
    "fiscal_policy", "risk_premium", "capital_flows", "political", "general",
]


def _compute_relevance(title: str, language: str = "en") -> float:
    """Score a headline against USDCOP relevance keywords."""
    if not title:
        return 0.0
    title_lower = title.lower()
    score = 0.0
    for weight, keywords in RELEVANCE_KEYWORDS.items():
        for kw in keywords:
            if kw in title_lower:
                score += weight
    if language and language.lower().startswith("es"):
        score += SPANISH_BOOST
    return round(score, 2)


class NewsDataLayer:
    """Dual-backend data access for USDCOP news articles.

    Tries PostgreSQL first (asyncpg); falls back to loading the GDELT
    historical CSV via pandas when the database is unavailable.
    """

    def __init__(self) -> None:
        self._pg_pool = None
        self._csv_df = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def connect(self) -> str:
        """Establish a connection. Returns a string describing the backend."""
        # Attempt PostgreSQL
        try:
            import asyncpg  # noqa: F811

            host = os.getenv("USDCOP_DB_HOST", "localhost")
            port = int(os.getenv("USDCOP_DB_PORT", "5432"))
            dbname = os.getenv("USDCOP_DB_NAME", "usdcop")
            user = os.getenv("USDCOP_DB_USER", "postgres")
            password = os.getenv("USDCOP_DB_PASSWORD", "")

            self._pg_pool = await asyncpg.create_pool(
                host=host,
                port=port,
                database=dbname,
                user=user,
                password=password,
                min_size=1,
                max_size=5,
                command_timeout=30,
            )
            # Quick health check
            async with self._pg_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            logger.info("Connected to PostgreSQL at %s:%s/%s", host, port, dbname)
            return f"postgresql://{host}:{port}/{dbname}"

        except Exception as exc:
            logger.warning("PostgreSQL unavailable (%s), falling back to CSV", exc)
            self._pg_pool = None

        # Fall back to CSV
        csv_path = PROJECT_ROOT / "data" / "news" / "gdelt_articles_historical.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Neither PostgreSQL nor CSV fallback available. "
                f"Expected CSV at {csv_path}"
            )

        import pandas as pd

        self._csv_df = pd.read_csv(csv_path, low_memory=False)
        # Normalise date column
        for col in ("date", "published_at", "seendate"):
            if col in self._csv_df.columns:
                self._csv_df["_date"] = pd.to_datetime(
                    self._csv_df[col], errors="coerce", utc=True
                )
                break
        else:
            self._csv_df["_date"] = pd.NaT

        # Ensure title column exists
        if "title" not in self._csv_df.columns:
            self._csv_df["title"] = ""

        logger.info(
            "Loaded CSV fallback with %d rows from %s",
            len(self._csv_df),
            csv_path,
        )
        return f"csv://{csv_path}"

    @property
    def is_pg(self) -> bool:
        return self._pg_pool is not None

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        source: Optional[str] = None,
        language: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Full-text search across articles."""
        limit = min(limit, 100)

        if self.is_pg:
            return await self._pg_search(
                query, date_from, date_to, source, language, category, limit
            )
        return self._csv_search(
            query, date_from, date_to, source, language, category, limit
        )

    async def _pg_search(
        self,
        query: str,
        date_from: Optional[str],
        date_to: Optional[str],
        source: Optional[str],
        language: Optional[str],
        category: Optional[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        conditions = ["to_tsvector('simple', title) @@ plainto_tsquery('simple', $1)"]
        params: list = [query]
        idx = 2

        if date_from:
            conditions.append(f"date >= ${idx}::timestamptz")
            params.append(date_from)
            idx += 1
        if date_to:
            conditions.append(f"date <= ${idx}::timestamptz")
            params.append(date_to)
            idx += 1
        if source:
            conditions.append(f"source ILIKE ${idx}")
            params.append(f"%{source}%")
            idx += 1
        if language:
            conditions.append(f"language = ${idx}")
            params.append(language)
            idx += 1
        if category:
            conditions.append(f"category = ${idx}")
            params.append(category)
            idx += 1

        where = " AND ".join(conditions)
        sql = f"""
            SELECT title, source, domain, date, language, url, tone, category, relevance,
                   ts_rank(to_tsvector('simple', title), plainto_tsquery('simple', $1)) AS rank
            FROM news_articles_search
            WHERE {where}
            ORDER BY rank DESC, date DESC
            LIMIT {limit}
        """

        async with self._pg_pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        return [dict(r) for r in rows]

    def _csv_search(
        self,
        query: str,
        date_from: Optional[str],
        date_to: Optional[str],
        source: Optional[str],
        language: Optional[str],
        category: Optional[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        import pandas as pd

        df = self._csv_df.copy()

        # Keyword match on title
        if query:
            mask = df["title"].fillna("").str.contains(query, case=False, na=False)
            df = df[mask]

        # Date filters
        if date_from:
            df = df[df["_date"] >= pd.Timestamp(date_from, tz="UTC")]
        if date_to:
            df = df[df["_date"] <= pd.Timestamp(date_to, tz="UTC")]

        # Source filter
        if source:
            src_col = "source" if "source" in df.columns else "domain"
            if src_col in df.columns:
                df = df[df[src_col].fillna("").str.contains(source, case=False, na=False)]

        # Language filter
        if language and "language" in df.columns:
            df = df[df["language"].fillna("").str.lower() == language.lower()]

        # Category filter
        if category and "category" in df.columns:
            df = df[df["category"].fillna("").str.lower() == category.lower()]

        # Sort by date descending
        df = df.sort_values("_date", ascending=False).head(limit)
        return self._df_to_dicts(df)

    # ------------------------------------------------------------------
    # Top headlines
    # ------------------------------------------------------------------

    async def top_headlines(
        self,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 10,
        diversify_sources: bool = True,
        min_relevance: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Top USDCOP-relevant articles with optional source diversity."""
        limit = min(limit, 50)

        if self.is_pg:
            return await self._pg_top_headlines(
                date_from, date_to, limit, diversify_sources, min_relevance
            )
        return self._csv_top_headlines(
            date_from, date_to, limit, diversify_sources, min_relevance
        )

    async def _pg_top_headlines(
        self,
        date_from: Optional[str],
        date_to: Optional[str],
        limit: int,
        diversify_sources: bool,
        min_relevance: float,
    ) -> List[Dict[str, Any]]:
        conditions = [f"relevance >= {min_relevance}"]
        params: list = []
        idx = 1

        if date_from:
            conditions.append(f"date >= ${idx}::timestamptz")
            params.append(date_from)
            idx += 1
        if date_to:
            conditions.append(f"date <= ${idx}::timestamptz")
            params.append(date_to)
            idx += 1

        where = " AND ".join(conditions)

        if diversify_sources:
            sql = f"""
                WITH ranked AS (
                    SELECT *, ROW_NUMBER() OVER (PARTITION BY source ORDER BY relevance DESC, date DESC) AS rn
                    FROM news_articles_search
                    WHERE {where}
                )
                SELECT title, source, domain, date, language, url, tone, category, relevance
                FROM ranked
                WHERE rn <= 3
                ORDER BY relevance DESC, date DESC
                LIMIT {limit}
            """
        else:
            sql = f"""
                SELECT title, source, domain, date, language, url, tone, category, relevance
                FROM news_articles_search
                WHERE {where}
                ORDER BY relevance DESC, date DESC
                LIMIT {limit}
            """

        async with self._pg_pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        return [dict(r) for r in rows]

    def _csv_top_headlines(
        self,
        date_from: Optional[str],
        date_to: Optional[str],
        limit: int,
        diversify_sources: bool,
        min_relevance: float,
    ) -> List[Dict[str, Any]]:
        import pandas as pd

        df = self._csv_df.copy()

        if date_from:
            df = df[df["_date"] >= pd.Timestamp(date_from, tz="UTC")]
        if date_to:
            df = df[df["_date"] <= pd.Timestamp(date_to, tz="UTC")]

        # Compute relevance
        lang_col = "language" if "language" in df.columns else None
        df["_relevance"] = df.apply(
            lambda r: _compute_relevance(
                str(r.get("title", "")),
                str(r[lang_col]) if lang_col else "en",
            ),
            axis=1,
        )
        df = df[df["_relevance"] >= min_relevance]

        if diversify_sources:
            src_col = "source" if "source" in df.columns else "domain"
            if src_col in df.columns:
                df = (
                    df.sort_values("_relevance", ascending=False)
                    .groupby(src_col)
                    .head(3)
                )

        df = df.sort_values("_relevance", ascending=False).head(limit)
        return self._df_to_dicts(df)

    # ------------------------------------------------------------------
    # By category
    # ------------------------------------------------------------------

    async def by_category(
        self,
        category: str,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Filter articles by enrichment category."""
        limit = min(limit, 100)

        if self.is_pg:
            conditions = ["category = $1"]
            params: list = [category]
            idx = 2
            if date_from:
                conditions.append(f"date >= ${idx}::timestamptz")
                params.append(date_from)
                idx += 1
            if date_to:
                conditions.append(f"date <= ${idx}::timestamptz")
                params.append(date_to)
                idx += 1

            where = " AND ".join(conditions)
            sql = f"""
                SELECT title, source, domain, date, language, url, tone, category, relevance
                FROM news_articles_search
                WHERE {where}
                ORDER BY date DESC
                LIMIT {limit}
            """
            async with self._pg_pool.acquire() as conn:
                rows = await conn.fetch(sql, *params)
            return [dict(r) for r in rows]

        # CSV fallback
        import pandas as pd

        df = self._csv_df.copy()
        if "category" in df.columns:
            df = df[df["category"].fillna("").str.lower() == category.lower()]
        if date_from:
            df = df[df["_date"] >= pd.Timestamp(date_from, tz="UTC")]
        if date_to:
            df = df[df["_date"] <= pd.Timestamp(date_to, tz="UTC")]
        df = df.sort_values("_date", ascending=False).head(limit)
        return self._df_to_dicts(df)

    # ------------------------------------------------------------------
    # Source stats
    # ------------------------------------------------------------------

    async def source_stats(
        self,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Volume, sentiment, and category breakdown by source."""
        if self.is_pg:
            return await self._pg_source_stats(date_from, date_to)
        return self._csv_source_stats(date_from, date_to)

    async def _pg_source_stats(
        self,
        date_from: Optional[str],
        date_to: Optional[str],
    ) -> Dict[str, Any]:
        conditions: list = []
        params: list = []
        idx = 1

        if date_from:
            conditions.append(f"date >= ${idx}::timestamptz")
            params.append(date_from)
            idx += 1
        if date_to:
            conditions.append(f"date <= ${idx}::timestamptz")
            params.append(date_to)
            idx += 1

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        sql_volume = f"""
            SELECT source, COUNT(*) AS article_count,
                   ROUND(AVG(tone)::numeric, 3) AS avg_tone
            FROM news_articles_search
            {where}
            GROUP BY source
            ORDER BY article_count DESC
        """
        sql_category = f"""
            SELECT category, COUNT(*) AS article_count
            FROM news_articles_search
            {where}
            GROUP BY category
            ORDER BY article_count DESC
        """
        sql_total = f"""
            SELECT COUNT(*) AS total,
                   ROUND(AVG(tone)::numeric, 3) AS avg_tone,
                   MIN(date) AS earliest,
                   MAX(date) AS latest
            FROM news_articles_search
            {where}
        """

        async with self._pg_pool.acquire() as conn:
            vol_rows = await conn.fetch(sql_volume, *params)
            cat_rows = await conn.fetch(sql_category, *params)
            total_row = await conn.fetchrow(sql_total, *params)

        return {
            "total_articles": total_row["total"] if total_row else 0,
            "avg_tone": float(total_row["avg_tone"]) if total_row and total_row["avg_tone"] else None,
            "date_range": {
                "earliest": str(total_row["earliest"]) if total_row and total_row["earliest"] else None,
                "latest": str(total_row["latest"]) if total_row and total_row["latest"] else None,
            },
            "by_source": [dict(r) for r in vol_rows],
            "by_category": [dict(r) for r in cat_rows],
        }

    def _csv_source_stats(
        self,
        date_from: Optional[str],
        date_to: Optional[str],
    ) -> Dict[str, Any]:
        import pandas as pd

        df = self._csv_df.copy()
        if date_from:
            df = df[df["_date"] >= pd.Timestamp(date_from, tz="UTC")]
        if date_to:
            df = df[df["_date"] <= pd.Timestamp(date_to, tz="UTC")]

        src_col = "source" if "source" in df.columns else "domain"
        tone_col = "tone" if "tone" in df.columns else None

        by_source = []
        if src_col in df.columns:
            grouped = df.groupby(src_col)
            for name, group in grouped:
                entry: Dict[str, Any] = {
                    "source": name,
                    "article_count": len(group),
                }
                if tone_col and tone_col in group.columns:
                    entry["avg_tone"] = round(group[tone_col].mean(), 3)
                by_source.append(entry)
            by_source.sort(key=lambda x: x["article_count"], reverse=True)

        by_category = []
        if "category" in df.columns:
            cat_grouped = df.groupby("category")
            for name, group in cat_grouped:
                by_category.append({
                    "category": name,
                    "article_count": len(group),
                })
            by_category.sort(key=lambda x: x["article_count"], reverse=True)

        avg_tone = None
        if tone_col and tone_col in df.columns:
            avg_tone = round(df[tone_col].mean(), 3)

        return {
            "total_articles": len(df),
            "avg_tone": avg_tone,
            "date_range": {
                "earliest": str(df["_date"].min()) if not df.empty else None,
                "latest": str(df["_date"].max()) if not df.empty else None,
            },
            "by_source": by_source,
            "by_category": by_category,
        }

    # ------------------------------------------------------------------
    # Daily briefing
    # ------------------------------------------------------------------

    async def daily_briefing(
        self,
        date_str: str,
        max_headlines: int = 10,
    ) -> Dict[str, Any]:
        """Day-level summary grouped by category."""
        date_from = date_str
        # End of day
        try:
            dt = datetime.fromisoformat(date_str)
            date_to = (dt + timedelta(days=1)).isoformat()
        except ValueError:
            date_to = date_str

        articles = await self.top_headlines(
            date_from=date_from,
            date_to=date_to,
            limit=max_headlines * 3,
            diversify_sources=True,
            min_relevance=0.0,
        )

        # Group by category
        categories: Dict[str, List[Dict[str, Any]]] = {}
        for art in articles:
            cat = art.get("category", "general") or "general"
            categories.setdefault(cat, []).append(art)

        # Trim per category
        for cat in categories:
            categories[cat] = categories[cat][:max_headlines]

        # Top headlines across all categories
        top = sorted(
            articles,
            key=lambda a: (a.get("relevance", 0) or 0, a.get("_relevance", 0) or 0),
            reverse=True,
        )[:max_headlines]

        return {
            "date": date_str,
            "total_articles": len(articles),
            "top_headlines": top,
            "by_category": categories,
        }

    # ------------------------------------------------------------------
    # Weekly summary
    # ------------------------------------------------------------------

    async def weekly_summary(
        self,
        week_start: str,
        week_end: str,
        headlines_per_day: int = 5,
    ) -> Dict[str, Any]:
        """Week summary with per-day breakdown and prompt injection text."""
        try:
            start_dt = datetime.fromisoformat(week_start)
        except ValueError:
            start_dt = datetime.strptime(week_start, "%Y-%m-%d")

        try:
            end_dt = datetime.fromisoformat(week_end)
        except ValueError:
            end_dt = datetime.strptime(week_end, "%Y-%m-%d")

        daily_summaries: List[Dict[str, Any]] = []
        all_headlines: List[str] = []

        current = start_dt
        while current <= end_dt:
            day_str = current.strftime("%Y-%m-%d")
            briefing = await self.daily_briefing(day_str, max_headlines=headlines_per_day)
            daily_summaries.append(briefing)

            for art in briefing.get("top_headlines", []):
                title = art.get("title", "")
                if title:
                    all_headlines.append(f"- [{day_str}] {title}")

            current += timedelta(days=1)

        # Build prompt injection text
        prompt_lines = [
            f"## News Summary: {week_start} to {week_end}",
            f"Total days covered: {len(daily_summaries)}",
            "",
            "### Key Headlines:",
        ]
        prompt_lines.extend(all_headlines[:30])  # Cap at 30 headlines

        stats = await self.source_stats(date_from=week_start, date_to=week_end)
        if stats.get("total_articles"):
            prompt_lines.append("")
            prompt_lines.append(f"Total articles in period: {stats['total_articles']}")
            if stats.get("avg_tone") is not None:
                prompt_lines.append(f"Average sentiment tone: {stats['avg_tone']}")

        return {
            "week_start": week_start,
            "week_end": week_end,
            "daily_summaries": daily_summaries,
            "total_headlines": len(all_headlines),
            "source_stats": stats,
            "prompt_injection_text": "\n".join(prompt_lines),
        }

    # ------------------------------------------------------------------
    # Raw query (PG only)
    # ------------------------------------------------------------------

    async def raw_query(self, sql: str) -> List[Dict[str, Any]]:
        """Execute a read-only SQL query against PostgreSQL."""
        if not self.is_pg:
            raise RuntimeError(
                "Raw SQL queries require a PostgreSQL connection. "
                "CSV fallback does not support raw SQL."
            )

        # Validate read-only
        forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE"]
        sql_upper = sql.upper().strip()
        for keyword in forbidden:
            if keyword in sql_upper:
                raise ValueError(
                    f"Write operation detected ({keyword}). "
                    f"Only read-only queries (SELECT) are allowed."
                )

        async with self._pg_pool.acquire() as conn:
            rows = await conn.fetch(sql)
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _df_to_dicts(self, df) -> List[Dict[str, Any]]:
        """Convert a DataFrame slice to a list of serialisable dicts."""
        results = []
        for _, row in df.iterrows():
            entry: Dict[str, Any] = {}
            for col in df.columns:
                if col.startswith("_"):
                    # Internal columns - convert _date to string
                    if col == "_date":
                        entry["date"] = str(row[col]) if not _is_nat(row[col]) else None
                    elif col == "_relevance":
                        entry["relevance"] = row[col]
                    continue
                val = row[col]
                if _is_nat(val):
                    entry[col] = None
                elif hasattr(val, "isoformat"):
                    entry[col] = val.isoformat()
                else:
                    entry[col] = val
            results.append(entry)
        return results

    async def close(self) -> None:
        """Close the underlying connection pool."""
        if self._pg_pool is not None:
            await self._pg_pool.close()
            self._pg_pool = None
        self._csv_df = None


def _is_nat(val) -> bool:
    """Check if a value is NaT or NaN."""
    import math

    if val is None:
        return True
    try:
        import pandas as pd

        if pd.isna(val):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(val, float) and math.isnan(val):
        return True
    return False
