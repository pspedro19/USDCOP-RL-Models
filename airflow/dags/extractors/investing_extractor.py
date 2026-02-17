# -*- coding: utf-8 -*-
"""
Investing.com Extractor - REST API (no Selenium).

Extracts:
- DXY (Dollar Index)
- VIX (Volatility Index)
- UST10Y (US 10Y Treasury)
- WTI, BRENT (Oil)
- GOLD, COFFEE (Commodities)
- USDMXN, USDCLP (FX pairs)
- COLCAP (Colombian equity index)
- COL10Y, COL5Y (Colombian bonds)

API discovered via browser inspection:
    GET https://api.investing.com/api/financialdata/historical/{instrument_id}
    Params: start-date, end-date, time-frame=Daily

Uses cloudscraper for Cloudflare bypass.
"""

import logging
import time
import random
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

import pandas as pd

from .base import BaseExtractor, ExtractionResult

logger = logging.getLogger(__name__)

try:
    import cloudscraper
    HAS_CLOUDSCRAPER = True
except ImportError:
    HAS_CLOUDSCRAPER = False
    logger.warning("cloudscraper not installed - Investing extractor will fail")


class InvestingExtractor(BaseExtractor):
    """Investing.com REST API Extractor."""

    API_BASE = "https://api.investing.com/api/financialdata/historical"
    MAX_CHUNK_DAYS = 365

    # Domain mapping for referer headers
    DOMAIN_MAP = {
        'www': 'https://www.investing.com',
        'es': 'https://es.investing.com',
    }

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._variables_config = {
            v['name']: v for v in config.get('variables', [])
        }
        self._session = None
        self._rate_limit_delay = config.get('rate_limit_seconds', 3)

    @property
    def source_name(self) -> str:
        return "investing"

    @property
    def variables(self) -> List[str]:
        return list(self._variables_config.keys())

    def _get_session(self):
        """Create or return cached cloudscraper session."""
        if not HAS_CLOUDSCRAPER:
            raise ImportError("cloudscraper required: pip install cloudscraper")

        if self._session is None:
            self._session = cloudscraper.create_scraper(
                browser={
                    "browser": "chrome",
                    "platform": "windows",
                    "desktop": True,
                }
            )
            self._session.headers.update({
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json",
                "Accept-Language": "en-US,en;q=0.9,es;q=0.8",
            })
        return self._session

    def _get_variable_config(self, variable: str) -> dict:
        """Get config for a variable."""
        return self._variables_config.get(variable, {})

    def extract(
        self,
        variable: str,
        start_date: datetime,
        end_date: datetime,
        last_n: Optional[int] = None
    ) -> ExtractionResult:
        """Extract OHLCV data from Investing.com API."""

        cfg = self._get_variable_config(variable)
        instrument_id = cfg.get('instrument_id')

        if not instrument_id:
            return ExtractionResult(
                source=self.source_name,
                variable=variable,
                data=pd.DataFrame(),
                success=False,
                error=f"No instrument_id configured for {variable}"
            )

        try:
            session = self._get_session()
            df = self._fetch_chunked(
                session,
                instrument_id,
                cfg.get('domain_id', 'www'),
                cfg.get('referer_url', ''),
                start_date,
                end_date
            )

            if df.empty:
                return ExtractionResult(
                    source=self.source_name,
                    variable=variable,
                    data=df,
                    success=False,
                    error=f"No data returned for instrument {instrument_id}"
                )

            # Rename close column to variable name
            df = df.rename(columns={'close': variable})

            if last_n and len(df) > last_n:
                df = df.tail(last_n)

            return ExtractionResult(
                source=self.source_name,
                variable=variable,
                data=df,
                success=True,
                metadata={'instrument_id': instrument_id}
            )

        except Exception as e:
            logger.error("[Investing] Failed to extract %s: %s", variable, e)
            return ExtractionResult(
                source=self.source_name,
                variable=variable,
                data=pd.DataFrame(),
                success=False,
                error=str(e)
            )

    def _fetch_chunked(
        self,
        session,
        instrument_id: int,
        domain_id: str,
        referer_url: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch data in annual chunks to handle API limits."""

        chunks = self._split_date_range(start_date, end_date)
        all_rows = []

        for i, (chunk_start, chunk_end) in enumerate(chunks):
            logger.debug(
                "  Chunk %d/%d: %s -> %s",
                i + 1, len(chunks),
                chunk_start.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d")
            )

            rows = self._with_retry(
                self._fetch_chunk,
                session,
                instrument_id,
                domain_id,
                referer_url,
                chunk_start.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d")
            )

            if rows:
                all_rows.extend(rows)

            # Inter-chunk delay
            if i < len(chunks) - 1:
                time.sleep(self._rate_limit_delay + random.uniform(0, 2))

        return self._parse_rows(all_rows)

    def _split_date_range(self, start: datetime, end: datetime) -> List[tuple]:
        """Split date range into annual chunks."""
        chunks = []
        chunk_start = start
        while chunk_start <= end:
            chunk_end = min(
                chunk_start + timedelta(days=self.MAX_CHUNK_DAYS - 1),
                end
            )
            chunks.append((chunk_start, chunk_end))
            chunk_start = chunk_end + timedelta(days=1)
        return chunks

    def _fetch_chunk(
        self,
        session,
        instrument_id: int,
        domain_id: str,
        referer_url: str,
        start_date: str,
        end_date: str
    ) -> List[dict]:
        """Fetch a single chunk from API."""

        url = f"{self.API_BASE}/{instrument_id}"
        params = {
            "start-date": start_date,
            "end-date": end_date,
            "time-frame": "Daily",
            "add-missing-rows": "false",
        }
        headers = {
            "Accept": "application/json",
            "Referer": referer_url or f"{self.DOMAIN_MAP.get(domain_id, self.DOMAIN_MAP['www'])}/",
            "domain-id": domain_id,
        }

        resp = session.get(url, params=params, headers=headers, timeout=self.timeout)
        resp.raise_for_status()

        data = resp.json()
        return data.get("data", [])

    def _parse_rows(self, rows: List[dict]) -> pd.DataFrame:
        """Parse API JSON rows into DataFrame."""
        if not rows:
            return pd.DataFrame()

        records = []
        for item in rows:
            date_str = item.get("rowDateTimestamp", item.get("rowDate", ""))
            if "T" in date_str:
                date_part = date_str.split("T")[0]
            else:
                continue

            try:
                dt = datetime.strptime(date_part, "%Y-%m-%d")
            except ValueError:
                continue

            def _safe_float(val):
                if val is None or val == "" or val == "-":
                    return None
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return None

            records.append({
                "fecha": dt,
                "close": _safe_float(item.get("last_closeRaw", item.get("last_close"))),
                "open": _safe_float(item.get("last_openRaw", item.get("last_open"))),
                "high": _safe_float(item.get("last_maxRaw", item.get("last_max"))),
                "low": _safe_float(item.get("last_minRaw", item.get("last_min"))),
                "volume": _safe_float(item.get("volumeRaw", item.get("volume"))),
                "change_pct": _safe_float(item.get("change_precent", item.get("change_percent"))),
            })

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.drop_duplicates(subset=["fecha"])
            df = df.sort_values("fecha").reset_index(drop=True)

        return df

    def get_latest_date(self, variable: str) -> Optional[datetime]:
        """Get latest available date for variable."""
        cfg = self._get_variable_config(variable)
        instrument_id = cfg.get('instrument_id')

        if not instrument_id:
            return None

        try:
            session = self._get_session()
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)

            rows = self._fetch_chunk(
                session,
                instrument_id,
                cfg.get('domain_id', 'www'),
                cfg.get('referer_url', ''),
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d")
            )

            df = self._parse_rows(rows)
            if not df.empty:
                return df['fecha'].max()

        except Exception as e:
            logger.warning("[Investing] Failed to get latest date for %s: %s", variable, e)

        return None
