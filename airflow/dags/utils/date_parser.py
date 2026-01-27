"""
Unified Date Parser for L0 Macro Data Acquisition
==================================================

Single Source of Truth for date parsing across all macro extractors.
Handles all date formats encountered in the pipeline.

Contract: CTR-L0-DATEPARSE-001

Supported Formats:
    - ISO format: 2024-01-15
    - Slash format: 2024/01/15
    - European format: 15/01/2024
    - US format: 01/15/2024
    - Compact format: 20240115
    - EMBI format: 06Ene26 (Spanish months)
    - Investing.com English: "Jan 22, 2026", "January 22, 2026"
    - Investing.com Spanish: "22 ene. 2026", "22 enero 2026"
    - pandas Timestamp
    - datetime/date objects

Version: 1.1.0
"""

from __future__ import annotations

import re
from datetime import datetime, date
from typing import Any, Optional, Union

import pandas as pd


class DateParser:
    """
    Single Source of Truth for date parsing across all macro extractors.
    Handles all date formats encountered in the pipeline.
    """

    # Spanish month names for EMBI/BanRep/Investing.com parsing
    SPANISH_MONTHS = {
        'ene': '01', 'enero': '01',
        'feb': '02', 'febrero': '02',
        'mar': '03', 'marzo': '03',
        'abr': '04', 'abril': '04',
        'may': '05', 'mayo': '05',
        'jun': '06', 'junio': '06',
        'jul': '07', 'julio': '07',
        'ago': '08', 'agosto': '08',
        'sep': '09', 'set': '09', 'sept': '09', 'septiembre': '09',
        'oct': '10', 'octubre': '10',
        'nov': '11', 'noviembre': '11',
        'dic': '12', 'diciembre': '12',
    }

    # English month names for Investing.com parsing
    ENGLISH_MONTHS = {
        'jan': '01', 'january': '01',
        'feb': '02', 'february': '02',
        'mar': '03', 'march': '03',
        'apr': '04', 'april': '04',
        'may': '05',
        'jun': '06', 'june': '06',
        'jul': '07', 'july': '07',
        'aug': '08', 'august': '08',
        'sep': '09', 'sept': '09', 'september': '09',
        'oct': '10', 'october': '10',
        'nov': '11', 'november': '11',
        'dec': '12', 'december': '12',
    }

    # Common date formats to try (in order of priority)
    DATE_FORMATS = [
        '%Y-%m-%d',      # ISO format (most common)
        '%Y/%m/%d',      # Slash format
        '%d/%m/%Y',      # European format
        '%m/%d/%Y',      # US format
        '%Y%m%d',        # Compact format
        '%d-%m-%Y',      # Dash European
        '%d.%m.%Y',      # Dot European
        '%Y.%m.%d',      # Dot ISO
    ]

    @classmethod
    def parse(cls, value: Any) -> Optional[str]:
        """
        Parse any date value to ISO format string (YYYY-MM-DD).

        Args:
            value: Can be datetime, date, pd.Timestamp, str, or any parseable format

        Returns:
            ISO date string (YYYY-MM-DD) or None if unparseable
        """
        if value is None:
            return None

        # Already a date object (not datetime)
        if isinstance(value, date) and not isinstance(value, datetime):
            return value.strftime('%Y-%m-%d')

        # datetime object
        if isinstance(value, datetime):
            return value.strftime('%Y-%m-%d')

        # pandas Timestamp
        if isinstance(value, pd.Timestamp):
            return value.strftime('%Y-%m-%d')

        # String - try various formats
        if isinstance(value, str):
            value = value.strip()

            # Empty string
            if not value:
                return None

            # Try EMBI format first (e.g., "06Ene26" or "17Nov25")
            embi_result = cls._parse_embi_format(value)
            if embi_result:
                return embi_result

            # Try Investing.com formats
            investing_result = cls._parse_investing_format(value)
            if investing_result:
                return investing_result

            # Try standard formats
            for fmt in cls.DATE_FORMATS:
                try:
                    return datetime.strptime(value, fmt).strftime('%Y-%m-%d')
                except ValueError:
                    continue

            # Try pandas as last resort (handles many formats)
            try:
                parsed = pd.to_datetime(value, errors='coerce')
                if pd.notna(parsed):
                    return parsed.strftime('%Y-%m-%d')
            except Exception:
                pass

        return None

    @classmethod
    def _parse_embi_format(cls, value: str) -> Optional[str]:
        """
        Parse EMBI date format like '06Ene26' or '17Nov25'.

        Args:
            value: Date string in EMBI format (DDMmmYY)

        Returns:
            ISO date string or None
        """
        match = re.match(r'(\d{2})([A-Za-z]{3})(\d{2})', value)
        if match:
            day, month_txt, year_2d = match.groups()
            month = cls.SPANISH_MONTHS.get(month_txt.lower()[:3])
            if month:
                # 2-digit year conversion (00-49 = 2000s, 50-99 = 1900s)
                year_int = int(year_2d)
                year = f"20{year_2d}" if year_int <= 49 else f"19{year_2d}"
                return f"{year}-{month}-{day}"
        return None

    @classmethod
    def _parse_investing_format(cls, value: str) -> Optional[str]:
        """
        Parse Investing.com date formats.

        Supported formats:
            - English: "Jan 22, 2026", "January 22, 2026"
            - Spanish: "22 ene. 2026", "22 enero 2026"

        Args:
            value: Date string from Investing.com

        Returns:
            ISO date string (YYYY-MM-DD) or None
        """
        if not value:
            return None

        # Clean up the string
        value = value.strip().lower()
        value = value.replace('.', '').strip()

        # Try English format: "jan 22, 2026" or "january 22, 2026"
        try:
            parts = value.replace(',', '').split()
            if len(parts) == 3:
                month_str = parts[0]
                day = int(parts[1])
                year = int(parts[2])

                # Check English months first
                month = cls.ENGLISH_MONTHS.get(month_str[:3])
                if month:
                    return f"{year:04d}-{month}-{day:02d}"
        except (ValueError, IndexError):
            pass

        # Try Spanish format: "22 ene 2026" or "22 enero 2026"
        try:
            parts = value.split()
            if len(parts) == 3:
                day = int(parts[0])
                month_str = parts[1][:3]  # First 3 chars
                year = int(parts[2])

                month = cls.SPANISH_MONTHS.get(month_str)
                if month:
                    return f"{year:04d}-{month}-{day:02d}"
        except (ValueError, IndexError):
            pass

        return None

    @classmethod
    def parse_investing_date(cls, value: str) -> Optional[str]:
        """
        Public method to parse Investing.com date formats.

        This is a convenience wrapper for external use.

        Args:
            value: Date string from Investing.com

        Returns:
            ISO date string (YYYY-MM-DD) or None
        """
        return cls._parse_investing_format(value) or cls.parse(value)

    @classmethod
    def parse_to_date(cls, value: Any) -> Optional[date]:
        """
        Parse any date value to a Python date object.

        Args:
            value: Any date representation

        Returns:
            date object or None
        """
        iso_str = cls.parse(value)
        if iso_str:
            return datetime.strptime(iso_str, '%Y-%m-%d').date()
        return None

    @classmethod
    def parse_to_datetime(cls, value: Any) -> Optional[datetime]:
        """
        Parse any date value to a Python datetime object (midnight).

        Args:
            value: Any date representation

        Returns:
            datetime object or None
        """
        iso_str = cls.parse(value)
        if iso_str:
            return datetime.strptime(iso_str, '%Y-%m-%d')
        return None

    @classmethod
    def is_valid_date(cls, value: Any) -> bool:
        """
        Check if value can be parsed as a valid date.

        Args:
            value: Any value to check

        Returns:
            True if parseable, False otherwise
        """
        return cls.parse(value) is not None

    @classmethod
    def normalize_dates_dict(
        cls,
        data: dict,
        date_key: str = 'fecha'
    ) -> dict:
        """
        Normalize all date keys in a dictionary to ISO format.

        Args:
            data: Dictionary with date keys
            date_key: Name of the date field if data is list of dicts

        Returns:
            Dictionary with normalized date keys
        """
        if not data:
            return data

        normalized = {}
        for key, value in data.items():
            parsed = cls.parse(key)
            if parsed:
                normalized[parsed] = value

        return normalized

    @classmethod
    def parse_banrep_date(cls, value: str) -> Optional[str]:
        """
        Parse BanRep SUAMECA date format (YYYY/MM/DD).

        Args:
            value: Date string from BanRep

        Returns:
            ISO date string or None
        """
        if not value:
            return None

        # BanRep uses YYYY/MM/DD format
        try:
            return datetime.strptime(value.strip(), '%Y/%m/%d').strftime('%Y-%m-%d')
        except ValueError:
            # Fall back to general parser
            return cls.parse(value)

    @classmethod
    def format_for_api(cls, value: Any, api_format: str = 'iso') -> Optional[str]:
        """
        Format a date for a specific API.

        Args:
            value: Date to format
            api_format: Target format ('iso', 'fred', 'twelvedata')

        Returns:
            Formatted date string
        """
        parsed = cls.parse_to_date(value)
        if not parsed:
            return None

        formats = {
            'iso': '%Y-%m-%d',
            'fred': '%Y-%m-%d',
            'twelvedata': '%Y-%m-%d',
            'compact': '%Y%m%d',
        }

        fmt = formats.get(api_format, '%Y-%m-%d')
        return parsed.strftime(fmt)


class DateValidator:
    """
    Validates dates against trading calendar rules.
    """

    @staticmethod
    def is_weekend(date_obj: Union[date, datetime, str]) -> bool:
        """Check if a date is a weekend (Saturday=5, Sunday=6)."""
        if isinstance(date_obj, str):
            date_obj = DateParser.parse_to_date(date_obj)
        if isinstance(date_obj, datetime):
            date_obj = date_obj.date()
        if date_obj is None:
            return False
        return date_obj.weekday() >= 5

    @staticmethod
    def is_future(date_obj: Union[date, datetime, str]) -> bool:
        """Check if a date is in the future."""
        if isinstance(date_obj, str):
            date_obj = DateParser.parse_to_date(date_obj)
        if isinstance(date_obj, datetime):
            date_obj = date_obj.date()
        if date_obj is None:
            return False
        return date_obj > date.today()

    @staticmethod
    def days_since(date_obj: Union[date, datetime, str]) -> int:
        """Calculate days since a date."""
        if isinstance(date_obj, str):
            date_obj = DateParser.parse_to_date(date_obj)
        if isinstance(date_obj, datetime):
            date_obj = date_obj.date()
        if date_obj is None:
            return -1
        return (date.today() - date_obj).days
