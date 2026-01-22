"""
Macro Extraction Service
========================

Orchestrates macro data extraction across all sources.
Implements retry logic, error handling, and metrics recording.

Contract: CTR-L0-SERVICE-001

Pattern: Service Orchestrator
    - Coordinates multiple extraction strategies
    - Handles parallel/sequential execution
    - Aggregates results for downstream processing

Usage:
    service = MacroExtractionService()
    results = service.extract_all(**context)

Version: 1.0.0
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

# Add paths for imports
sys.path.insert(0, '/opt/airflow')
sys.path.insert(0, '/opt/airflow/dags')

from src.core.factories.macro_extractor_factory import (
    MacroExtractorFactory,
    MacroSource,
    get_extractor_factory,
)
from src.core.interfaces.macro_extractor import ExtractionResult

# CRITICAL: Import strategies module to trigger decorator registration
# This must happen BEFORE any factory.create_all_extractors() calls
import services.macro_extraction_strategies  # noqa: F401 - imported for side effects

# Import contracts
from contracts.l0_data_contracts import (
    L0XComKeys,
    ExtractionBatchResult,
    ExtractionAttempt,
    ExtractionOutcome,
    DataSourceType,
)

logger = logging.getLogger(__name__)


class MacroExtractionService:
    """
    Orchestrates macro data extraction across all sources.

    Coordinates multiple extraction strategies, handles retries,
    and aggregates results for downstream processing.

    Attributes:
        factory: MacroExtractorFactory instance
        config: Service configuration
    """

    # Mapping from MacroSource to L0XComKeys
    XCOM_KEY_MAPPING = {
        MacroSource.FRED: L0XComKeys.FRED_DATA,
        MacroSource.TWELVEDATA: L0XComKeys.TWELVEDATA_DATA,
        MacroSource.INVESTING: L0XComKeys.INVESTING_DATA,
        MacroSource.BANREP: L0XComKeys.BANREP_DATA,
        MacroSource.BANREP_BOP: L0XComKeys.BANREP_BOP_DATA,
        MacroSource.BCRP: L0XComKeys.EMBI_DATA,
        MacroSource.FEDESARROLLO: L0XComKeys.FEDESARROLLO_DATA,
        MacroSource.DANE: L0XComKeys.DANE_DATA,
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize service with configuration.

        Args:
            config_path: Path to l0_macro_sources.yaml
        """
        self.config_path = config_path or '/opt/airflow/config/l0_macro_sources.yaml'
        self._factory: Optional[MacroExtractorFactory] = None

    @property
    def factory(self) -> MacroExtractorFactory:
        """Lazy-load factory instance."""
        if self._factory is None:
            self._factory = get_extractor_factory(self.config_path)
        return self._factory

    @property
    def config(self) -> Dict[str, Any]:
        """Get configuration from factory."""
        return self.factory.config

    def extract_all(self, **context) -> Dict[str, Dict]:
        """
        Extract data from all configured sources.

        This is the main entry point for Airflow task execution.
        Iterates through all enabled sources and extracts data.

        Args:
            **context: Airflow context (includes 'ti' for XCom)

        Returns:
            Dictionary mapping source name to extraction results
        """
        logger.info("=" * 60)
        logger.info("MACRO EXTRACTION SERVICE - Starting extraction")
        logger.info("=" * 60)

        results = {}
        all_extractors = self.factory.create_all_extractors()

        for source, extractor in all_extractors.items():
            try:
                # Get source-specific configuration
                indicators = self.factory.get_indicators_for_source(source)
                lookback = self.factory.get_lookback_days(source)

                logger.info(f"\n[{source.value.upper()}] Starting extraction...")
                logger.info(f"  Indicators: {len(indicators)}")
                logger.info(f"  Lookback: {lookback} days")

                # Execute extraction with retry logic
                result = self._extract_with_retry(
                    extractor=extractor,
                    indicators=indicators,
                    lookback_days=lookback,
                    source=source
                )

                # Store result
                results[source.value] = result.to_dict()

                # Push to XCom
                xcom_key = self.XCOM_KEY_MAPPING.get(source)
                if xcom_key and 'ti' in context:
                    context['ti'].xcom_push(
                        key=xcom_key.value,
                        value=result.data
                    )

                # Log summary
                logger.info(f"[{source.value.upper()}] Completed: "
                           f"{result.records_extracted} records, "
                           f"{len(result.errors)} errors")

            except Exception as e:
                logger.error(f"[{source.value.upper()}] Failed: {e}")
                results[source.value] = {
                    'data': {},
                    'errors': [str(e)],
                    'records_extracted': 0,
                    'source_name': source.value,
                }

        # Log final summary
        total_records = sum(
            r.get('records_extracted', 0) for r in results.values()
        )
        total_errors = sum(
            len(r.get('errors', [])) for r in results.values()
        )
        logger.info("=" * 60)
        logger.info(f"EXTRACTION COMPLETE: {total_records} total records, {total_errors} errors")
        logger.info("=" * 60)

        return results

    def _extract_with_retry(
        self,
        extractor,
        indicators: Dict[str, str],
        lookback_days: int,
        source: MacroSource
    ) -> ExtractionResult:
        """
        Execute extraction with retry logic.

        Args:
            extractor: Extractor instance
            indicators: Indicator mappings
            lookback_days: Days of lookback
            source: Source enum

        Returns:
            ExtractionResult
        """
        global_config = self.config.get('global', {})
        max_retries = global_config.get('retry_attempts', 3)
        retry_delay = global_config.get('retry_delay_seconds', 60)

        last_error = None

        for attempt in range(max_retries):
            try:
                logger.info(f"[{source.value}] Attempt {attempt + 1}/{max_retries}")
                result = extractor.extract(indicators, lookback_days)

                if result.is_successful:
                    return result

                # If no data but no errors, might be a temporary issue
                if not result.has_errors:
                    logger.warning(f"[{source.value}] No data returned, will retry")
                else:
                    # Has errors, return as is
                    return result

            except Exception as e:
                last_error = e
                logger.error(f"[{source.value}] Attempt {attempt + 1} failed: {e}")

            # Wait before retry (exponential backoff)
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                logger.info(f"[{source.value}] Waiting {wait_time}s before retry...")
                time.sleep(wait_time)

        # All retries failed
        return ExtractionResult(
            data={},
            errors=[f"All {max_retries} attempts failed. Last error: {last_error}"],
            records_extracted=0,
            source_name=source.value,
        )

    def extract_single(
        self,
        source: MacroSource,
        **context
    ) -> Dict[str, Any]:
        """
        Extract data from a single source.

        Useful for targeted re-extraction or testing.

        Args:
            source: Source to extract from
            **context: Airflow context

        Returns:
            Extraction result dictionary
        """
        if not self.factory.is_source_enabled(source):
            logger.warning(f"[{source.value}] Source is not enabled")
            return {'data': {}, 'errors': ['Source not enabled'], 'records_extracted': 0}

        extractor = self.factory.create(source)
        indicators = self.factory.get_indicators_for_source(source)
        lookback = self.factory.get_lookback_days(source)

        result = self._extract_with_retry(
            extractor=extractor,
            indicators=indicators,
            lookback_days=lookback,
            source=source
        )

        # Push to XCom
        xcom_key = self.XCOM_KEY_MAPPING.get(source)
        if xcom_key and 'ti' in context:
            context['ti'].xcom_push(key=xcom_key.value, value=result.data)

        return result.to_dict()

    def get_extraction_summary(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Generate summary of extraction results.

        Args:
            results: Results from extract_all()

        Returns:
            Summary dictionary
        """
        summary = {
            'total_sources': len(results),
            'successful_sources': 0,
            'failed_sources': 0,
            'total_records': 0,
            'total_errors': 0,
            'by_source': {},
        }

        for source, result in results.items():
            records = result.get('records_extracted', 0)
            errors = result.get('errors', [])

            summary['total_records'] += records
            summary['total_errors'] += len(errors)

            if records > 0:
                summary['successful_sources'] += 1
            else:
                summary['failed_sources'] += 1

            summary['by_source'][source] = {
                'records': records,
                'errors': len(errors),
                'success': records > 0,
            }

        return summary


# =============================================================================
# Airflow Task Functions
# =============================================================================

def extract_all_sources(**context) -> Dict[str, Any]:
    """
    Airflow task function for extracting from all sources.

    This is the main task callable for the DAG.

    Args:
        **context: Airflow context

    Returns:
        Extraction results summary
    """
    service = MacroExtractionService()
    results = service.extract_all(**context)
    return service.get_extraction_summary(results)


def extract_fred(**context) -> Dict[str, Any]:
    """Extract from FRED API only."""
    service = MacroExtractionService()
    return service.extract_single(MacroSource.FRED, **context)


def extract_twelvedata(**context) -> Dict[str, Any]:
    """Extract from TwelveData API only."""
    service = MacroExtractionService()
    return service.extract_single(MacroSource.TWELVEDATA, **context)


def extract_investing(**context) -> Dict[str, Any]:
    """Extract from Investing.com only."""
    service = MacroExtractionService()
    return service.extract_single(MacroSource.INVESTING, **context)


def extract_banrep(**context) -> Dict[str, Any]:
    """Extract from BanRep SUAMECA only."""
    service = MacroExtractionService()
    return service.extract_single(MacroSource.BANREP, **context)


def extract_bcrp(**context) -> Dict[str, Any]:
    """Extract EMBI from BCRP only."""
    service = MacroExtractionService()
    return service.extract_single(MacroSource.BCRP, **context)


def extract_fedesarrollo(**context) -> Dict[str, Any]:
    """Extract from Fedesarrollo only."""
    service = MacroExtractionService()
    return service.extract_single(MacroSource.FEDESARROLLO, **context)


def extract_dane(**context) -> Dict[str, Any]:
    """Extract from DANE only."""
    service = MacroExtractionService()
    return service.extract_single(MacroSource.DANE, **context)


def extract_banrep_bop(**context) -> Dict[str, Any]:
    """Extract Balance of Payments data from BanRep SUAMECA."""
    service = MacroExtractionService()
    return service.extract_single(MacroSource.BANREP_BOP, **context)
