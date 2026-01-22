"""
Unit Tests for Macro Extraction Service
=======================================

Tests for the MacroExtractionService orchestrator.

Contract: CTR-L0-SERVICE-001-TEST
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

import sys
sys.path.insert(0, '/opt/airflow')

from src.core.interfaces.macro_extractor import ExtractionResult
from src.core.factories.macro_extractor_factory import MacroSource


class TestMacroExtractionService:
    """Test cases for MacroExtractionService."""

    @patch('airflow.dags.services.macro_extraction_service.get_extractor_factory')
    def test_service_initialization(self, mock_factory):
        """Test service initialization."""
        from airflow.dags.services.macro_extraction_service import MacroExtractionService

        mock_factory.return_value = Mock()
        service = MacroExtractionService(config_path='/test/config.yaml')

        assert service.config_path == '/test/config.yaml'

    @patch('airflow.dags.services.macro_extraction_service.get_extractor_factory')
    def test_extract_all_sources(self, mock_factory_func):
        """Test extracting from all sources."""
        from airflow.dags.services.macro_extraction_service import MacroExtractionService

        # Create mock factory
        mock_factory = Mock()
        mock_factory.config = {
            'global': {'retry_attempts': 1, 'retry_delay_seconds': 1},
            'enabled_sources': ['fred'],
        }

        # Create mock extractor
        mock_extractor = Mock()
        mock_extractor.extract.return_value = ExtractionResult(
            data={'2024-01-15': {'dxy': 103.5}},
            records_extracted=1,
            source_name='fred'
        )

        mock_factory.create_all_extractors.return_value = {
            MacroSource.FRED: mock_extractor
        }
        mock_factory.get_indicators_for_source.return_value = {'DTWEXBGS': 'dxy'}
        mock_factory.get_lookback_days.return_value = 90

        mock_factory_func.return_value = mock_factory

        # Create service and extract
        service = MacroExtractionService()
        mock_ti = Mock()
        results = service.extract_all(ti=mock_ti)

        assert 'fred' in results
        assert results['fred']['records_extracted'] == 1

    @patch('airflow.dags.services.macro_extraction_service.get_extractor_factory')
    def test_extract_with_retry(self, mock_factory_func):
        """Test extraction with retry logic."""
        from airflow.dags.services.macro_extraction_service import MacroExtractionService

        mock_factory = Mock()
        mock_factory.config = {
            'global': {'retry_attempts': 3, 'retry_delay_seconds': 0.1}
        }
        mock_factory_func.return_value = mock_factory

        # Create mock extractor that fails first 2 times
        call_count = [0]

        def mock_extract(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("API error")
            return ExtractionResult(
                data={'2024-01-15': {'dxy': 103.5}},
                records_extracted=1,
                source_name='fred'
            )

        mock_extractor = Mock()
        mock_extractor.extract = mock_extract

        service = MacroExtractionService()
        result = service._extract_with_retry(
            extractor=mock_extractor,
            indicators={'TEST': 'test'},
            lookback_days=90,
            source=MacroSource.FRED
        )

        assert result.is_successful
        assert call_count[0] == 3

    @patch('airflow.dags.services.macro_extraction_service.get_extractor_factory')
    def test_extract_single_source(self, mock_factory_func):
        """Test extracting from a single source."""
        from airflow.dags.services.macro_extraction_service import MacroExtractionService

        mock_factory = Mock()
        mock_factory.config = {
            'global': {'retry_attempts': 1, 'retry_delay_seconds': 1}
        }
        mock_factory.is_source_enabled.return_value = True
        mock_factory.get_indicators_for_source.return_value = {'TEST': 'test'}
        mock_factory.get_lookback_days.return_value = 90

        mock_extractor = Mock()
        mock_extractor.extract.return_value = ExtractionResult(
            data={'2024-01-15': {'test': 100}},
            records_extracted=1,
            source_name='fred'
        )
        mock_factory.create.return_value = mock_extractor

        mock_factory_func.return_value = mock_factory

        service = MacroExtractionService()
        mock_ti = Mock()
        result = service.extract_single(MacroSource.FRED, ti=mock_ti)

        assert result['records_extracted'] == 1

    @patch('airflow.dags.services.macro_extraction_service.get_extractor_factory')
    def test_extract_single_disabled_source(self, mock_factory_func):
        """Test extracting from a disabled source."""
        from airflow.dags.services.macro_extraction_service import MacroExtractionService

        mock_factory = Mock()
        mock_factory.is_source_enabled.return_value = False
        mock_factory_func.return_value = mock_factory

        service = MacroExtractionService()
        result = service.extract_single(MacroSource.FRED)

        assert result['records_extracted'] == 0
        assert 'not enabled' in result['errors'][0].lower()

    @patch('airflow.dags.services.macro_extraction_service.get_extractor_factory')
    def test_get_extraction_summary(self, mock_factory_func):
        """Test extraction summary generation."""
        from airflow.dags.services.macro_extraction_service import MacroExtractionService

        mock_factory = Mock()
        mock_factory.config = {}
        mock_factory_func.return_value = mock_factory

        service = MacroExtractionService()

        results = {
            'fred': {'records_extracted': 10, 'errors': []},
            'twelvedata': {'records_extracted': 5, 'errors': ['warning']},
            'investing': {'records_extracted': 0, 'errors': ['failed']},
        }

        summary = service.get_extraction_summary(results)

        assert summary['total_sources'] == 3
        assert summary['successful_sources'] == 2
        assert summary['failed_sources'] == 1
        assert summary['total_records'] == 15
        assert summary['total_errors'] == 2


class TestExtractionTaskFunctions:
    """Test Airflow task functions."""

    @patch('airflow.dags.services.macro_extraction_service.MacroExtractionService')
    def test_extract_all_sources_task(self, mock_service_class):
        """Test extract_all_sources task function."""
        from airflow.dags.services.macro_extraction_service import extract_all_sources

        mock_service = Mock()
        mock_service.extract_all.return_value = {'fred': {'records_extracted': 1}}
        mock_service.get_extraction_summary.return_value = {'total_records': 1}
        mock_service_class.return_value = mock_service

        result = extract_all_sources(ti=Mock())

        assert result['total_records'] == 1
        mock_service.extract_all.assert_called_once()

    @patch('airflow.dags.services.macro_extraction_service.MacroExtractionService')
    def test_extract_fred_task(self, mock_service_class):
        """Test extract_fred task function."""
        from airflow.dags.services.macro_extraction_service import extract_fred

        mock_service = Mock()
        mock_service.extract_single.return_value = {'records_extracted': 10}
        mock_service_class.return_value = mock_service

        result = extract_fred(ti=Mock())

        assert result['records_extracted'] == 10
        mock_service.extract_single.assert_called_once_with(MacroSource.FRED, ti=mock_service.extract_single.call_args[1]['ti'])


class TestXComKeyMapping:
    """Test XCom key mapping."""

    def test_xcom_key_mapping_completeness(self):
        """Test all sources have XCom key mappings."""
        from airflow.dags.services.macro_extraction_service import MacroExtractionService

        # Check all MacroSource values have mapping
        for source in MacroSource:
            assert source in MacroExtractionService.XCOM_KEY_MAPPING, \
                f"Missing XCom key mapping for {source}"

    def test_xcom_key_values(self):
        """Test XCom key values are valid."""
        from airflow.dags.services.macro_extraction_service import MacroExtractionService
        from contracts.l0_data_contracts import L0XComKeys

        for source, xcom_key in MacroExtractionService.XCOM_KEY_MAPPING.items():
            assert isinstance(xcom_key, L0XComKeys), \
                f"XCom key for {source} is not L0XComKeys enum"


class TestMacroMergeService:
    """Test cases for MacroMergeService."""

    def test_source_release_offsets(self):
        """Test source release offset values."""
        from airflow.dags.services.macro_merge_service import MacroMergeService

        service = MacroMergeService()

        # Check expected offsets
        assert service.SOURCE_RELEASE_OFFSETS['fred'] == 1
        assert service.SOURCE_RELEASE_OFFSETS['twelvedata'] == 0
        assert service.SOURCE_RELEASE_OFFSETS['fedesarrollo'] == 15
        assert service.SOURCE_RELEASE_OFFSETS['dane'] == 45


class TestMacroCleanupService:
    """Test cases for MacroCleanupService."""

    def test_is_weekend(self):
        """Test weekend detection."""
        from airflow.dags.services.macro_cleanup_service import MacroCleanupService
        from datetime import date

        service = MacroCleanupService()

        # Saturday
        assert service.is_weekend(date(2024, 1, 13)) is True
        # Sunday
        assert service.is_weekend(date(2024, 1, 14)) is True
        # Monday
        assert service.is_weekend(date(2024, 1, 15)) is False

    def test_is_us_holiday_fixed(self):
        """Test US fixed holiday detection."""
        from airflow.dags.services.macro_cleanup_service import MacroCleanupService
        from datetime import date

        service = MacroCleanupService()

        # New Year's Day
        assert service.is_us_holiday(date(2024, 1, 1)) is True
        # Independence Day
        assert service.is_us_holiday(date(2024, 7, 4)) is True
        # Christmas
        assert service.is_us_holiday(date(2024, 12, 25)) is True
        # Regular day
        assert service.is_us_holiday(date(2024, 1, 15)) is False

    def test_is_trading_day(self):
        """Test trading day detection."""
        from airflow.dags.services.macro_cleanup_service import MacroCleanupService
        from datetime import date

        service = MacroCleanupService()

        # Regular trading day (Monday)
        assert service.is_trading_day(date(2024, 1, 15)) is True
        # Weekend
        assert service.is_trading_day(date(2024, 1, 13)) is False
        # Holiday
        assert service.is_trading_day(date(2024, 1, 1)) is False

    def test_get_non_trading_days(self):
        """Test getting non-trading days in range."""
        from airflow.dags.services.macro_cleanup_service import MacroCleanupService
        from datetime import date

        service = MacroCleanupService()

        # One week range
        non_trading = service.get_non_trading_days(
            date(2024, 1, 8),  # Monday
            date(2024, 1, 14)  # Sunday
        )

        # Should have at least Saturday and Sunday
        reasons = [d['reason'] for d in non_trading]
        assert 'weekend' in reasons


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
