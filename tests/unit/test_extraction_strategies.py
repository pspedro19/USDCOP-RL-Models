"""
Unit Tests for Macro Extraction Strategies
==========================================

Tests for the Strategy Pattern extraction implementations.

Contract: CTR-L0-STRATEGY-001-TEST
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import pandas as pd

import sys
sys.path.insert(0, '/opt/airflow')

from src.core.interfaces.macro_extractor import (
    BaseMacroExtractor,
    ConfigurableExtractor,
    ExtractionResult,
)
from src.core.factories.macro_extractor_factory import (
    MacroExtractorFactory,
    MacroSource,
)


class TestExtractionResult:
    """Test cases for ExtractionResult dataclass."""

    def test_empty_result(self):
        """Test empty extraction result."""
        result = ExtractionResult()
        assert result.data == {}
        assert result.errors == []
        assert result.records_extracted == 0
        assert result.is_successful is False
        assert result.has_errors is False

    def test_successful_result(self):
        """Test successful extraction result."""
        result = ExtractionResult(
            data={'2024-01-15': {'dxy': 103.5}},
            errors=[],
            records_extracted=1,
            source_name='fred'
        )
        assert result.is_successful is True
        assert result.has_errors is False

    def test_result_with_errors(self):
        """Test result with errors."""
        result = ExtractionResult(
            data={},
            errors=['API error', 'Timeout'],
            records_extracted=0,
            source_name='fred'
        )
        assert result.is_successful is False
        assert result.has_errors is True

    def test_to_dict_serialization(self):
        """Test to_dict serialization."""
        result = ExtractionResult(
            data={'2024-01-15': {'dxy': 103.5}},
            errors=['warning'],
            records_extracted=1,
            source_name='fred',
            duration_seconds=1.5
        )
        d = result.to_dict()
        assert d['data'] == {'2024-01-15': {'dxy': 103.5}}
        assert d['errors'] == ['warning']
        assert d['records_extracted'] == 1
        assert d['source_name'] == 'fred'
        assert d['duration_seconds'] == 1.5

    def test_from_dict_deserialization(self):
        """Test from_dict deserialization."""
        data = {
            'data': {'2024-01-15': {'dxy': 103.5}},
            'errors': ['warning'],
            'records_extracted': 1,
            'source_name': 'fred',
            'extraction_time': '2024-01-15T12:00:00',
            'duration_seconds': 1.5
        }
        result = ExtractionResult.from_dict(data)
        assert result.data == {'2024-01-15': {'dxy': 103.5}}
        assert result.errors == ['warning']
        assert result.records_extracted == 1
        assert result.source_name == 'fred'


class TestMacroExtractorFactory:
    """Test cases for MacroExtractorFactory."""

    def test_macro_source_enum(self):
        """Test MacroSource enum values."""
        assert MacroSource.FRED.value == "fred"
        assert MacroSource.TWELVEDATA.value == "twelvedata"
        assert MacroSource.INVESTING.value == "investing"
        assert MacroSource.BANREP.value == "banrep"
        assert MacroSource.BCRP.value == "bcrp"
        assert MacroSource.FEDESARROLLO.value == "fedesarrollo"
        assert MacroSource.DANE.value == "dane"

    def test_factory_register_decorator(self):
        """Test strategy registration via decorator."""
        # Create a test strategy
        @MacroExtractorFactory.register_strategy(MacroSource.FRED)
        class TestStrategy(BaseMacroExtractor):
            @property
            def source_name(self):
                return "test"

            def extract(self, indicators, lookback_days, **kwargs):
                return ExtractionResult()

        assert MacroSource.FRED in MacroExtractorFactory._strategies

    def test_factory_get_registered_sources(self):
        """Test getting list of registered sources."""
        sources = MacroExtractorFactory.get_registered_sources()
        assert isinstance(sources, list)


class TestConfigurableExtractor:
    """Test cases for ConfigurableExtractor base class."""

    def test_get_config_value(self):
        """Test getting configuration values."""
        class TestExtractor(ConfigurableExtractor):
            @property
            def source_name(self):
                return "test"

            def extract(self, indicators, lookback_days, **kwargs):
                return ExtractionResult()

        extractor = TestExtractor(config={'timeout': 30, 'retries': 3})
        assert extractor.get_config_value('timeout') == 30
        assert extractor.get_config_value('retries') == 3
        assert extractor.get_config_value('missing', 'default') == 'default'

    def test_get_timeout(self):
        """Test get_timeout method."""
        class TestExtractor(ConfigurableExtractor):
            @property
            def source_name(self):
                return "test"

            def extract(self, indicators, lookback_days, **kwargs):
                return ExtractionResult()

        extractor = TestExtractor(config={'request_timeout_seconds': 45})
        assert extractor.get_timeout() == 45

        # Default timeout
        extractor2 = TestExtractor(config={})
        assert extractor2.get_timeout() == 30

    def test_get_retry_config(self):
        """Test get_retry_config method."""
        class TestExtractor(ConfigurableExtractor):
            @property
            def source_name(self):
                return "test"

            def extract(self, indicators, lookback_days, **kwargs):
                return ExtractionResult()

        extractor = TestExtractor(config={
            'retry_attempts': 5,
            'retry_delay_seconds': 120
        })
        attempts, delay = extractor.get_retry_config()
        assert attempts == 5
        assert delay == 120


class TestFREDStrategy:
    """Test cases for FRED extraction strategy."""

    @patch('airflow.dags.services.macro_extraction_strategies.Fred')
    def test_fred_extract_success(self, mock_fred_class):
        """Test FRED extraction with successful API response."""
        # Import strategy (triggers registration)
        from airflow.dags.services.macro_extraction_strategies import FREDExtractionStrategy

        # Mock FRED API
        mock_fred = Mock()
        mock_fred_class.return_value = mock_fred

        # Create mock series data
        mock_series = pd.Series(
            [103.5, 103.8, 104.0],
            index=pd.to_datetime(['2024-01-15', '2024-01-16', '2024-01-17'])
        )
        mock_fred.get_series.return_value = mock_series

        # Create extractor with mock API key
        with patch.object(FREDExtractionStrategy, 'get_api_key', return_value='test_key'):
            extractor = FREDExtractionStrategy(config={})
            result = extractor.extract(
                indicators={'DTWEXBGS': 'fxrt_index_dxy_usa_d_dxy'},
                lookback_days=90
            )

        assert result.is_successful
        assert '2024-01-15' in result.data
        assert result.data['2024-01-15']['fxrt_index_dxy_usa_d_dxy'] == 103.5

    def test_fred_extract_no_api_key(self):
        """Test FRED extraction without API key."""
        from airflow.dags.services.macro_extraction_strategies import FREDExtractionStrategy

        with patch.object(FREDExtractionStrategy, 'get_api_key', return_value=None):
            extractor = FREDExtractionStrategy(config={})
            result = extractor.extract(
                indicators={'DTWEXBGS': 'fxrt_index_dxy_usa_d_dxy'},
                lookback_days=90
            )

        assert not result.is_successful
        assert result.has_errors
        assert 'not found' in result.errors[0].lower()


class TestTwelveDataStrategy:
    """Test cases for TwelveData extraction strategy."""

    @patch('airflow.dags.services.macro_extraction_strategies.requests')
    def test_twelvedata_extract_success(self, mock_requests):
        """Test TwelveData extraction with successful API response."""
        from airflow.dags.services.macro_extraction_strategies import TwelveDataExtractionStrategy

        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = {
            'values': [
                {'datetime': '2024-01-15', 'close': '17.50'},
                {'datetime': '2024-01-16', 'close': '17.55'},
            ]
        }
        mock_requests.get.return_value = mock_response

        with patch.object(TwelveDataExtractionStrategy, 'get_api_key', return_value='test_key'):
            extractor = TwelveDataExtractionStrategy(config={
                'interval': '1day',
                'outputsize': 60
            })
            result = extractor.extract(
                indicators={'USD/MXN': 'fxrt_spot_usdmxn_mex_d_usdmxn'},
                lookback_days=60
            )

        assert result.is_successful
        assert '2024-01-15' in result.data
        assert result.data['2024-01-15']['fxrt_spot_usdmxn_mex_d_usdmxn'] == 17.50


class TestBCRPStrategy:
    """Test cases for BCRP (EMBI) extraction strategy."""

    def test_parse_embi_date(self):
        """Test EMBI date parsing."""
        from airflow.dags.services.macro_extraction_strategies import BCRPExtractionStrategy

        extractor = BCRPExtractionStrategy(config={})

        assert extractor._parse_embi_date('06Ene26') == '2026-01-06'
        assert extractor._parse_embi_date('17Nov25') == '2025-11-17'
        assert extractor._parse_embi_date('31Dic24') == '2024-12-31'
        assert extractor._parse_embi_date('invalid') is None


class TestStrategyValidation:
    """Test strategy validation methods."""

    def test_validate_indicators_empty(self):
        """Test validation with empty indicators."""
        class TestExtractor(BaseMacroExtractor):
            @property
            def source_name(self):
                return "test"

            def extract(self, indicators, lookback_days, **kwargs):
                return ExtractionResult()

        extractor = TestExtractor(config={})
        errors = extractor.validate_indicators({})
        assert len(errors) > 0
        assert 'No indicators' in errors[0]

    def test_validate_indicators_valid(self):
        """Test validation with valid indicators."""
        class TestExtractor(BaseMacroExtractor):
            @property
            def source_name(self):
                return "test"

            def extract(self, indicators, lookback_days, **kwargs):
                return ExtractionResult()

        extractor = TestExtractor(config={})
        errors = extractor.validate_indicators({'TEST': 'test_column'})
        assert len(errors) == 0


class TestDateNormalization:
    """Test date normalization in extractors."""

    def test_normalize_date_iso(self):
        """Test date normalization with ISO format."""
        class TestExtractor(BaseMacroExtractor):
            @property
            def source_name(self):
                return "test"

            def extract(self, indicators, lookback_days, **kwargs):
                return ExtractionResult()

        extractor = TestExtractor(config={})
        assert extractor._normalize_date('2024-01-15') == '2024-01-15'

    def test_normalize_date_datetime(self):
        """Test date normalization with datetime object."""
        class TestExtractor(BaseMacroExtractor):
            @property
            def source_name(self):
                return "test"

            def extract(self, indicators, lookback_days, **kwargs):
                return ExtractionResult()

        extractor = TestExtractor(config={})
        dt = datetime(2024, 1, 15, 12, 30, 45)
        assert extractor._normalize_date(dt) == '2024-01-15'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
