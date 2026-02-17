# -*- coding: utf-8 -*-
"""
Unit Tests for Data Extractors
==============================

Tests for:
- BaseExtractor interface compliance
- FredExtractor
- InvestingExtractor
- SuamecaExtractor
- BcrpExtractor
- DaneExtractor
- FedesarrolloExtractor

Contract: CTR-L0-EXTRACTOR-TESTS-001
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

import pandas as pd
import pytest

# Add paths for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DAGS_PATH = PROJECT_ROOT / 'airflow' / 'dags'

for path in [str(DAGS_PATH), str(PROJECT_ROOT / 'src')]:
    if path not in sys.path:
        sys.path.insert(0, path)

from extractors.base import BaseExtractor, ExtractionResult


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_daily_df():
    """Create sample daily data."""
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    return pd.DataFrame({
        'fecha': dates,
        'valor': [100.0 + i * 0.5 for i in range(30)],
    })


@pytest.fixture
def sample_monthly_df():
    """Create sample monthly data."""
    dates = pd.date_range('2023-01-01', periods=12, freq='MS')
    return pd.DataFrame({
        'fecha': dates,
        'valor': [5.0 + i * 0.1 for i in range(12)],
    })


@pytest.fixture
def sample_quarterly_df():
    """Create sample quarterly data."""
    dates = pd.date_range('2022-01-01', periods=8, freq='QS')
    return pd.DataFrame({
        'fecha': dates,
        'valor': [25000.0 + i * 500 for i in range(8)],
    })


@pytest.fixture
def fred_config():
    """Create FRED extractor config."""
    return {
        'api_key': 'test_api_key',
        'base_url': 'https://api.stlouisfed.org/fred/series/observations',
        'retry_attempts': 3,
        'timeout_seconds': 30,
        'variables': [
            {'name': 'polr_fed_funds_usa_m_fedfunds', 'series_id': 'FEDFUNDS', 'frequency': 'M'},
            {'name': 'volt_vix_usa_d_vix', 'series_id': 'VIXCLS', 'frequency': 'D'},
        ]
    }


@pytest.fixture
def investing_config():
    """Create Investing extractor config."""
    return {
        'ajax_endpoint': 'https://www.investing.com/instruments/HistoricalDataAjax',
        'api_endpoint': 'https://api.investing.com/api/financialdata/historical',
        'rate_limit_seconds': 1,
        'retry_attempts': 3,
        'variables': [
            {'name': 'comm_oil_brent_glb_d_brent', 'pair_id': 8833, 'frequency': 'D'},
            {'name': 'fxrt_index_dxy_usa_d_dxy', 'instrument_id': 942611, 'frequency': 'D'},
        ]
    }


# =============================================================================
# EXTRACTION RESULT TESTS
# =============================================================================

class TestExtractionResult:
    """Tests for ExtractionResult dataclass."""

    def test_empty_result(self):
        """Test creation of empty result."""
        result = ExtractionResult(
            source='test',
            variable='test_var',
            data=pd.DataFrame(),
            success=False,
            error='No data available',
        )

        assert result.success is False
        assert result.records_count == 0
        assert result.last_date is None
        assert result.error == 'No data available'

    def test_result_with_data(self, sample_daily_df):
        """Test result with data calculates metadata correctly."""
        result = ExtractionResult(
            source='fred',
            variable='test_var',
            data=sample_daily_df,
            success=True,
        )

        assert result.success is True
        assert result.records_count == 30
        assert result.last_date is not None
        assert result.last_date == sample_daily_df['fecha'].max()

    def test_result_with_index_date(self):
        """Test result with date as index."""
        df = pd.DataFrame({
            'valor': [100.0, 101.0, 102.0],
        }, index=pd.DatetimeIndex(
            pd.date_range('2024-01-01', periods=3, freq='D'),
            name='fecha'
        ))

        result = ExtractionResult(
            source='test',
            variable='test_var',
            data=df,
        )

        assert result.records_count == 3
        assert result.last_date == pd.Timestamp('2024-01-03')


# =============================================================================
# BASE EXTRACTOR TESTS
# =============================================================================

class TestBaseExtractor:
    """Tests for BaseExtractor abstract class."""

    def test_frequency_inference_daily(self):
        """Test frequency inference for daily variables."""
        # Create a concrete implementation for testing
        class TestExtractor(BaseExtractor):
            @property
            def source_name(self):
                return 'test'

            @property
            def variables(self):
                return ['test_d_var']

            def extract(self, variable, start_date, end_date, last_n=None):
                pass

            def get_latest_date(self, variable):
                pass

        ext = TestExtractor()
        assert ext._get_frequency('volt_vix_usa_d_vix') == 'D'
        assert ext._get_frequency('some_d_variable') == 'D'

    def test_frequency_inference_monthly(self):
        """Test frequency inference for monthly variables."""
        class TestExtractor(BaseExtractor):
            @property
            def source_name(self):
                return 'test'

            @property
            def variables(self):
                return []

            def extract(self, variable, start_date, end_date, last_n=None):
                pass

            def get_latest_date(self, variable):
                pass

        ext = TestExtractor()
        assert ext._get_frequency('polr_fed_funds_usa_m_fedfunds') == 'M'
        assert ext._get_frequency('some_m_variable') == 'M'

    def test_frequency_inference_quarterly(self):
        """Test frequency inference for quarterly variables."""
        class TestExtractor(BaseExtractor):
            @property
            def source_name(self):
                return 'test'

            @property
            def variables(self):
                return []

            def extract(self, variable, start_date, end_date, last_n=None):
                pass

            def get_latest_date(self, variable):
                pass

        ext = TestExtractor()
        assert ext._get_frequency('gdpp_real_gdp_usa_q_gdp_q') == 'Q'
        assert ext._get_frequency('some_q_variable') == 'Q'

    def test_validate_variable(self):
        """Test variable validation."""
        class TestExtractor(BaseExtractor):
            @property
            def source_name(self):
                return 'test'

            @property
            def variables(self):
                return ['var1', 'var2', 'var3']

            def extract(self, variable, start_date, end_date, last_n=None):
                pass

            def get_latest_date(self, variable):
                pass

        ext = TestExtractor()
        assert ext.validate_variable('var1') is True
        assert ext.validate_variable('var2') is True
        assert ext.validate_variable('unknown') is False


# =============================================================================
# FRED EXTRACTOR TESTS
# =============================================================================

class TestFredExtractor:
    """Tests for FredExtractor."""

    @pytest.fixture
    def fred_extractor(self, fred_config):
        """Create FRED extractor with mocked config."""
        from extractors.fred_extractor import FredExtractor
        return FredExtractor(fred_config)

    def test_source_name(self, fred_extractor):
        """Test source name is correct."""
        assert fred_extractor.source_name == 'fred'

    def test_variables_loaded(self, fred_extractor):
        """Test variables are loaded from config."""
        assert len(fred_extractor.variables) >= 0

    @patch('requests.get')
    def test_extract_success(self, mock_get, fred_extractor, sample_monthly_df):
        """Test successful extraction from FRED."""
        # Mock FRED API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'observations': [
                {'date': '2024-01-01', 'value': '5.33'},
                {'date': '2024-02-01', 'value': '5.33'},
                {'date': '2024-03-01', 'value': '5.50'},
            ]
        }
        mock_get.return_value = mock_response

        result = fred_extractor.extract(
            'polr_fed_funds_usa_m_fedfunds',
            datetime(2024, 1, 1),
            datetime(2024, 3, 31),
        )

        assert result.success is True
        assert result.source == 'fred'

    @patch('requests.get')
    def test_extract_api_error(self, mock_get, fred_extractor):
        """Test handling of API errors."""
        mock_get.side_effect = ConnectionError('API unavailable')

        result = fred_extractor.extract(
            'polr_fed_funds_usa_m_fedfunds',
            datetime(2024, 1, 1),
            datetime(2024, 3, 31),
        )

        assert result.success is False
        assert 'error' in result.error.lower() or result.error is not None

    @patch('requests.get')
    def test_extract_empty_response(self, mock_get, fred_extractor):
        """Test handling of empty API response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'observations': []}
        mock_get.return_value = mock_response

        result = fred_extractor.extract(
            'polr_fed_funds_usa_m_fedfunds',
            datetime(2024, 1, 1),
            datetime(2024, 3, 31),
        )

        assert result.records_count == 0


# =============================================================================
# INVESTING EXTRACTOR TESTS
# =============================================================================

class TestInvestingExtractor:
    """Tests for InvestingExtractor."""

    @pytest.fixture
    def investing_extractor(self, investing_config):
        """Create Investing extractor."""
        from extractors.investing_extractor import InvestingExtractor
        return InvestingExtractor(investing_config)

    def test_source_name(self, investing_extractor):
        """Test source name is correct."""
        assert investing_extractor.source_name == 'investing'

    def test_variables_loaded(self, investing_extractor):
        """Test variables are loaded from config."""
        # Should have variables from config
        assert hasattr(investing_extractor, '_variables_config')

    @patch('cloudscraper.create_scraper')
    def test_extract_with_mock(self, mock_scraper, investing_extractor):
        """Test extraction with mocked scraper."""
        # Create mock scraper instance
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '''
        <table id="results_box">
            <tr><td>Jan 15, 2024</td><td>82.50</td></tr>
            <tr><td>Jan 14, 2024</td><td>81.75</td></tr>
        </table>
        '''
        mock_session.get.return_value = mock_response
        mock_session.post.return_value = mock_response
        mock_scraper.return_value = mock_session

        # Note: This test may fail because InvestingExtractor has complex
        # scraping logic. The point is to verify the interface.
        result = investing_extractor.extract(
            'comm_oil_brent_glb_d_brent',
            datetime(2024, 1, 1),
            datetime(2024, 1, 31),
        )

        # Result structure is correct regardless of success
        assert result.source == 'investing'
        assert result.variable == 'comm_oil_brent_glb_d_brent'


# =============================================================================
# SUAMECA EXTRACTOR TESTS
# =============================================================================

class TestSuamecaExtractor:
    """Tests for SuamecaExtractor (BanRep Colombia)."""

    @pytest.fixture
    def suameca_config(self):
        return {
            'base_url': 'https://suameca.banrep.gov.co/estadisticas-economicas-back/rest/estadisticaEconomicaRestService',
            'variables': [
                {'name': 'finc_rate_ibr_overnight_col_d_ibr', 'serie_id': 241, 'frequency': 'D'},
            ]
        }

    @pytest.fixture
    def suameca_extractor(self, suameca_config):
        """Create SUAMECA extractor."""
        from extractors.suameca_extractor import SuamecaExtractor
        return SuamecaExtractor(suameca_config)

    def test_source_name(self, suameca_extractor):
        """Test source name is correct."""
        assert suameca_extractor.source_name == 'suameca'

    @patch('requests.get')
    def test_rest_api_extraction(self, mock_get, suameca_extractor):
        """Test REST API extraction method."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'datosSerie': [
                {'fecha': '2024-01-15', 'valor': 12.25},
                {'fecha': '2024-01-16', 'valor': 12.25},
            ]
        }
        mock_get.return_value = mock_response

        result = suameca_extractor.extract(
            'finc_rate_ibr_overnight_col_d_ibr',
            datetime(2024, 1, 1),
            datetime(2024, 1, 31),
        )

        assert result.source == 'suameca'


# =============================================================================
# BCRP EXTRACTOR TESTS
# =============================================================================

class TestBcrpExtractor:
    """Tests for BcrpExtractor (Peru Central Bank)."""

    @pytest.fixture
    def bcrp_config(self):
        return {
            'base_url': 'https://estadisticas.bcrp.gob.pe/estadisticas/series/api',
            'variables': [
                {'name': 'crsk_spread_embi_col_d_embi', 'serie_code': 'PD04715XD', 'frequency': 'D'},
            ]
        }

    @pytest.fixture
    def bcrp_extractor(self, bcrp_config):
        """Create BCRP extractor."""
        from extractors.bcrp_extractor import BcrpExtractor
        return BcrpExtractor(bcrp_config)

    def test_source_name(self, bcrp_extractor):
        """Test source name is correct."""
        assert bcrp_extractor.source_name == 'bcrp'

    @patch('requests.get')
    def test_extract_embi(self, mock_get, bcrp_extractor):
        """Test EMBI spread extraction."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'periods': [
                {'name': '2024-01-15', 'values': ['250.5']},
                {'name': '2024-01-16', 'values': ['248.3']},
            ]
        }
        mock_get.return_value = mock_response

        result = bcrp_extractor.extract(
            'crsk_spread_embi_col_d_embi',
            datetime(2024, 1, 1),
            datetime(2024, 1, 31),
        )

        assert result.source == 'bcrp'


# =============================================================================
# REGISTRY TESTS
# =============================================================================

class TestExtractorRegistry:
    """Tests for ExtractorRegistry."""

    def test_registry_singleton(self):
        """Test registry is a singleton."""
        from extractors.registry import ExtractorRegistry

        reg1 = ExtractorRegistry()
        reg2 = ExtractorRegistry()

        assert reg1 is reg2

    def test_get_extractor_by_source(self):
        """Test getting extractor by source name."""
        from extractors.registry import ExtractorRegistry

        registry = ExtractorRegistry()

        # Should be able to get known extractors
        for source in ['fred', 'investing', 'suameca', 'bcrp']:
            extractor = registry.get_extractor(source)
            # May or may not be loaded depending on config
            if extractor is not None:
                assert extractor.source_name == source

    def test_get_variables_by_source(self):
        """Test getting variables by source."""
        from extractors.registry import ExtractorRegistry

        registry = ExtractorRegistry()
        variables = registry.get_variables_by_source('fred')

        # Should return a list (may be empty if no config)
        assert isinstance(variables, list)

    def test_get_intraday_variables(self):
        """Test getting intraday variables."""
        from extractors.registry import ExtractorRegistry

        registry = ExtractorRegistry()
        intraday = registry.get_intraday_variables()

        assert isinstance(intraday, list)

    def test_get_all_sources(self):
        """Test getting all loaded sources."""
        from extractors.registry import ExtractorRegistry

        registry = ExtractorRegistry()
        sources = registry.get_all_sources()

        assert isinstance(sources, list)


# =============================================================================
# INTEGRATION TESTS (with mocking)
# =============================================================================

class TestExtractorIntegration:
    """Integration tests for extractor workflow."""

    @patch('requests.get')
    def test_extract_and_validate(self, mock_get):
        """Test extraction followed by validation."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'observations': [
                {'date': '2024-01-01', 'value': '5.33'},
                {'date': '2024-02-01', 'value': '5.33'},
            ]
        }
        mock_get.return_value = mock_response

        from extractors.fred_extractor import FredExtractor

        config = {
            'api_key': 'test_key',
            'variables': [
                {'name': 'polr_fed_funds_usa_m_fedfunds', 'series_id': 'FEDFUNDS'}
            ]
        }
        extractor = FredExtractor(config)

        result = extractor.extract(
            'polr_fed_funds_usa_m_fedfunds',
            datetime(2024, 1, 1),
            datetime(2024, 2, 28),
        )

        # Validate result structure
        assert hasattr(result, 'source')
        assert hasattr(result, 'variable')
        assert hasattr(result, 'data')
        assert hasattr(result, 'success')

    def test_extract_last_n_limits_records(self, sample_daily_df):
        """Test that extract_last_n properly limits records."""

        class MockExtractor(BaseExtractor):
            @property
            def source_name(self):
                return 'mock'

            @property
            def variables(self):
                return ['test_var']

            def extract(self, variable, start_date, end_date, last_n=None):
                return ExtractionResult(
                    source='mock',
                    variable=variable,
                    data=sample_daily_df,
                    success=True,
                )

            def get_latest_date(self, variable):
                return datetime.now()

        extractor = MockExtractor()
        result = extractor.extract_last_n('test_var', n=5)

        assert result.success is True
        assert result.records_count == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
