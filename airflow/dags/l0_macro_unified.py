"""
DAG: l0_macro_unified
=====================
Unified Daily Macro Indicators DAG - ALL 41+ Variables

Purpose:
    Single DAG that fetches ALL macro variables from multiple sources using
    the Strategy Pattern with centralized configuration.

Sources:
    - FRED API (14 variables) - US economic indicators
    - TwelveData API (4 variables) - FX and commodities
    - BanRep SUAMECA via Selenium (6 variables) - Colombian monetary data
    - Investing.com via Cloudscraper (5 variables) - Market indices
    - Fedesarrollo (2 variables) - CCI, ICI confidence indices
    - DANE (2 variables) - Exports, Imports trade balance
    - BCRP Peru (1 variable) - EMBI spread

Schedule:
    50 12 * * 1-5 (7:50am COT, Mon-Fri, pre-session)

Data Flow:
    Multiple Sources -> macro_indicators_daily (UPSERT) -> cleanup -> ffill -> readiness report

Architecture:
    Uses Strategy Pattern with MacroExtractorFactory for interchangeable extractors.
    Configuration loaded from config/l0_macro_sources.yaml (SSOT).

Contract: CTR-L0-DAG-001
Version: 2.0.0 (Refactored)
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator

# Service imports
from services.macro_extraction_service import extract_all_sources
from services.macro_merge_service import (
    merge_and_upsert,
    apply_forward_fill,
    generate_readiness_report,
)
from services.macro_cleanup_service import cleanup_non_trading_days

# DAG registry
from contracts.dag_registry import L0_MACRO_DAILY

# =============================================================================
# DAG CONFIGURATION
# =============================================================================

DAG_ID = L0_MACRO_DAILY

default_args = {
    'owner': 'usdcop-trading',
    'depends_on_past': False,
    'email_on_failure': True,
    'email': ['trading-alerts@example.com'],
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# =============================================================================
# DAG DEFINITION
# =============================================================================

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description='Unified macro DAG - ALL 41+ variables from all sources (Strategy Pattern)',
    schedule_interval='50 12 * * 1-5',  # 7:50am COT Mon-Fri
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=['l0', 'macro', 'unified', 'daily', 'strategy-pattern'],
) as dag:

    # Start marker
    start = EmptyOperator(task_id='start')

    # ==========================================================================
    # EXTRACTION TASK
    # Uses MacroExtractionService to extract from all enabled sources
    # ==========================================================================
    extract_all = PythonOperator(
        task_id='extract_all_sources',
        python_callable=extract_all_sources,
        execution_timeout=timedelta(minutes=30),
        doc_md="""
        ### Extract All Sources

        Extracts data from all configured sources using Strategy Pattern:
        - FRED API (14 indicators)
        - TwelveData API (4 indicators)
        - Investing.com (5 indicators)
        - BanRep SUAMECA (6 indicators)
        - BCRP Peru (1 indicator - EMBI)
        - Fedesarrollo (2 indicators)
        - DANE (2 indicators)

        Configuration loaded from: config/l0_macro_sources.yaml
        """,
    )

    # ==========================================================================
    # MERGE AND UPSERT TASK
    # Combines all source data and writes to database
    # ==========================================================================
    merge = PythonOperator(
        task_id='merge_and_upsert',
        python_callable=merge_and_upsert,
        execution_timeout=timedelta(minutes=5),
        doc_md="""
        ### Merge and Upsert

        Merges data from all sources and upserts to macro_indicators_daily table.
        Tracks release_date for each indicator based on source publication timing.
        """,
    )

    # ==========================================================================
    # CLEANUP TASK
    # Removes weekends and holidays (COL + USA)
    # ==========================================================================
    cleanup = PythonOperator(
        task_id='cleanup_non_trading_days',
        python_callable=cleanup_non_trading_days,
        execution_timeout=timedelta(minutes=5),
        doc_md="""
        ### Cleanup Non-Trading Days

        Removes records for weekends and holidays from macro_indicators_daily.
        Checks both US and Colombian holidays.
        """,
    )

    # ==========================================================================
    # FORWARD FILL TASK
    # Applies bounded forward-fill respecting publication schedules
    # ==========================================================================
    ffill = PythonOperator(
        task_id='apply_forward_fill',
        python_callable=apply_forward_fill,
        execution_timeout=timedelta(minutes=5),
        doc_md="""
        ### Forward Fill

        Applies bounded forward-fill for each indicator respecting max days:
        - Daily indicators: max 5 days
        - Weekly indicators: max 10 days
        - Monthly indicators: max 35 days
        - Quarterly indicators: max 95 days
        """,
    )

    # ==========================================================================
    # READINESS REPORT TASK
    # Single source of truth for inference readiness
    # ==========================================================================
    readiness = PythonOperator(
        task_id='generate_readiness_report',
        python_callable=generate_readiness_report,
        execution_timeout=timedelta(minutes=2),
        doc_md="""
        ### Generate Readiness Report

        Creates DailyDataReadinessReport with:
        - Fresh/FFilled/Stale/Missing indicator counts
        - Readiness score (0-100%)
        - Blocking issues for inference

        Pushes 'is_ready_for_inference' to XCom for downstream DAGs.
        """,
    )

    # End marker
    end = EmptyOperator(task_id='end')

    # ==========================================================================
    # TASK DEPENDENCIES
    # ==========================================================================
    start >> extract_all >> merge >> cleanup >> ffill >> readiness >> end
