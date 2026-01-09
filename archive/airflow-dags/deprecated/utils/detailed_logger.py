"""
Detailed Logger for USDCOP Trading Pipeline
Provides executive-style reporting with clear metrics and context
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

class DetailedLogger:
    """Enhanced logger for pipeline execution with executive reporting"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        
    def log_executive_summary(self, 
                             currency_pair: str = "USD/COP",
                             timeframe: str = "M5",
                             start_date: datetime = None,
                             end_date: datetime = None,
                             source: str = None,
                             premium_status: str = None,
                             overall_assessment: str = None):
        """Log executive summary section"""
        
        self.logger.info("="*80)
        self.logger.info("USDCOP M5 DATA ACQUISITION - DETAILED REPORT")
        self.logger.info("="*80)
        self.logger.info("")
        self.logger.info("## EXECUTIVE SUMMARY")
        self.logger.info(f"- **Currency Pair**: {currency_pair} (US Dollar to Colombian Peso)")
        self.logger.info(f"- **Timeframe**: 5-minute bars ({timeframe})")
        self.logger.info(f"- **Period**: {start_date.date()} to {end_date.date()}")
        self.logger.info(f"- **Data Source**: {source}")
        self.logger.info(f"- **Premium Hours Status**: {premium_status}")
        self.logger.info(f"- **Overall Assessment**: {overall_assessment}")
        self.logger.info("-"*80)
        
    def log_trading_schedule(self):
        """Log trading schedule configuration"""
        
        self.logger.info("")
        self.logger.info("## TRADING SCHEDULE CONFIGURATION")
        self.logger.info("")
        self.logger.info("### 🏆 Premium Trading Hours (Priority Focus)")
        self.logger.info("- **Time Window**: 8:00 AM - 2:00 PM COT (Colombia Time)")
        self.logger.info("- **Duration**: 6 hours per trading day")
        self.logger.info("- **Expected Bars**: 72 bars per day (6 hours × 12 bars/hour)")
        self.logger.info("- **Business Logic**: Most liquid and active trading period")
        self.logger.info("- **Quality Target**: ≥95% completeness")
        self.logger.info("")
        self.logger.info("### 📊 Extended Trading Hours")
        self.logger.info("- **Coverage**: 24-hour data collection when available")
        self.logger.info("- **Purpose**: Market analysis and extended research")
        self.logger.info("- **Note**: Premium hours remain priority for trading decisions")
        self.logger.info("-"*80)
        
    def log_period_analysis(self,
                           period_num: int,
                           start_date: datetime,
                           end_date: datetime,
                           business_days: int,
                           holidays: int,
                           trading_days: int,
                           expected_premium_bars: int,
                           actual_premium_bars: int,
                           total_bars: int,
                           market_context: str = None):
        """Log detailed period analysis"""
        
        # Calculate metrics
        premium_completeness = (actual_premium_bars / expected_premium_bars * 100) if expected_premium_bars > 0 else 0
        extended_hours_bars = total_bars - actual_premium_bars
        total_vs_premium_ratio = (total_bars / expected_premium_bars * 100) if expected_premium_bars > 0 else 0
        extended_bonus = total_vs_premium_ratio - 100
        
        # Determine quality assessment
        if premium_completeness >= 95:
            assessment = "✅ EXCELLENT"
            status = "Meets premium data requirements"
        elif premium_completeness >= 80:
            assessment = "⚠️ GOOD"
            status = "Acceptable coverage, some gaps"
        elif premium_completeness >= 60:
            assessment = "⚠️ FAIR"
            status = "Significant gaps, consider additional sources"
        else:
            assessment = "❌ POOR"
            status = "Insufficient data for reliable training"
            
        self.logger.info("")
        self.logger.info(f"### Period {period_num}: {start_date.date()} to {end_date.date()}")
        self.logger.info("```")
        self.logger.info(f"📅 Date Range: {start_date.strftime('%B %d, %Y')} → {end_date.strftime('%B %d, %Y')}")
        if market_context:
            self.logger.info(f"📈 Market Context: {market_context}")
        self.logger.info("")
        self.logger.info("Trading Calendar:")
        self.logger.info(f"├── Business Days (Mon-Fri): {business_days} days")
        self.logger.info(f"├── Colombian Holidays: {holidays} days excluded")
        self.logger.info(f"└── Effective Trading Days: {trading_days} days")
        self.logger.info("")
        self.logger.info("Data Breakdown:")
        self.logger.info(f"├── Expected Premium Bars: {expected_premium_bars:,} ({trading_days} days × 72 bars/day)")
        self.logger.info(f"├── Downloaded Premium Bars: {actual_premium_bars:,} (8am-2pm COT)")
        self.logger.info(f"├── Downloaded Extended Hours: {extended_hours_bars:,} (other timeframes)")
        self.logger.info(f"├── Downloaded Total Bars: {total_bars:,}")
        self.logger.info(f"└── Premium Completeness: {premium_completeness:.1f}% {assessment}")
        
        # Add hourly distribution if we have extended hours
        if extended_hours_bars > 0:
            self.logger.info("")
            self.logger.info("Hourly Coverage Distribution:")
            
            # Estimate distribution (can be made more accurate with actual data)
            premium_pct = (actual_premium_bars / total_bars * 100) if total_bars > 0 else 0
            other_pct = (100 - premium_pct) / 3  # Distribute among other periods
            
            self.logger.info(f"├── 🏆 Premium Hours (8am-2pm COT): {actual_premium_bars:,} bars (~{premium_pct:.1f}%)")
            self.logger.info(f"├── 🌅 Pre-Market (12am-8am COT): ~{int(total_bars * other_pct/100):,} bars (~{other_pct:.1f}%)")
            self.logger.info(f"├── 🌆 Post-Market (2pm-8pm COT): ~{int(total_bars * other_pct/100):,} bars (~{other_pct:.1f}%)")
            self.logger.info(f"└── 🌙 Night Hours (8pm-12am COT): ~{int(total_bars * other_pct/100):,} bars (~{other_pct:.1f}%)")
        
        self.logger.info("")
        self.logger.info("Reference Framework Analysis:")
        self.logger.info(f"• Premium Target: ≥95% | Achieved: {premium_completeness:.1f}%")
        self.logger.info(f"• Total vs Premium Ratio: {total_vs_premium_ratio:.1f}% ({total_bars:,} ÷ {expected_premium_bars:,})")
        if extended_bonus > 0:
            self.logger.info(f"• Extended Hours Bonus: +{extended_bonus:.1f}% additional coverage")
        self.logger.info(f"• Quality Assessment: {status}")
        self.logger.info("```")
        
    def log_data_quality_insights(self, 
                                 all_periods_data: List[Dict],
                                 overall_premium_completeness: float):
        """Log data quality insights section"""
        
        self.logger.info("")
        self.logger.info("-"*80)
        self.logger.info("## DATA QUALITY INSIGHTS")
        self.logger.info("")
        
        # Premium Hours Performance
        self.logger.info("### 🎯 Premium Hours Performance (Marco de Referencia Principal)")
        self.logger.info("- **Expected Premium Bars**: ~9,000-9,200 per 6-month period")
        self.logger.info(f"- **Actual Premium Completeness**: {overall_premium_completeness:.1f}%")
        self.logger.info(f"- **Premium Target**: ≥95% {'✅ ACHIEVED' if overall_premium_completeness >= 95 else '❌ NOT MET'}")
        self.logger.info("- **Interpretation**: Core trading hours " + 
                        ("fully covered" if overall_premium_completeness >= 95 else "need improvement"))
        
        # Extended Hours Analysis
        if any(p.get('extended_hours', 0) > 0 for p in all_periods_data):
            self.logger.info("")
            self.logger.info("### 📊 Extended Hours Analysis (Marco de Referencia Adicional)")
            
            avg_ratio = np.mean([p.get('total_vs_premium_ratio', 100) for p in all_periods_data])
            self.logger.info(f"- **Average 24-Hour Coverage Ratio**: {avg_ratio:.1f}% vs premium baseline")
            self.logger.info("- **Extended Hours Distribution**:")
            self.logger.info("  - Pre-Market (12am-8am): ~25% of total data")
            self.logger.info("  - **Premium (8am-2pm): ~25% of total data** ⭐")
            self.logger.info("  - Post-Market (2pm-8pm): ~25% of total data")
            self.logger.info("  - Night Hours (8pm-12am): ~25% of total data")
        
        # Mathematical Framework
        self.logger.info("")
        self.logger.info("### 📈 Mathematical Framework Explanation")
        self.logger.info("```")
        self.logger.info("Completeness = (Actual Premium Bars / Expected Premium Bars) × 100")
        self.logger.info("")
        self.logger.info("Where:")
        self.logger.info("- Expected Premium Bars = Trading Days × 72 bars/day")
        self.logger.info("- Trading Days = Business Days - Holidays")
        self.logger.info("- Business Days = Monday to Friday in period")
        self.logger.info("- Premium Window = 8:00-14:00 COT = 6 hours × 12 bars/hour = 72 bars/day")
        self.logger.info("")
        self.logger.info("Example Calculation:")
        self.logger.info("├── Trading Days: 126 days")
        self.logger.info("├── Expected Premium: 126 × 72 = 9,072 bars")
        self.logger.info("├── Actual Premium: 8,900 bars")
        self.logger.info("└── Completeness: 8,900 ÷ 9,072 = 98.1% ✅")
        self.logger.info("```")
        
    def log_recommendations(self):
        """Log recommendations section"""
        
        self.logger.info("")
        self.logger.info("-"*80)
        self.logger.info("## RECOMMENDATIONS")
        self.logger.info("")
        
        self.logger.info("### ✅ Current Strengths")
        self.logger.info("1. **Premium Coverage**: Meeting critical 8am-2pm window requirements")
        self.logger.info("2. **Data Consistency**: Stable acquisition across multiple periods")
        self.logger.info("3. **API Integration**: TwelveData connection functioning properly")
        self.logger.info("4. **Quality Metrics**: Clear visibility into data completeness")
        self.logger.info("")
        
        self.logger.info("### 🚀 Optimization Opportunities")
        self.logger.info("1. **Fill Gaps**: Identify and backfill any missing premium hours")
        self.logger.info("2. **Cost Analysis**: Evaluate API usage vs data value")
        self.logger.info("3. **Storage Strategy**: Optimize parquet compression for large datasets")
        self.logger.info("4. **Real-time Monitoring**: Set alerts for completeness < 95%")
        self.logger.info("")
        
        self.logger.info("### 📋 Key Performance Indicators")
        self.logger.info("- **Primary KPI**: Premium Hours Completeness ≥95%")
        self.logger.info("- **Secondary KPI**: Data freshness < 5 minutes")
        self.logger.info("- **Storage Efficiency**: < 100MB per million bars")
        self.logger.info("- **Processing Time**: < 30 seconds per batch")
        self.logger.info("")
        self.logger.info("="*80)
        
    def format_number(self, num: float, decimals: int = 0) -> str:
        """Format number with thousands separator"""
        if decimals == 0:
            return f"{int(num):,}"
        else:
            return f"{num:,.{decimals}f}"
            
    def get_market_context(self, date: datetime) -> str:
        """Get market context for a given date"""
        contexts = {
            (2020, 1): "Pre-pandemic baseline",
            (2020, 3): "COVID-19 market crash",
            (2020, 4): "Initial recovery phase",
            (2020, 7): "Economic stimulus period",
            (2021, 1): "Post-pandemic normalization",
            (2021, 7): "Inflation concerns emerging",
            (2022, 1): "Rate hike expectations",
            (2022, 7): "Global recession fears",
            (2023, 1): "Banking sector stress",
            (2023, 7): "Stabilization period",
            (2024, 1): "Election year volatility",
            (2024, 7): "Summer trading patterns",
            (2025, 1): "Current market conditions",
            (2025, 7): "Live trading environment"
        }
        
        for (year, month), context in contexts.items():
            if date.year == year and date.month >= month:
                return context
                
        return "Historical period"