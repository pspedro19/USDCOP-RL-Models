# USDCOP Trading Dashboard - Comprehensive Updates

## Overview
The Next.js frontend dashboard has been completely updated to provide comprehensive monitoring and visualization of all pipeline layers (L0-L6), API consumption tracking, and real-time data analysis.

## ✅ Completed Features

### 1. Pipeline Integration (L0-L6)
- **✅ Updated MinIO Service**: Connected to correct bucket names from pipeline configuration
  - L0: `00-raw-usdcop-marketdata` (Raw market data)
  - L1: `01-l1-ds-usdcop-standardize` (Standardized features)
  - L2: `02-l2-ds-usdcop-prepare` (Prepared data)
  - L3: `03-l3-ds-usdcop-feature` (Feature engineering)
  - L4: `04-l4-ds-usdcop-rlready` (RL-ready data)
  - L5: `05-l5-ds-usdcop-serving` (Model serving)
  - L6: `99-common-trading-reports` (Backtest results)

### 2. API Consumption Monitoring
- **✅ TwelveData API Monitoring**: Comprehensive tracking of 8 API keys
  - Real-time usage statistics (calls, costs, limits)
  - Rate limiting detection and management
  - Key rotation and load balancing
  - Historical usage charts and analytics
  - Cost analysis and projections
  - Success rate monitoring

### 3. Real-time Data Visualizations

#### L0 Raw Data Dashboard
- **✅ Real-time price feeds** from TwelveData
- **✅ Data completeness tracking** and quality metrics
- **✅ Source breakdown** (MT5 vs TwelveData)
- **✅ Spread analysis** and volume profiling
- **✅ Auto-refresh intervals** (10s, 30s, 1min, 5min)

#### L1 Feature Statistics
- **✅ Data quality reports** with pass/warn/fail indicators
- **✅ Feature processing metrics** and completion rates
- **✅ Session breakdown** (Premium, London, Afternoon)
- **✅ Anomaly detection** with outlier classification
- **✅ Processing performance** metrics (records/sec, latency)
- **✅ Statistical distributions** for key features

#### L3 Correlation Analysis
- **✅ Interactive correlation matrix** with filtering
- **✅ Feature importance ranking** with stability scores
- **✅ Multicollinearity detection** (VIF scores)
- **✅ Statistical tests** (normality, stationarity)
- **✅ Feature selection summary** and recommendations

#### L4 RL-Ready Data
- **✅ Clip rate analysis** with severity levels (LOW/MEDIUM/HIGH/CRITICAL)
- **✅ Episode statistics** and action space analysis
- **✅ Reward distribution** histograms
- **✅ Data validation results** for RL compatibility
- **✅ Preprocessing configuration** display

#### L5 Model Serving
- **✅ Real-time predictions** with confidence scores
- **✅ Latency metrics** (avg, p95, p99)
- **✅ Model performance** tracking (accuracy, Sharpe ratio)
- **✅ Serving health status** and error monitoring

#### L6 Backtest Results
- **✅ Trade ledger** with detailed transaction history
- **✅ Equity curve** visualization
- **✅ Performance metrics** (win rate, profit factor, drawdown)
- **✅ Monthly returns** breakdown

### 4. System Monitoring

#### Pipeline Health Monitor
- **✅ Real-time health status** for all layers (L0-L6)
- **✅ Data freshness tracking** with alerting
- **✅ Error detection** and status indicators
- **✅ Overall system health** dashboard

#### API Usage Statistics Panel
- **✅ Comprehensive usage tracking** across all API keys
- **✅ Cost monitoring** and budget alerts
- **✅ Success rate** and latency tracking
- **✅ Peak usage analysis** and optimization insights

## 🏗️ Technical Implementation

### New Services Created
1. **`pipeline.ts`** - Enhanced MinIO integration with correct bucket mapping
2. **`api-monitor.ts`** - Comprehensive TwelveData API monitoring service
3. **`twelvedata.ts`** - Updated with monitoring integration

### New Components Created
1. **`L0RawDataDashboard.tsx`** - Real-time market data visualization
2. **`L1FeatureStats.tsx`** - Feature quality and processing metrics
3. **`L3CorrelationMatrix.tsx`** - Correlation analysis and feature importance
4. **`L4RLReadyData.tsx`** - RL data validation and clipping analysis
5. **`L5ModelDashboard.tsx`** - Model serving performance monitoring
6. **`L6BacktestResults.tsx`** - Backtest results and trade analysis
7. **`PipelineHealthMonitor.tsx`** - System-wide health monitoring
8. **`APIUsagePanel.tsx`** - API consumption analytics

### Dashboard Navigation
- **Enhanced sidebar** with categorized navigation:
  - **Trading**: Real-time charts, signals, risk management
  - **Pipeline**: L0-L6 layer-specific dashboards
  - **System**: Health monitoring, API usage, legacy pipeline

## 🚀 Key Features

### Error Handling & Resilience
- **Graceful degradation** when pipeline data is unavailable
- **Automatic retries** and fallback mechanisms
- **Real-time error monitoring** with user-friendly alerts
- **Connection status indicators** and health checks

### Data Formats Support
- **Parquet files** for structured pipeline data
- **JSON format** for metadata and reports
- **CSV files** for legacy data compatibility
- **Real-time streaming** via WebSocket connections

### Interactive Visualizations
- **Filterable charts** with time range selection
- **Drill-down capabilities** for detailed analysis
- **Real-time updates** with configurable refresh rates
- **Responsive design** for different screen sizes

### Performance Optimization
- **Lazy loading** of dashboard components
- **Efficient data fetching** with caching
- **Minimal re-renders** using React optimization
- **Background data updates** without UI blocking

## 🔧 Configuration

### Environment Variables Required
```bash
# MinIO Connection
NEXT_PUBLIC_S3_ENDPOINT=http://localhost:9000
NEXT_PUBLIC_S3_ACCESS_KEY=airflow
NEXT_PUBLIC_S3_SECRET_KEY=airflow

# TwelveData API Keys (up to 8 keys for load balancing)
NEXT_PUBLIC_TWELVEDATA_API_KEY_1=your_key_here
# ... up to _8
```

### Bucket Mapping
The dashboard automatically connects to the correct pipeline buckets:
- Uses configuration from `pipeline_dataflow.yml`
- Maps DAG outputs to appropriate bucket names
- Handles both new and legacy data formats

## 🎯 Usage

### Starting the Dashboard
```bash
cd usdcop-trading-dashboard
npm install
npm run dev
```

### Navigating the Interface
1. **Trading Section**: Focus on live trading data and signals
2. **Pipeline Section**: Monitor each processing layer (L0-L6)
3. **System Section**: Track overall health and API usage

### Monitoring Capabilities
- **Real-time alerts** for system issues
- **Performance metrics** for all pipeline stages
- **Cost tracking** for API consumption
- **Quality gates** for data validation

## 🔍 Data Flow Integration

The dashboard integrates with your existing pipeline:
1. **L0**: Monitors raw data ingestion from MT5 and TwelveData
2. **L1**: Tracks feature standardization and quality metrics
3. **L2**: Analyzes data preparation and cleaning results
4. **L3**: Visualizes feature engineering and correlations
5. **L4**: Validates RL-ready data with clip rate analysis
6. **L5**: Monitors model serving performance and predictions
7. **L6**: Displays backtest results and trading performance

## 🚨 Monitoring & Alerts

### Health Monitoring
- **Pipeline status**: Green/Yellow/Red indicators
- **Data freshness**: Alerts for stale data
- **Processing errors**: Real-time error tracking
- **API limits**: Rate limiting notifications

### Performance Tracking
- **Latency metrics**: Response times for all services
- **Throughput monitoring**: Data processing rates
- **Resource usage**: Memory and CPU tracking
- **Success rates**: Error rate monitoring

## 📊 Available Metrics

### Trading Metrics
- Real-time USD/COP prices and spreads
- Volume analysis and market microstructure
- Trading signals and model predictions
- Backtest results and performance analytics

### Technical Metrics
- Feature correlation matrices and importance
- Data quality scores and validation results
- Model serving latency and throughput
- API consumption and cost analysis

### System Metrics
- Pipeline health and data freshness
- Processing performance and error rates
- Resource utilization and capacity planning
- Integration status and connectivity

## 🎉 Benefits

1. **Complete Visibility**: Monitor every aspect of your trading pipeline
2. **Real-time Monitoring**: Instant alerts and live data updates
3. **Cost Optimization**: Track and optimize API usage costs
4. **Quality Assurance**: Comprehensive data validation and quality gates
5. **Performance Optimization**: Identify bottlenecks and optimize processing
6. **Risk Management**: Monitor model performance and trading risks
7. **Historical Analysis**: Track trends and performance over time
8. **Operational Excellence**: Proactive monitoring and maintenance

The dashboard now provides comprehensive monitoring of your entire USDCOP trading pipeline, from raw data ingestion through model serving and backtesting, with full integration to your existing MinIO infrastructure and TwelveData API services.