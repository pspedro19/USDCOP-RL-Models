/**
 * Pipeline API Endpoints Documentation
 * GET /api/pipeline/endpoints
 *
 * Returns comprehensive documentation of all L0-L6 pipeline API endpoints
 */

import { NextResponse } from 'next/server';

export async function GET() {
  const endpoints = {
    version: '1.0.0',
    timestamp: new Date().toISOString(),
    baseUrl: '/api/pipeline',
    description: 'Complete USD/COP RL Trading Pipeline API - All endpoints use real data from PostgreSQL, MinIO, and TwelveData',

    layers: {
      L0: {
        description: 'Raw market data layer - OHLC bars from TwelveData (92,936 records: 2020-01-02 to 2025-10-10)',
        bucket: '00-raw-usdcop-marketdata',
        postgres_table: 'market_data',
        endpoints: [
          {
            path: '/api/pipeline/l0/raw-data',
            method: 'GET',
            description: 'Get raw OHLC market data with multi-source support',
            parameters: {
              start_date: 'ISO date string (e.g., "2024-01-01")',
              end_date: 'ISO date string (e.g., "2024-12-31")',
              limit: 'Max records (default: 1000, max: 10000)',
              offset: 'Pagination offset (default: 0)',
              source: 'Data source: postgres | minio | twelvedata | all (default: postgres)',
            },
            example: '/api/pipeline/l0/raw-data?start_date=2024-01-01&end_date=2024-12-31&limit=100',
          },
          {
            path: '/api/pipeline/l0/statistics',
            method: 'GET',
            description: 'Get aggregate statistics on L0 data (completeness, quality, distribution)',
            parameters: {
              start_date: 'Optional start date filter',
              end_date: 'Optional end date filter',
            },
            example: '/api/pipeline/l0/statistics?start_date=2024-01-01',
          },
          {
            path: '/api/pipeline/l0',
            method: 'GET',
            description: 'Legacy L0 endpoint with MinIO fallback and TwelveData support',
            example: '/api/pipeline/l0?startDate=2024-01-01&endDate=2024-12-31',
          },
        ],
      },

      L1: {
        description: 'Standardized episodes - 60-bar episodes passing quality gates (929 accepted episodes)',
        bucket: '01-l1-ds-usdcop-standardize',
        format: 'Parquet files partitioned by episode_date',
        endpoints: [
          {
            path: '/api/pipeline/l1/quality-report',
            method: 'GET',
            description: 'Get L1 quality gate reports and episode acceptance metrics',
            parameters: {
              run_id: 'Specific pipeline run ID',
              start_date: 'Filter reports by date',
              end_date: 'Filter reports by date',
            },
            example: '/api/pipeline/l1/quality-report',
          },
          {
            path: '/api/pipeline/l1/episodes',
            method: 'GET',
            description: 'List or retrieve L1 standardized episodes',
            parameters: {
              episode_id: 'Specific episode ID',
              limit: 'Max episodes (default: 100, max: 1000)',
              start_date: 'Filter by episode date',
            },
            example: '/api/pipeline/l1/episodes?limit=50',
          },
        ],
      },

      L2: {
        description: 'Prepared data - Deseasonalized features with HoD baselines and return series',
        bucket: '02-l2-ds-usdcop-prep',
        features: ['HoD baselines', 'Deseasonalized OHLC', 'Return series', 'Winsorization'],
        endpoints: [
          {
            path: '/api/pipeline/l2/prepared-data',
            method: 'GET',
            description: 'Get L2 prepared/deseasonalized data and HoD baselines',
            parameters: {
              episode_id: 'Specific episode ID',
              limit: 'Max episodes (default: 100)',
            },
            example: '/api/pipeline/l2/prepared-data?limit=50',
          },
        ],
      },

      L3: {
        description: 'Engineered features - 17 features per episode with IC compliance checks',
        bucket: '03-l3-ds-usdcop-features',
        featureCount: 17,
        features: [
          'Price momentum indicators',
          'Volatility measures',
          'Volume features',
          'Technical indicators',
          'Market microstructure features',
        ],
        endpoints: [
          {
            path: '/api/pipeline/l3/features',
            method: 'GET',
            description: 'Get L3 engineered features with IC analysis',
            parameters: {
              episode_id: 'Specific episode ID',
              limit: 'Max episodes (default: 100)',
            },
            example: '/api/pipeline/l3/features?episode_id=20240101',
          },
        ],
      },

      L4: {
        description: 'RL-ready datasets - Train/Val/Test splits for reinforcement learning',
        bucket: '04-l4-ds-usdcop-rlready',
        splits: {
          train: '557 episodes (60%)',
          val: '186 episodes (20%)',
          test: '186 episodes (20%)',
          total: '929 episodes',
        },
        endpoints: [
          {
            path: '/api/pipeline/l4/dataset',
            method: 'GET',
            description: 'Get RL-ready dataset with train/val/test splits',
            parameters: {
              split: 'Data split: train | val | test',
              episode_id: 'Specific episode ID (requires split parameter)',
            },
            example: '/api/pipeline/l4/dataset?split=test',
          },
        ],
      },

      L5: {
        description: 'Model serving - Trained RL models with ONNX export and inference profiles',
        bucket: '05-l5-ds-usdcop-serving',
        artifacts: ['ONNX models', 'Checkpoints', 'Training metrics', 'Latency profiles'],
        endpoints: [
          {
            path: '/api/pipeline/l5/models',
            method: 'GET',
            description: 'Get trained models and serving artifacts',
            parameters: {
              model_id: 'Specific model ID',
              format: 'Model format: onnx | checkpoint',
            },
            example: '/api/pipeline/l5/models',
          },
        ],
      },

      L6: {
        description: 'Backtest results - Hedge-fund grade performance metrics and trade analysis',
        bucket: 'usdcop-l6-backtest',
        metrics: [
          'Sharpe Ratio',
          'Sortino Ratio',
          'Calmar Ratio',
          'Maximum Drawdown',
          'Win Rate',
          'Profit Factor',
          'Trade ledger',
          'Daily returns',
        ],
        endpoints: [
          {
            path: '/api/pipeline/l6/backtest-results',
            method: 'GET',
            description: 'Get comprehensive backtest results with hedge-fund metrics',
            parameters: {
              run_id: 'Specific backtest run ID (defaults to latest)',
              split: 'test | val (defaults to both)',
              metric: 'Specific metric name (e.g., "sharpe_ratio")',
            },
            example: '/api/pipeline/l6/backtest-results?split=test',
          },
          {
            path: '/api/backtest/results',
            method: 'GET',
            description: 'Legacy backtest results endpoint (maintained for compatibility)',
            example: '/api/backtest/results',
          },
        ],
      },
    },

    dataFlow: [
      'L0: Raw OHLC (PostgreSQL + MinIO + TwelveData) → 92,936 records',
      'L1: Quality Gates → 929 accepted episodes (60 bars each)',
      'L2: Deseasonalization + HoD Baselines → Prepared episodes',
      'L3: Feature Engineering → 17 features per episode',
      'L4: RL Dataset Creation → 557 train / 186 val / 186 test',
      'L5: Model Training → ONNX export + Inference profiles',
      'L6: Backtesting → Hedge-fund grade metrics (Sharpe, Sortino, Calmar, etc.)',
    ],

    dataSources: {
      postgresql: {
        host: 'localhost:5432',
        database: 'usdcop_trading',
        table: 'market_data',
        records: 92936,
        dateRange: '2020-01-02 to 2025-10-10',
        type: 'TimescaleDB',
      },
      minio: {
        endpoint: 'localhost:9000',
        buckets: [
          '00-raw-usdcop-marketdata',
          '01-l1-ds-usdcop-standardize',
          '02-l2-ds-usdcop-prep',
          '03-l3-ds-usdcop-features',
          '04-l4-ds-usdcop-rlready',
          '05-l5-ds-usdcop-serving',
          'usdcop-l6-backtest',
        ],
      },
      twelvedata: {
        api: 'TwelveData Forex API',
        symbol: 'USD/COP',
        interval: '5min',
        description: 'Real-time market data source',
      },
    },

    technicalDetails: {
      barFrequency: '5 minutes',
      tradingHours: '8:00 AM - 12:55 PM COT (UTC-5)',
      episodeLength: '60 bars (5 hours of trading)',
      marketDays: 'Monday - Friday',
      currency: 'Colombian Peso (COP)',
      baseCurrency: 'US Dollar (USD)',
    },

    notes: [
      'All endpoints return JSON responses',
      'Date parameters should be in ISO 8601 format (YYYY-MM-DD)',
      'Pagination is supported on list endpoints via limit/offset',
      'All data is sourced from actual pipeline outputs (no hardcoded/mock data)',
      'MinIO buckets may not exist until pipelines have run',
      'PostgreSQL is the primary source for L0 data',
      'TwelveData API is used for real-time/live data',
    ],
  };

  return NextResponse.json(endpoints, {
    headers: {
      'Content-Type': 'application/json',
      'Cache-Control': 'public, max-age=3600', // Cache for 1 hour
    },
  });
}
