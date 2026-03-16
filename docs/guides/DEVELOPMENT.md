# USDCOP Trading System - Development Guide

**Version:** 2.0.0
**Date:** October 22, 2025
**Audience:** Software Engineers, Data Scientists, DevOps

---

## Table of Contents

1. [Development Environment Setup](#development-environment-setup)
2. [Project Structure](#project-structure)
3. [Adding New Features](#adding-new-features)
4. [Testing Guidelines](#testing-guidelines)
5. [Coding Standards](#coding-standards)
6. [Database Development](#database-development)
7. [API Development](#api-development)
8. [Dashboard Development](#dashboard-development)
9. [Pipeline Development](#pipeline-development)
10. [Git Workflow](#git-workflow)
11. [Pull Request Checklist](#pull-request-checklist)

---

## Development Environment Setup

### Prerequisites

**Required Software:**
- **Docker** 20.10+ and **Docker Compose** 2.0+
- **Python** 3.11+
- **Node.js** 18.0+ and **npm** 9.0+
- **Git** 2.30+
- **PostgreSQL Client** (psql) for database access
- **Code Editor**: VS Code (recommended) or PyCharm

**Recommended System:**
- CPU: 4+ cores
- RAM: 16GB+ (8GB minimum)
- Disk: 50GB+ free space
- OS: Linux (Ubuntu 22.04), macOS, or Windows WSL2

---

### Step 1: Clone Repository

```bash
# Clone the repository
git clone https://github.com/your-org/USDCOP-RL-Models.git
cd USDCOP-RL-Models

# Create development branch
git checkout -b dev/your-feature-name
```

---

### Step 2: Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

**Development `.env` Configuration:**

```bash
# PostgreSQL
POSTGRES_USER=admin
POSTGRES_PASSWORD=admin123
POSTGRES_DB=usdcop_trading
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=redis123

# MinIO (Local)
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123
MINIO_ENDPOINT=localhost:9000

# TwelveData API Keys (Get from https://twelvedata.com)
TWELVEDATA_API_KEY_1=your_key_here

# API Key Groups (for L0 pipeline)
API_KEY_G1_1=your_key_1
API_KEY_G1_2=your_key_2
# ... add more keys as needed

# Airflow
AIRFLOW_USER=admin
AIRFLOW_PASSWORD=admin123

# Development Flags
NODE_ENV=development
LOG_LEVEL=DEBUG
```

---

### Step 3: Start Infrastructure Services

```bash
# Start only infrastructure (no application services yet)
docker compose up -d postgres redis minio

# Wait for services to be healthy
docker ps

# Initialize MinIO buckets
docker compose up minio-init

# Verify MinIO buckets
docker exec usdcop-minio-init mc ls minio
```

---

### Step 4: Initialize Database

```bash
# Run initialization scripts
docker exec -i usdcop-postgres-timescale psql -U admin -d usdcop_trading < init-scripts/01-schema.sql
docker exec -i usdcop-postgres-timescale psql -U admin -d usdcop_trading < init-scripts/02-timescaledb.sql

# Verify tables created
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c "\dt"

# Seed development data (optional)
python scripts/seed_development_data.py
```

---

### Step 5: Python Development Setup

```bash
# Create Python virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development tools

# Install pre-commit hooks
pre-commit install
```

**requirements-dev.txt:**

```
# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
pytest-mock==3.12.0

# Linting
pylint==3.0.3
black==23.12.1
isort==5.13.2
mypy==1.8.0

# Type stubs
types-redis==4.6.0.10
types-requests==2.31.0.10

# Development tools
ipython==8.19.0
ipdb==0.13.13
```

---

### Step 6: Dashboard Development Setup

```bash
cd usdcop-trading-dashboard

# Install dependencies
npm install

# Start development server (with hot reload)
npm run dev

# In another terminal, start WebSocket server
npm run ws

# Or run both together
npm run dev:all
```

**Access Dashboard:**
- http://localhost:5000 - Main dashboard
- http://localhost:5000/trading - Trading terminal
- http://localhost:5000/ml-analytics - ML analytics

---

### Step 7: Airflow Development Setup

```bash
# Start Airflow services
docker compose up -d airflow-init
# Wait for initialization to complete

docker compose up -d airflow-scheduler airflow-webserver

# Access Airflow UI
# http://localhost:8080
# Username: admin
# Password: admin123
```

---

### Step 8: VS Code Configuration

**Install Recommended Extensions:**
- Python (ms-python.python)
- Pylance (ms-python.vscode-pylance)
- ESLint (dbaeumer.vscode-eslint)
- Prettier (esbenp.prettier-vscode)
- Docker (ms-azuretools.vscode-docker)
- GitLens (eamodio.gitlens)

**Workspace Settings** (`.vscode/settings.json`):

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    "**/.pytest_cache": true
  }
}
```

---

## Project Structure

```
USDCOP-RL-Models/
├── airflow/
│   ├── dags/                       # Airflow DAG definitions
│   │   ├── base_pipeline.py        # Base class for all DAGs
│   │   ├── usdcop_m5__01_l0_intelligent_acquire.py
│   │   ├── usdcop_m5__02_l1_standardize.py
│   │   └── ...
│   ├── configs/                    # Configuration files
│   │   ├── twelve_data_config.yaml
│   │   └── pipeline_health_config.yaml
│   ├── Dockerfile.prod             # Airflow container image
│   └── requirements.txt            # Airflow dependencies
│
├── services/                       # FastAPI microservices
│   ├── trading_api_realtime.py     # Trading API (Port 8000)
│   ├── trading_analytics_api.py    # Analytics API (Port 8001)
│   ├── pipeline_data_api.py        # Pipeline API (Port 8002)
│   ├── compliance_api.py           # Compliance API (Port 8003)
│   ├── usdcop_realtime_orchestrator.py  # RT Orchestrator (Port 8085)
│   ├── websocket_service.py        # WebSocket service (Port 8082)
│   ├── Dockerfile.trading-api      # Docker images per service
│   ├── Dockerfile.orchestrator
│   └── requirements-api.txt        # API dependencies
│
├── usdcop-trading-dashboard/       # Next.js dashboard
│   ├── app/                        # Next.js App Router
│   │   ├── page.tsx                # Main dashboard page
│   │   ├── trading/                # Trading terminal
│   │   ├── ml-analytics/           # ML analytics
│   │   ├── diagnostico/            # System diagnostics
│   │   └── api/                    # Next.js API routes
│   ├── components/                 # React components
│   │   ├── charts/                 # Chart components
│   │   ├── pipeline/               # Pipeline status components
│   │   └── trading/                # Trading components
│   ├── lib/                        # Utilities and services
│   │   ├── api-client.ts           # API client
│   │   ├── websocket-client.ts     # WebSocket client
│   │   └── utils.ts                # Helper functions
│   ├── hooks/                      # Custom React hooks
│   ├── tests/                      # Test files
│   │   ├── unit/                   # Unit tests
│   │   ├── integration/            # Integration tests
│   │   └── e2e/                    # End-to-end tests
│   ├── package.json                # Node dependencies
│   └── Dockerfile.prod             # Dashboard container image
│
├── scripts/                        # Utility scripts
│   ├── backup_restore_system.py    # Backup/restore
│   ├── seed_development_data.py    # Seed test data
│   ├── test_api_endpoints.sh       # API integration tests
│   └── validate_100_percent.sh     # System validation
│
├── init-scripts/                   # Database initialization
│   ├── 01-schema.sql               # Table definitions
│   ├── 02-timescaledb.sql          # TimescaleDB setup
│   └── 03-seed-data.sql            # Sample data
│
├── docs/                           # Documentation
│   ├── ARCHITECTURE.md
│   ├── API_REFERENCE_V2.md
│   ├── DEVELOPMENT.md (this file)
│   ├── RUNBOOK.md
│   └── MIGRATION_GUIDE.md
│
├── docker-compose.yml              # Service orchestration
├── .env.example                    # Environment template
├── requirements.txt                # Python dependencies
└── README.md                       # Project overview
```

---

## Adding New Features

### Adding a New Symbol (e.g., USDBRL)

**Step 1: Update Configuration**

```yaml
# airflow/configs/usdcop_config.yaml
symbols:
  - symbol: "USDCOP"
    name: "US Dollar / Colombian Peso"
    exchange: "FOREX"
    timeframe: "5min"

  # Add new symbol
  - symbol: "USDBRL"
    name: "US Dollar / Brazilian Real"
    exchange: "FOREX"
    timeframe: "5min"
```

**Step 2: Update Database Schema**

```sql
-- Add symbol-specific table if needed
CREATE TABLE usdbrl_m5_ohlcv (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL DEFAULT 'USDBRL',
    open NUMERIC(20,8),
    high NUMERIC(20,8),
    low NUMERIC(20,8),
    close NUMERIC(20,8),
    volume NUMERIC(20,2),
    PRIMARY KEY (time, symbol)
);

SELECT create_hypertable('usdbrl_m5_ohlcv', 'time');
```

**Step 3: Create Airflow DAGs**

```bash
# Copy existing DAG templates
cd airflow/dags
cp usdcop_m5__01_l0_intelligent_acquire.py usdbrl_m5__01_l0_intelligent_acquire.py

# Update symbol references
sed -i 's/USDCOP/USDBRL/g' usdbrl_m5__01_l0_intelligent_acquire.py
sed -i 's/usdcop/usdbrl/g' usdbrl_m5__01_l0_intelligent_acquire.py

# Repeat for all layers (L0-L6)
```

**Step 4: Update APIs**

```python
# services/trading_api_realtime.py
SUPPORTED_SYMBOLS = ["USDCOP", "USDBRL"]

@app.get("/api/candlesticks/{symbol}")
async def get_candlesticks(
    symbol: str,
    timeframe: str = "5m",
    limit: int = 100
):
    if symbol not in SUPPORTED_SYMBOLS:
        raise HTTPException(status_code=400, detail=f"Symbol {symbol} not supported")

    # Query logic remains the same (supports any symbol)
    table_name = f"{symbol.lower()}_m5_ohlcv"
    # ...
```

**Step 5: Update Dashboard**

```typescript
// usdcop-trading-dashboard/lib/constants.ts
export const SUPPORTED_SYMBOLS = [
  { value: 'USDCOP', label: 'USD/COP' },
  { value: 'USDBRL', label: 'USD/BRL' },
];

// Add symbol selector in trading page
<SymbolSelector symbols={SUPPORTED_SYMBOLS} />
```

**Step 6: Test**

```bash
# Run integration tests
./scripts/test_api_endpoints.sh

# Trigger DAG manually
docker exec usdcop-airflow-webserver airflow dags trigger usdbrl_m5__01_l0_intelligent_acquire

# Verify data in database
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c \
  "SELECT COUNT(*) FROM usdbrl_m5_ohlcv;"
```

---

### Adding a New API Endpoint

**Example: Add `/api/volume-profile` endpoint**

**Step 1: Define Pydantic Models**

```python
# services/trading_api_realtime.py
from pydantic import BaseModel
from typing import List

class VolumeProfileBin(BaseModel):
    price_level: float
    volume: float
    percentage: float

class VolumeProfileResponse(BaseModel):
    symbol: str
    timeframe: str
    bins: List[VolumeProfileBin]
```

**Step 2: Implement Endpoint**

```python
@app.get("/api/volume-profile/{symbol}", response_model=VolumeProfileResponse)
async def get_volume_profile(
    symbol: str,
    timeframe: str = "5m",
    num_bins: int = 20
):
    """
    Calculate volume profile (price levels with most trading activity)
    """
    try:
        async with db_pool.acquire() as conn:
            # Query data
            rows = await conn.fetch("""
                SELECT
                    FLOOR(close / $3) * $3 AS price_level,
                    SUM(volume) AS total_volume
                FROM usdcop_m5_ohlcv
                WHERE symbol = $1
                AND time >= NOW() - INTERVAL '1 day'
                GROUP BY price_level
                ORDER BY price_level
            """, symbol, timeframe, num_bins)

            # Calculate percentages
            total_volume = sum(row['total_volume'] for row in rows)
            bins = [
                VolumeProfileBin(
                    price_level=row['price_level'],
                    volume=row['total_volume'],
                    percentage=(row['total_volume'] / total_volume) * 100
                )
                for row in rows
            ]

            return VolumeProfileResponse(
                symbol=symbol,
                timeframe=timeframe,
                bins=bins
            )

    except Exception as e:
        logger.error(f"Error fetching volume profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

**Step 3: Add Tests**

```python
# tests/test_trading_api.py
import pytest
from fastapi.testclient import TestClient
from services.trading_api_realtime import app

client = TestClient(app)

def test_volume_profile():
    response = client.get("/api/volume-profile/USDCOP?timeframe=5m&num_bins=20")
    assert response.status_code == 200

    data = response.json()
    assert data['symbol'] == 'USDCOP'
    assert len(data['bins']) > 0

    # Verify percentages sum to 100
    total_percentage = sum(bin['percentage'] for bin in data['bins'])
    assert abs(total_percentage - 100.0) < 0.1
```

**Step 4: Update API Documentation**

```markdown
# docs/API_REFERENCE_V2.md

### GET /api/volume-profile/{symbol}

**Description:** Calculate volume profile for a symbol.

**Parameters:**
- `symbol` (path): Trading symbol (e.g., "USDCOP")
- `timeframe` (query): Timeframe (default: "5m")
- `num_bins` (query): Number of price bins (default: 20)

**Response:**
\```json
{
  "symbol": "USDCOP",
  "timeframe": "5m",
  "bins": [
    {
      "price_level": 4350.0,
      "volume": 123456.78,
      "percentage": 15.2
    }
  ]
}
\```
```

**Step 5: Integrate in Dashboard**

```typescript
// usdcop-trading-dashboard/lib/api-client.ts
export async function getVolumeProfile(
  symbol: string,
  timeframe: string = '5m',
  numBins: number = 20
) {
  const response = await fetch(
    `${API_BASE_URL}/api/volume-profile/${symbol}?timeframe=${timeframe}&num_bins=${numBins}`
  );

  if (!response.ok) {
    throw new Error('Failed to fetch volume profile');
  }

  return response.json();
}
```

---

## Testing Guidelines

### Testing Strategy

**Test Pyramid:**
```
         ┌─────────────┐
         │  E2E Tests  │  (10%)
         │  Playwright │
         ├─────────────┤
         │ Integration │  (30%)
         │    Tests    │
         ├─────────────┤
         │ Unit Tests  │  (60%)
         │   Vitest    │
         └─────────────┘
```

---

### Unit Tests (Python - pytest)

```python
# tests/test_utils.py
import pytest
from utils import calculate_rsi, calculate_ema

def test_calculate_rsi():
    prices = [44, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08,
              45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64]

    rsi = calculate_rsi(prices, period=14)

    assert 0 <= rsi <= 100
    assert abs(rsi - 66.32) < 0.5  # Expected RSI value

def test_calculate_ema():
    prices = [22.27, 22.19, 22.08, 22.17, 22.18, 22.13, 22.23, 22.43, 22.24, 22.29]

    ema = calculate_ema(prices, period=10)

    assert ema > 0
    assert abs(ema - 22.22) < 0.01

@pytest.mark.asyncio
async def test_database_connection():
    from services.trading_api_realtime import get_db_pool

    pool = await get_db_pool()
    async with pool.acquire() as conn:
        result = await conn.fetchval("SELECT 1")
        assert result == 1
```

**Run Python Tests:**

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=services --cov-report=html

# Run specific test file
pytest tests/test_trading_api.py

# Run specific test
pytest tests/test_utils.py::test_calculate_rsi
```

---

### Unit Tests (JavaScript/TypeScript - Vitest)

```typescript
// usdcop-trading-dashboard/tests/unit/utils.test.ts
import { describe, it, expect } from 'vitest';
import { formatCurrency, formatPercentage, calculatePnL } from '@/lib/utils';

describe('Utility Functions', () => {
  it('should format currency correctly', () => {
    expect(formatCurrency(4350.50)).toBe('$4,350.50');
    expect(formatCurrency(1000000)).toBe('$1,000,000.00');
  });

  it('should format percentage correctly', () => {
    expect(formatPercentage(0.15)).toBe('15.00%');
    expect(formatPercentage(-0.05)).toBe('-5.00%');
  });

  it('should calculate P&L correctly', () => {
    const pnl = calculatePnL({
      entryPrice: 4350,
      exitPrice: 4400,
      quantity: 100,
      side: 'long'
    });

    expect(pnl).toBe(5000);  // (4400 - 4350) * 100
  });
});
```

**Run Dashboard Tests:**

```bash
cd usdcop-trading-dashboard

# Run all unit tests
npm run test

# Run with UI
npm run test:ui

# Run with coverage
npm run test:coverage

# Watch mode (rerun on file changes)
npm run test:watch
```

---

### Integration Tests

```python
# tests/integration/test_api_flow.py
import pytest
import asyncio
from httpx import AsyncClient
from services.trading_api_realtime import app

@pytest.mark.asyncio
async def test_full_trading_flow():
    """Test complete trading workflow"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Step 1: Get latest price
        response = await client.get("/api/latest/USDCOP")
        assert response.status_code == 200
        latest_data = response.json()
        current_price = latest_data['price']

        # Step 2: Place order
        order_response = await client.post("/api/orders", json={
            "symbol": "USDCOP",
            "side": "buy",
            "quantity": 100,
            "price": current_price
        })
        assert order_response.status_code == 201
        order_id = order_response.json()['order_id']

        # Step 3: Check positions
        positions_response = await client.get("/api/trading/positions")
        assert positions_response.status_code == 200
        positions = positions_response.json()
        assert any(p['order_id'] == order_id for p in positions)
```

---

### End-to-End Tests (Playwright)

```typescript
// usdcop-trading-dashboard/tests/e2e/dashboard.spec.ts
import { test, expect } from '@playwright/test';

test.describe('Dashboard', () => {
  test('should load main dashboard', async ({ page }) => {
    await page.goto('http://localhost:5000');

    // Check title
    await expect(page).toHaveTitle(/USDCOP Trading/);

    // Check pipeline status cards visible
    await expect(page.locator('[data-testid="layer-l0"]')).toBeVisible();
    await expect(page.locator('[data-testid="layer-l1"]')).toBeVisible();

    // Check health indicators
    const healthStatus = await page.locator('[data-testid="health-status"]').textContent();
    expect(healthStatus).toMatch(/Healthy|Degraded/);
  });

  test('should navigate to trading terminal', async ({ page }) => {
    await page.goto('http://localhost:5000');

    // Click trading link
    await page.click('a[href="/trading"]');

    // Wait for chart to load
    await page.waitForSelector('[data-testid="trading-chart"]');

    // Check chart is visible
    const chart = page.locator('[data-testid="trading-chart"]');
    await expect(chart).toBeVisible();
  });

  test('should display real-time data via WebSocket', async ({ page }) => {
    await page.goto('http://localhost:5000/trading');

    // Wait for WebSocket connection
    await page.waitForFunction(() => {
      return window.wsConnected === true;
    }, { timeout: 10000 });

    // Check price updates
    const initialPrice = await page.locator('[data-testid="current-price"]').textContent();

    // Wait for price update (max 10 seconds)
    await page.waitForFunction((initial) => {
      const current = document.querySelector('[data-testid="current-price"]')?.textContent;
      return current !== initial;
    }, initialPrice, { timeout: 10000 });
  });
});
```

**Run E2E Tests:**

```bash
cd usdcop-trading-dashboard

# Install browsers (first time only)
npm run playwright:install

# Run E2E tests
npm run test:e2e

# Run with UI
npm run test:e2e:ui

# Debug mode
npm run test:e2e:debug
```

---

## Coding Standards

### Python (PEP 8 + Black + isort)

**Code Formatting:**

```python
# Use Black for consistent formatting
black services/ airflow/dags/ scripts/

# Use isort for import sorting
isort services/ airflow/dags/ scripts/

# Run linter
pylint services/
```

**Style Guidelines:**

```python
# Good: Type hints, docstrings, clear variable names
from typing import List, Optional
from datetime import datetime

async def get_market_data(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Fetch market data for a symbol within a date range.

    Args:
        symbol: Trading symbol (e.g., "USDCOP")
        start_date: Start of date range (inclusive)
        end_date: End of date range (inclusive)
        limit: Maximum number of records to return

    Returns:
        List of OHLCV dictionaries

    Raises:
        ValueError: If start_date > end_date
        DatabaseError: If database connection fails
    """
    if start_date > end_date:
        raise ValueError("start_date must be <= end_date")

    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT time, open, high, low, close, volume
            FROM usdcop_m5_ohlcv
            WHERE symbol = $1
            AND time BETWEEN $2 AND $3
            ORDER BY time DESC
            LIMIT $4
        """, symbol, start_date, end_date, limit)

        return [dict(row) for row in rows]


# Bad: No type hints, no docstring, unclear names
async def get_data(s, sd, ed, l=100):
    rows = await conn.fetch("SELECT * FROM table WHERE symbol = $1", s)
    return rows
```

**Error Handling:**

```python
# Good: Specific exceptions, logging, user-friendly messages
import logging
from fastapi import HTTPException

logger = logging.getLogger(__name__)

@app.get("/api/data/{symbol}")
async def get_data(symbol: str):
    try:
        data = await fetch_market_data(symbol)
        return {"status": "success", "data": data}

    except ValueError as e:
        logger.warning(f"Invalid request for symbol {symbol}: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except DatabaseError as e:
        logger.error(f"Database error fetching {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Database unavailable")

    except Exception as e:
        logger.exception(f"Unexpected error fetching {symbol}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Bad: Generic exception, no logging
@app.get("/api/data/{symbol}")
async def get_data(symbol: str):
    try:
        return fetch_market_data(symbol)
    except:
        return {"error": "something went wrong"}
```

---

### TypeScript (ESLint + Prettier)

**Configuration:**

```json
// .eslintrc.json
{
  "extends": [
    "next/core-web-vitals",
    "plugin:@typescript-eslint/recommended"
  ],
  "rules": {
    "@typescript-eslint/no-explicit-any": "warn",
    "@typescript-eslint/no-unused-vars": "error",
    "prefer-const": "error"
  }
}
```

**Style Guidelines:**

```typescript
// Good: Type safety, clear interfaces, functional components
interface MarketData {
  symbol: string;
  price: number;
  timestamp: string;
  volume: number;
}

interface MarketDataProps {
  symbol: string;
  onPriceUpdate?: (price: number) => void;
}

export const MarketDataDisplay: React.FC<MarketDataProps> = ({
  symbol,
  onPriceUpdate
}) => {
  const [data, setData] = useState<MarketData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const response = await fetch(`/api/latest/${symbol}`);

        if (!response.ok) {
          throw new Error(`HTTP error ${response.status}`);
        }

        const data: MarketData = await response.json();
        setData(data);
        onPriceUpdate?.(data.price);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [symbol, onPriceUpdate]);

  if (loading) return <LoadingSpinner />;
  if (error) return <ErrorMessage message={error} />;
  if (!data) return null;

  return (
    <div className="market-data-display">
      <h2>{data.symbol}</h2>
      <p>Price: ${data.price.toFixed(2)}</p>
    </div>
  );
};


// Bad: No types, any everywhere, unclear logic
export const MarketDataDisplay = ({ symbol, onPriceUpdate }: any) => {
  const [data, setData] = useState<any>(null);

  useEffect(() => {
    fetch(`/api/latest/${symbol}`)
      .then((r) => r.json())
      .then((d) => {
        setData(d);
        if (onPriceUpdate) onPriceUpdate(d.price);
      });
  }, [symbol]);

  return <div>{data?.price}</div>;
};
```

**Run Linting:**

```bash
cd usdcop-trading-dashboard

# Check for linting errors
npm run lint

# Auto-fix issues
npm run lint -- --fix

# Format with Prettier
npx prettier --write .
```

---

## Database Development

### Schema Migrations

**Creating a Migration:**

```sql
-- migrations/001_add_volatility_column.sql
-- Description: Add volatility column to OHLCV table
-- Date: 2025-10-22

-- Add column
ALTER TABLE usdcop_m5_ohlcv
ADD COLUMN IF NOT EXISTS volatility NUMERIC(10,4);

-- Create index for performance
CREATE INDEX IF NOT EXISTS idx_volatility ON usdcop_m5_ohlcv(volatility);

-- Backfill data (for existing rows)
UPDATE usdcop_m5_ohlcv
SET volatility = (high - low) / close
WHERE volatility IS NULL;
```

**Running Migrations:**

```bash
# Apply migration
docker exec -i usdcop-postgres-timescale psql -U admin -d usdcop_trading < migrations/001_add_volatility_column.sql

# Verify migration
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c "\d usdcop_m5_ohlcv"
```

**Rollback Script:**

```sql
-- migrations/001_add_volatility_column_rollback.sql
ALTER TABLE usdcop_m5_ohlcv DROP COLUMN IF EXISTS volatility;
DROP INDEX IF EXISTS idx_volatility;
```

---

### Query Performance Optimization

**Use EXPLAIN ANALYZE:**

```sql
-- Check query execution plan
EXPLAIN ANALYZE
SELECT time, close, volume
FROM usdcop_m5_ohlcv
WHERE symbol = 'USDCOP'
AND time >= NOW() - INTERVAL '1 day'
ORDER BY time DESC
LIMIT 100;

-- Output shows:
-- - Execution time
-- - Index usage
-- - Rows scanned
```

**Optimization Tips:**

1. **Use indexes for frequently queried columns:**
   ```sql
   CREATE INDEX idx_symbol_time ON usdcop_m5_ohlcv(symbol, time DESC);
   ```

2. **Use hypertable chunks for time-based queries:**
   ```sql
   -- TimescaleDB automatically partitions by time
   -- Queries only scan relevant chunks
   ```

3. **Use materialized views for complex aggregations:**
   ```sql
   CREATE MATERIALIZED VIEW daily_stats AS
   SELECT
       symbol,
       DATE(time) AS date,
       AVG(close) AS avg_close,
       STDDEV(close) AS volatility,
       SUM(volume) AS total_volume
   FROM usdcop_m5_ohlcv
   GROUP BY symbol, DATE(time);

   -- Refresh daily
   REFRESH MATERIALIZED VIEW CONCURRENTLY daily_stats;
   ```

---

## API Development

### FastAPI Best Practices

**Dependency Injection for Database:**

```python
# services/database.py
from contextlib import asynccontextmanager
import asyncpg

db_pool: asyncpg.Pool = None

async def get_db_pool() -> asyncpg.Pool:
    global db_pool
    if db_pool is None:
        db_pool = await asyncpg.create_pool(
            host='postgres',
            port=5432,
            user='admin',
            password='admin123',
            database='usdcop_trading',
            min_size=5,
            max_size=20
        )
    return db_pool

@asynccontextmanager
async def get_db_connection():
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        yield conn
```

**Use in Endpoints:**

```python
# services/trading_api_realtime.py
from database import get_db_connection

@app.get("/api/latest/{symbol}")
async def get_latest(symbol: str):
    async with get_db_connection() as conn:
        row = await conn.fetchrow("""
            SELECT * FROM usdcop_m5_ohlcv
            WHERE symbol = $1
            ORDER BY time DESC
            LIMIT 1
        """, symbol)

        if not row:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")

        return dict(row)
```

---

## Dashboard Development

### Component Guidelines

**Use Server Components by Default:**

```typescript
// app/page.tsx (Server Component)
export default async function DashboardPage() {
  // Fetch data server-side (no need for useEffect)
  const pipelineStatus = await getPipelineStatus();

  return (
    <div>
      <h1>Pipeline Status</h1>
      <PipelineStatusCards data={pipelineStatus} />
    </div>
  );
}
```

**Use Client Components When Needed:**

```typescript
// components/charts/TradingChart.tsx
'use client';

import { useEffect, useState } from 'react';
import { createChart } from 'lightweight-charts';

export const TradingChart: React.FC<{ symbol: string }> = ({ symbol }) => {
  const [chart, setChart] = useState<IChartApi | null>(null);

  useEffect(() => {
    const chartInstance = createChart(chartContainerRef.current, {
      width: 800,
      height: 400,
    });

    setChart(chartInstance);

    return () => chartInstance.remove();
  }, []);

  // ...rest of component
};
```

---

## Pipeline Development

### Creating a New Pipeline Layer

**Example: Add L7 for advanced analytics**

**Step 1: Create DAG File**

```python
# airflow/dags/usdcop_m5__08_l7_advanced_analytics.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from base_pipeline import BasePipeline

class L7AdvancedAnalytics(BasePipeline):
    def __init__(self):
        super().__init__(
            dag_id='usdcop_m5__08_l7_advanced_analytics',
            description='L7: Advanced analytics and predictions',
            schedule_interval='*/30 * * * *',  # Every 30 minutes
        )

    def process_layer(self, **context):
        """Implement L7 processing logic"""
        # Read from L6 backtest results
        l6_data = self.read_from_minio('usdcop-l6-backtest')

        # Perform advanced analytics
        # - Market regime detection
        # - Correlation analysis
        # - Prediction confidence intervals

        analytics_results = self.compute_advanced_analytics(l6_data)

        # Write to PostgreSQL
        self.write_to_postgres('l7_advanced_analytics', analytics_results)

        # Write to MinIO
        self.write_to_minio('usdcop-l7-analytics', analytics_results)

        return {"status": "success", "records": len(analytics_results)}

# Create DAG
l7_dag = L7AdvancedAnalytics().create_dag()
```

**Step 2: Test DAG Locally**

```bash
# Test DAG syntax
docker exec usdcop-airflow-scheduler python /opt/airflow/dags/usdcop_m5__08_l7_advanced_analytics.py

# Test DAG run
docker exec usdcop-airflow-scheduler airflow dags test usdcop_m5__08_l7_advanced_analytics 2025-10-22
```

**Step 3: Deploy**

```bash
# DAG files are auto-detected by Airflow (mounted volume)
# Just save the file and it will appear in Airflow UI

# Trigger DAG
docker exec usdcop-airflow-webserver airflow dags trigger usdcop_m5__08_l7_advanced_analytics
```

---

## Git Workflow

### Branch Strategy

```
main (production)
  │
  ├─ develop (integration)
  │   │
  │   ├─ feature/add-new-symbol
  │   ├─ feature/dashboard-improvements
  │   ├─ bugfix/rt-orchestrator-memory-leak
  │   └─ hotfix/api-crash
```

### Commit Messages

**Format:**

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, no logic change)
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `test`: Adding tests
- `chore`: Maintenance (dependencies, build, etc.)

**Example:**

```
feat(api): add volume profile endpoint

- Implement /api/volume-profile/{symbol} endpoint
- Add VolumeProfileResponse Pydantic model
- Add unit tests for volume calculations
- Update API documentation

Closes #123
```

---

## Pull Request Checklist

Before submitting a PR, ensure:

- [ ] Code follows style guidelines (Black, isort, ESLint)
- [ ] All tests pass (`pytest`, `npm run test`)
- [ ] New features have tests (unit + integration)
- [ ] API changes documented in `docs/API_REFERENCE_V2.md`
- [ ] Database migrations included (if applicable)
- [ ] No secrets or credentials in code
- [ ] PR description explains WHAT and WHY (not just HOW)
- [ ] Linked to issue/ticket number
- [ ] Updated CHANGELOG.md (if significant change)

**PR Template:**

```markdown
## Description
[Brief description of changes]

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Related Issue
Closes #123

## Testing
- [x] Unit tests added/updated
- [x] Integration tests added/updated
- [ ] E2E tests added/updated
- [x] Manual testing performed

## Screenshots (if applicable)
[Add screenshots for UI changes]

## Checklist
- [x] Code follows style guidelines
- [x] Tests pass locally
- [x] Documentation updated
- [x] No merge conflicts
```

---

## Additional Resources

- **Architecture:** `docs/ARCHITECTURE.md`
- **API Reference:** `docs/API_REFERENCE_V2.md`
- **Runbook:** `docs/RUNBOOK.md`
- **FastAPI Docs:** https://fastapi.tiangolo.com/
- **Next.js Docs:** https://nextjs.org/docs
- **Airflow Docs:** https://airflow.apache.org/docs/
- **TimescaleDB Docs:** https://docs.timescale.com/

---

**Questions?** Contact the development team at dev@trading.com
