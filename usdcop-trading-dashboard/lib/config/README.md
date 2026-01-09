# Configuration System

Centralized configuration management for the USDCOP Trading Dashboard.

## Overview

This configuration system replaces all hardcoded values throughout the application with a centralized, type-safe configuration structure. It supports environment variable overrides and follows the Open/Closed Principle for easy extension.

## Structure

```
lib/config/
├── index.ts           # Main exports and unified config object
├── api.config.ts      # API endpoints and network configuration
├── market.config.ts   # Market-related settings (symbols, timeframes, intervals)
├── risk.config.ts     # Risk management thresholds and limits
├── ui.config.ts       # UI defaults (charts, tables, layout, animations)
└── README.md          # This file
```

## Usage

### Basic Import

```typescript
import { apiConfig, marketConfig, riskConfig, uiConfig } from '@/lib/config'

// Access specific values
const tradingUrl = apiConfig.trading.baseUrl
const defaultSymbol = marketConfig.defaultSymbol
const maxLeverage = riskConfig.leverage.max
const chartHeight = uiConfig.chart.dimensions.defaultHeight
```

### Unified Config Object

```typescript
import config from '@/lib/config'

// Access all configuration domains
const url = config.api.trading.baseUrl
const symbol = config.market.defaultSymbol
const leverage = config.risk.thresholds.maxLeverage
const height = config.ui.chart.dimensions.defaultHeight
```

### Using Helper Functions

```typescript
import { buildApiUrl, getTimeframeInfo, getChartHeight, getThemeColors } from '@/lib/config'

// Build API URLs with parameters
const url = buildApiUrl('trading', '/api/positions', { symbol: 'USDCOP' })

// Get timeframe information
const timeframe = getTimeframeInfo('5m')
// Returns: { value: '5m', label: '5 Minutes', interval: 300, seconds: 300 }

// Get chart height preset
const height = getChartHeight('large') // Returns: 700

// Get theme colors
const colors = getThemeColors('dark')
```

## Configuration Domains

### 1. API Configuration (`api.config.ts`)

Manages all backend API endpoints and network settings.

```typescript
import { apiConfig, buildApiUrl } from '@/lib/config'

// API base URLs
apiConfig.trading.baseUrl        // Trading API
apiConfig.analytics.baseUrl      // Analytics API
apiConfig.pipeline.baseUrl       // Pipeline API
apiConfig.compliance.baseUrl     // Compliance API
apiConfig.backtest.baseUrl       // Backtest API

// Specific endpoints
apiConfig.trading.endpoints.positions
apiConfig.analytics.endpoints.riskMetrics
apiConfig.pipeline.endpoints.l0.rawData

// Network settings
apiConfig.websocket.reconnectDelay
apiConfig.websocket.maxReconnectAttempts

// Build URLs
const url = buildApiUrl('trading', '/api/positions')
```

**Environment Variables:**
- `NEXT_PUBLIC_TRADING_API_URL`
- `NEXT_PUBLIC_ANALYTICS_API_URL`
- `NEXT_PUBLIC_PIPELINE_API_URL`
- `NEXT_PUBLIC_COMPLIANCE_API_URL`
- `BACKTEST_API_URL`

### 2. Market Configuration (`market.config.ts`)

Market-related settings including symbols, timeframes, and refresh intervals.

```typescript
import { marketConfig, timeframeConfig, refreshIntervals } from '@/lib/config'

// Symbol settings
marketConfig.defaultSymbol              // 'USDCOP'
marketConfig.supportedSymbols           // ['USDCOP', 'USDBRL', ...]
marketConfig.defaultPortfolioValue      // 10000000

// Timeframes
timeframeConfig.defaultTimeframe        // '5m'
timeframeConfig.supported               // Array of timeframe objects

// Refresh intervals (milliseconds)
refreshIntervals.realtime               // 1000
refreshIntervals.stats                  // 30000
refreshIntervals.analytics              // 60000

// Helper functions
import { getTimeframeInfo, isMarketOpen } from '@/lib/config'

const info = getTimeframeInfo('5m')
const isOpen = isMarketOpen('colombia')
```

**Environment Variables:**
- `NEXT_PUBLIC_DEFAULT_SYMBOL`
- `NEXT_PUBLIC_PORTFOLIO_VALUE`

### 3. Risk Configuration (`risk.config.ts`)

Risk management thresholds and limits.

```typescript
import { riskConfig, riskThresholds } from '@/lib/config'

// Leverage limits
riskConfig.leverage.max                 // 5.0
riskConfig.leverage.warning             // 4.0

// VaR limits
riskConfig.var.maxPercentage            // 0.10 (10%)
riskConfig.var.warningPercentage        // 0.08 (8%)

// Drawdown limits
riskConfig.drawdown.max                 // 0.15 (15%)
riskConfig.drawdown.warning             // 0.12 (12%)

// Position concentration
riskConfig.concentration.maxSinglePosition  // 0.40 (40%)

// Helper functions
import { calculateRiskSeverity, getRecommendedAction } from '@/lib/config'

const severity = calculateRiskSeverity(currentValue, limitValue, warningValue)
const action = getRecommendedAction('leverage', severity)
```

**Environment Variables:**
- `NEXT_PUBLIC_MAX_LEVERAGE`
- `NEXT_PUBLIC_MAX_VAR_PCT`
- `NEXT_PUBLIC_MAX_DRAWDOWN_PCT`
- `NEXT_PUBLIC_MAX_CONCENTRATION_PCT`
- `NEXT_PUBLIC_MIN_LIQUIDITY_SCORE`

### 4. UI Configuration (`ui.config.ts`)

User interface defaults including charts, tables, layout, and animations.

```typescript
import { chartConfig, tableConfig, layoutConfig } from '@/lib/config'

// Chart settings
chartConfig.dimensions.defaultHeight    // 700
chartConfig.settings.showVolume         // true
chartConfig.colors.dark.background      // '#0a0e27'

// Table settings
tableConfig.defaultPageSize             // 25
tableConfig.rowHeight                   // 48

// Layout settings
layoutConfig.sidebar.defaultWidth       // 280
layoutConfig.breakpoints.tablet         // 768

// Helper functions
import { getChartHeight, getThemeColors } from '@/lib/config'

const height = getChartHeight('standard')  // 500
const colors = getThemeColors('dark')
```

## Environment Variables

Create a `.env.local` file in the project root with the following variables:

```bash
# API Endpoints
NEXT_PUBLIC_TRADING_API_URL=http://localhost:8000
NEXT_PUBLIC_ANALYTICS_API_URL=http://localhost:8001
NEXT_PUBLIC_PIPELINE_API_URL=http://localhost:8002
NEXT_PUBLIC_COMPLIANCE_API_URL=http://localhost:8003
NEXT_PUBLIC_ML_ANALYTICS_API_URL=http://localhost:8004
BACKTEST_API_URL=http://localhost:8006

# Market Configuration
NEXT_PUBLIC_DEFAULT_SYMBOL=USDCOP
NEXT_PUBLIC_PORTFOLIO_VALUE=10000000

# Risk Configuration
NEXT_PUBLIC_MAX_LEVERAGE=5.0
NEXT_PUBLIC_MAX_VAR_PCT=0.10
NEXT_PUBLIC_MAX_DRAWDOWN_PCT=0.15
NEXT_PUBLIC_MAX_CONCENTRATION_PCT=0.40
NEXT_PUBLIC_MIN_LIQUIDITY_SCORE=0.7

# External APIs
NEXT_PUBLIC_TWELVEDATA_API_KEY=your_api_key_here
```

## Migration Guide

### Replacing Hardcoded Values

**Before:**
```typescript
const API_URL = 'http://localhost:8000'
const DEFAULT_SYMBOL = 'USDCOP'
const REFRESH_INTERVAL = 30000
const MAX_LEVERAGE = 5.0
const CHART_HEIGHT = 700
```

**After:**
```typescript
import { apiConfig, marketConfig, refreshIntervals, riskConfig, uiConfig } from '@/lib/config'

const API_URL = apiConfig.trading.baseUrl
const DEFAULT_SYMBOL = marketConfig.defaultSymbol
const REFRESH_INTERVAL = refreshIntervals.stats
const MAX_LEVERAGE = riskConfig.leverage.max
const CHART_HEIGHT = uiConfig.chart.dimensions.defaultHeight
```

### Updating Components

**Before:**
```typescript
function RiskEngine() {
  const maxLeverage = 5.0
  const varLimit = 0.10
  const maxDrawdown = 0.15

  // ... rest of component
}
```

**After:**
```typescript
import { riskConfig } from '@/lib/config'

function RiskEngine() {
  const { maxLeverage } = riskConfig.leverage
  const { maxPercentage: varLimit } = riskConfig.var
  const { max: maxDrawdown } = riskConfig.drawdown

  // ... rest of component
}
```

### Updating API Calls

**Before:**
```typescript
const response = await fetch('http://localhost:8000/api/positions')
```

**After:**
```typescript
import { buildApiUrl } from '@/lib/config'

const response = await fetch(buildApiUrl('trading', '/api/positions'))
```

## Type Safety

All configuration values are fully typed:

```typescript
import type { Config, Timeframe, ChartType, RiskSeverity } from '@/lib/config'

// Use types in your components
function MyComponent(props: { timeframe: Timeframe }) {
  // TypeScript knows timeframe is '1m' | '5m' | '15m' | ...
}
```

## Extending Configuration

To add new configuration values:

1. Add to the appropriate config file
2. Export from that file
3. Add to the unified config object in `index.ts`
4. Update types if needed

**Example: Adding a new API endpoint**

```typescript
// In api.config.ts
export const apiConfig = {
  // ... existing config
  myNewApi: {
    baseUrl: process.env.NEXT_PUBLIC_MY_NEW_API_URL || 'http://localhost:9000',
    endpoints: {
      data: '/api/data',
    },
  },
} as const;

// In index.ts
const config = {
  api: apiConfig,  // Automatically includes myNewApi
  // ... rest of config
};
```

## Best Practices

1. **Always use config values** instead of hardcoding
2. **Use environment variables** for deployment-specific values
3. **Use helper functions** when available for cleaner code
4. **Add new config values** to the appropriate domain
5. **Document new values** in this README
6. **Keep defaults sensible** for development
7. **Use TypeScript types** for type safety

## Validation

In development mode, the configuration system logs loaded values to the console for verification:

```
📋 Configuration System Loaded
  API Endpoints: { trading: 'http://localhost:8000', ... }
  Market Settings: { symbol: 'USDCOP', ... }
  Risk Thresholds: { maxLeverage: 5.0, ... }
```

## Troubleshooting

### Environment variables not working

- Ensure variables start with `NEXT_PUBLIC_` for client-side access
- Restart the dev server after changing `.env.local`
- Check that `.env.local` is in the project root (not in `lib/config`)

### TypeScript errors

- Import types from `@/lib/config`
- Use `as const` for readonly configuration objects
- Ensure you're importing the correct type exports

### Configuration not updating

- Clear Next.js cache: `rm -rf .next`
- Restart the development server
- Check browser console for validation errors

## Support

For issues or questions about the configuration system:
1. Check this README
2. Review the configuration files
3. Check the project's main documentation
4. Contact the development team
