# Configuration System - Quick Start Guide

Get started with the new centralized configuration system in 5 minutes.

## 1. Install (Already Done!)

The configuration system is already installed in `lib/config/`. You're ready to use it!

## 2. Basic Usage

### Import and Use Configurations

```typescript
// Import what you need
import { apiConfig, marketConfig, riskConfig, chartConfig } from '@/lib/config'

// Use in your component
function MyComponent() {
  const tradingUrl = apiConfig.trading.baseUrl      // 'http://localhost:8000'
  const symbol = marketConfig.defaultSymbol          // 'USDCOP'
  const maxLeverage = riskConfig.leverage.max       // 5.0
  const chartHeight = chartConfig.dimensions.defaultHeight  // 700

  return <div>...</div>
}
```

### Use Helper Functions

```typescript
import { buildApiUrl, getChartHeight, getTimeframeInfo } from '@/lib/config'

// Build API URLs
const url = buildApiUrl('trading', '/api/positions')
// Returns: 'http://localhost:8000/api/positions'

// Get chart height presets
const height = getChartHeight('standard')  // Returns: 500

// Get timeframe information
const timeframe = getTimeframeInfo('5m')
// Returns: { value: '5m', label: '5 Minutes', interval: 300, seconds: 300 }
```

## 3. Set Up Environment Variables (Optional)

Create `.env.local` in the project root:

```bash
# Copy the example file
cp .env.example .env.local

# Edit with your values
NEXT_PUBLIC_TRADING_API_URL=http://localhost:8000
NEXT_PUBLIC_DEFAULT_SYMBOL=USDCOP
NEXT_PUBLIC_MAX_LEVERAGE=5.0
```

Restart your dev server after creating `.env.local`:
```bash
npm run dev
```

## 4. Replace Your First Hardcoded Value

Let's replace a common hardcoded value as an example:

### Before:
```typescript
// app/trading/page.tsx
function TradingPage() {
  const symbol = 'USDCOP'  // ❌ Hardcoded
  const refreshInterval = 30000  // ❌ Hardcoded

  return <Chart symbol={symbol} />
}
```

### After:
```typescript
// app/trading/page.tsx
import { marketConfig, refreshIntervals } from '@/lib/config'

function TradingPage() {
  const symbol = marketConfig.defaultSymbol  // ✅ From config
  const refreshInterval = refreshIntervals.stats  // ✅ From config

  return <Chart symbol={symbol} />
}
```

## 5. Common Patterns

### API Calls

```typescript
import { apiConfig } from '@/lib/config'

// Before
const response = await fetch('http://localhost:8000/api/positions')

// After
const response = await fetch(`${apiConfig.trading.baseUrl}/api/positions`)

// Even better (with helper)
import { buildApiUrl } from '@/lib/config'
const response = await fetch(buildApiUrl('trading', '/api/positions'))
```

### Refresh Intervals

```typescript
import { refreshIntervals } from '@/lib/config'

// Before
setInterval(fetchData, 30000)  // 30 seconds
setInterval(fetchStats, 60000)  // 60 seconds

// After
setInterval(fetchData, refreshIntervals.stats)      // 30 seconds
setInterval(fetchStats, refreshIntervals.analytics) // 60 seconds
```

### Risk Thresholds

```typescript
import { riskConfig } from '@/lib/config'

// Before
const maxLeverage = 5.0
const varLimit = 0.10

// After
const maxLeverage = riskConfig.leverage.max
const varLimit = riskConfig.var.maxPercentage
```

### Chart Heights

```typescript
import { getChartHeight } from '@/lib/config'

// Before
<Chart height={500} />

// After
<Chart height={getChartHeight('standard')} />

// Or for custom height
import { chartConfig } from '@/lib/config'
<Chart height={chartConfig.dimensions.defaultHeight} />
```

## 6. Available Configurations

### API (`apiConfig`)
- `apiConfig.trading.baseUrl` - Trading API URL
- `apiConfig.analytics.baseUrl` - Analytics API URL
- `apiConfig.pipeline.baseUrl` - Pipeline API URL
- `apiConfig.websocket.tradingUrl` - WebSocket URL

### Market (`marketConfig`)
- `marketConfig.defaultSymbol` - Default trading symbol
- `marketConfig.defaultPortfolioValue` - Portfolio value
- `timeframeConfig.defaultTimeframe` - Default timeframe
- `refreshIntervals.*` - All refresh intervals

### Risk (`riskConfig`)
- `riskConfig.leverage.max` - Maximum leverage
- `riskConfig.var.maxPercentage` - VaR limit
- `riskConfig.drawdown.max` - Max drawdown
- `riskConfig.concentration.maxSinglePosition` - Position limit

### UI (`uiConfig`)
- `chartConfig.dimensions.*` - Chart sizes
- `chartConfig.colors.*` - Theme colors
- `tableConfig.*` - Table settings
- `layoutConfig.*` - Layout settings

## 7. Check Configuration

In development mode, the config system logs all loaded values:

```javascript
// Open browser console to see:
📋 Configuration System Loaded
  API Endpoints: { trading: 'http://localhost:8000', ... }
  Market Settings: { symbol: 'USDCOP', ... }
  Risk Thresholds: { maxLeverage: 5.0, ... }
```

## 8. Validation Checklist

After using the config system:

- [ ] Import from `@/lib/config` works
- [ ] Configuration values are correct
- [ ] TypeScript autocomplete works
- [ ] No hardcoded values remain
- [ ] Application runs without errors

## 9. Common Issues

### Issue: Module not found

**Solution:** Ensure you're using the correct import path:
```typescript
import { apiConfig } from '@/lib/config'  // ✅ Correct
import { apiConfig } from 'lib/config'    // ❌ Wrong
```

### Issue: Environment variables not working

**Solution:**
1. Ensure variable starts with `NEXT_PUBLIC_` for client-side
2. Restart dev server after creating `.env.local`
3. Clear Next.js cache: `rm -rf .next`

### Issue: TypeScript errors

**Solution:** Import types if needed:
```typescript
import type { Timeframe, ChartType } from '@/lib/config'
```

## 10. Next Steps

1. **Read the Full Documentation**
   - `README.md` - Complete usage guide
   - `MIGRATION_GUIDE.md` - Detailed migration steps

2. **Start Migrating Your Code**
   - Begin with high-priority files
   - Test as you go
   - Commit incrementally

3. **Set Up Production Environment**
   - Add environment variables to your hosting platform
   - Test with production URLs
   - Verify all configs work

## Quick Reference

### All Available Imports

```typescript
// Configuration objects
import {
  apiConfig,
  marketConfig,
  timeframeConfig,
  refreshIntervals,
  riskConfig,
  chartConfig,
  tableConfig,
  layoutConfig
} from '@/lib/config'

// Helper functions
import {
  buildApiUrl,
  getTimeframeInfo,
  getRefreshInterval,
  isMarketOpen,
  calculateRiskSeverity,
  getRecommendedAction,
  getChartHeight,
  getThemeColors
} from '@/lib/config'

// Types
import type {
  Config,
  ApiConfig,
  MarketConfig,
  RiskConfig,
  UiConfig,
  Timeframe,
  ChartType,
  Theme,
  RiskSeverity
} from '@/lib/config'

// Unified config (alternative)
import config from '@/lib/config'
```

### Environment Variables Template

```bash
# .env.local
NEXT_PUBLIC_TRADING_API_URL=http://localhost:8000
NEXT_PUBLIC_ANALYTICS_API_URL=http://localhost:8001
NEXT_PUBLIC_PIPELINE_API_URL=http://localhost:8002
NEXT_PUBLIC_DEFAULT_SYMBOL=USDCOP
NEXT_PUBLIC_PORTFOLIO_VALUE=10000000
NEXT_PUBLIC_MAX_LEVERAGE=5.0
```

## Support

Need help? Check these resources:

1. **`README.md`** - Full documentation
2. **`MIGRATION_GUIDE.md`** - Step-by-step migration
3. **`HARDCODED_VALUES_AUDIT.md`** - Specific replacements
4. **Configuration files** - Well-commented code

---

**You're now ready to use the configuration system!** 🎉

Start by importing `apiConfig` or `marketConfig` in your next component and replace one hardcoded value. Then move on to the next one. Happy coding!
