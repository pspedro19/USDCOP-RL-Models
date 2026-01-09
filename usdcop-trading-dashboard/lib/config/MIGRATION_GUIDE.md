# Configuration System Migration Guide

This guide helps you replace hardcoded values throughout the codebase with centralized configuration.

## Quick Reference

### Common Replacements

| Hardcoded Value | Config Import | Notes |
|-----------------|---------------|-------|
| `'http://localhost:8000'` | `apiConfig.trading.baseUrl` | Trading API |
| `'http://localhost:8001'` | `apiConfig.analytics.baseUrl` | Analytics API |
| `'http://localhost:8002'` | `apiConfig.pipeline.baseUrl` | Pipeline API |
| `'USDCOP'` | `marketConfig.defaultSymbol` | Default symbol |
| `'5m'` | `timeframeConfig.defaultTimeframe` | Default timeframe |
| `30000` (refresh) | `refreshIntervals.stats` | 30s refresh |
| `1000` (refresh) | `refreshIntervals.realtime` | 1s refresh |
| `5.0` (max leverage) | `riskConfig.leverage.max` | Max leverage |
| `0.10` (VaR limit) | `riskConfig.var.maxPercentage` | 10% VaR |
| `0.15` (max drawdown) | `riskConfig.drawdown.max` | 15% drawdown |
| `700` (chart height) | `chartConfig.dimensions.defaultHeight` | Default height |
| `500` (chart height) | `getChartHeight('standard')` | Standard height |

## Step-by-Step Migration

### 1. Update Real-Time Risk Engine

**File:** `lib/services/real-time-risk-engine.ts`

**Before:**
```typescript
// Risk thresholds
private readonly riskThresholds = {
  maxLeverage: 5.0,
  varLimit: 0.10,
  maxDrawdown: 0.15,
  minLiquidityScore: 0.7,
  maxConcentration: 0.4
};
```

**After:**
```typescript
import { riskConfig } from '@/lib/config'

// Risk thresholds from centralized config
private readonly riskThresholds = {
  maxLeverage: riskConfig.leverage.max,
  varLimit: riskConfig.var.maxPercentage,
  maxDrawdown: riskConfig.drawdown.max,
  minLiquidityScore: riskConfig.liquidity.minScore,
  maxConcentration: riskConfig.concentration.maxSinglePosition
};
```

### 2. Update Pipeline API Client

**File:** `lib/services/pipeline-api-client.ts`

**Before:**
```typescript
const API_BASE_URLS = {
  pipeline: process.env.NEXT_PUBLIC_PIPELINE_API_URL || 'http://localhost:8002',
  compliance: process.env.NEXT_PUBLIC_COMPLIANCE_API_URL || 'http://localhost:8003',
  tradingAnalytics: process.env.NEXT_PUBLIC_TRADING_ANALYTICS_API_URL || 'http://localhost:8001',
  trading: process.env.NEXT_PUBLIC_TRADING_API_URL || 'http://localhost:8000'
};
```

**After:**
```typescript
import { apiConfig } from '@/lib/config'

const API_BASE_URLS = {
  pipeline: apiConfig.pipeline.baseUrl,
  compliance: apiConfig.compliance.baseUrl,
  tradingAnalytics: apiConfig.analytics.baseUrl,
  trading: apiConfig.trading.baseUrl
};
```

### 3. Update Chart Components

**File:** `components/charts/RealDataTradingChart.tsx`

**Before:**
```typescript
interface RealDataTradingChartProps {
  symbol?: string
  timeframe?: string
  height?: number
  className?: string
}

export default function RealDataTradingChart({
  symbol = 'USDCOP',
  timeframe = '5m',
  height = 700,
  className = ''
}: RealDataTradingChartProps) {
  // ...
}
```

**After:**
```typescript
import { marketConfig, timeframeConfig, chartConfig } from '@/lib/config'

interface RealDataTradingChartProps {
  symbol?: string
  timeframe?: string
  height?: number
  className?: string
}

export default function RealDataTradingChart({
  symbol = marketConfig.defaultSymbol,
  timeframe = timeframeConfig.defaultTimeframe,
  height = chartConfig.dimensions.defaultHeight,
  className = ''
}: RealDataTradingChartProps) {
  // ...
}
```

### 4. Update Hooks with Refresh Intervals

**File:** `hooks/useMarketStats.ts`

**Before:**
```typescript
export function useMarketStats(
  symbol: string = 'USDCOP',
  refreshInterval: number = 30000
) {
  // ...
}
```

**After:**
```typescript
import { marketConfig, refreshIntervals } from '@/lib/config'

export function useMarketStats(
  symbol: string = marketConfig.defaultSymbol,
  refreshInterval: number = refreshIntervals.stats
) {
  // ...
}
```

### 5. Update API Routes

**File:** `app/api/trading/[...path]/route.ts`

**Before:**
```typescript
const TRADING_API_URL = process.env.TRADING_API_URL || 'http://localhost:8000';
```

**After:**
```typescript
import { apiConfig } from '@/lib/config'

const TRADING_API_URL = apiConfig.trading.baseUrl;
```

### 6. Update Components Using setInterval

**File:** `app/agent-trading/page.tsx`

**Before:**
```typescript
const interval = setInterval(checkMarketStatus, 60000)  // 60 seconds
const interval2 = setInterval(fetchStats, 30000)        // 30 seconds
```

**After:**
```typescript
import { refreshIntervals } from '@/lib/config'

const interval = setInterval(checkMarketStatus, refreshIntervals.marketStatus)
const interval2 = setInterval(fetchStats, refreshIntervals.stats)
```

### 7. Update Chart Heights in Components

**File:** `components/views/EquityCurveChart.tsx`

**Before:**
```typescript
<ResponsiveContainer width="100%" height={500}>
```

**After:**
```typescript
import { getChartHeight } from '@/lib/config'

<ResponsiveContainer width="100%" height={getChartHeight('standard')}>
```

## Files to Update

Based on the codebase analysis, here are the key files that need updates:

### High Priority (Core Services)

1. `lib/services/real-time-risk-engine.ts` - Risk thresholds
2. `lib/services/pipeline-api-client.ts` - API endpoints
3. `lib/services/market-data-service.ts` - API URLs and symbols
4. `components/charts/RealDataTradingChart.tsx` - Chart defaults

### Medium Priority (Hooks)

5. `hooks/useMarketStats.ts` - Refresh intervals and symbols
6. `hooks/useAnalytics.ts` - Refresh intervals
7. `hooks/useRealTimePrice.ts` - Symbol defaults
8. `hooks/useDbStats.ts` - Refresh intervals
9. `hooks/trading/useRealTimeMarketData.ts` - Intervals and symbols

### Lower Priority (API Routes)

10. `app/api/trading/[...path]/route.ts` - API URLs
11. `app/api/analytics/[...path]/route.ts` - API URLs
12. `app/api/pipeline/**/*.ts` - API URLs
13. `app/api/backtest/**/*.ts` - API URLs

### UI Components

14. `components/charts/AdvancedHistoricalChart.tsx` - Heights
15. `components/charts/ChartWithPositions.tsx` - Heights and intervals
16. `components/views/EquityCurveChart.tsx` - Heights
17. `components/ml-analytics/**/*.tsx` - Chart heights

## Search and Replace Patterns

Use your IDE's search and replace with these patterns:

### API URLs

Search: `'http://localhost:8000'`
Replace: `apiConfig.trading.baseUrl`
Note: Import `apiConfig` from `@/lib/config`

Search: `'http://localhost:8001'`
Replace: `apiConfig.analytics.baseUrl`

Search: `'http://localhost:8002'`
Replace: `apiConfig.pipeline.baseUrl`

### Symbols

Search: `= 'USDCOP'`
Replace: `= marketConfig.defaultSymbol`
Note: Import `marketConfig` from `@/lib/config`

### Refresh Intervals

Search: `30000` (in context of setInterval)
Replace: `refreshIntervals.stats`

Search: `60000` (in context of setInterval)
Replace: `refreshIntervals.analytics` or `refreshIntervals.health`

Search: `1000` (in context of setInterval)
Replace: `refreshIntervals.realtime`

### Chart Heights

Search: `height={500}`
Replace: `height={getChartHeight('standard')}`

Search: `height={700}`
Replace: `height={getChartHeight('large')}`

Search: `height = 700`
Replace: `height = chartConfig.dimensions.defaultHeight`

## Testing After Migration

After updating files, verify:

1. **Application starts without errors**
   ```bash
   npm run dev
   ```

2. **API endpoints are correct**
   - Check browser console for API calls
   - Verify they're hitting the right URLs

3. **Risk thresholds work**
   - Open browser console and check risk engine logs
   - Verify thresholds match config values

4. **Charts render correctly**
   - Check chart heights match config
   - Verify timeframes work

5. **Refresh intervals are correct**
   - Monitor network tab for periodic requests
   - Verify timing matches config

6. **Environment variables override defaults**
   - Create `.env.local` with test values
   - Restart dev server
   - Verify overrides work

## Rollback Plan

If issues occur:

1. **Keep git history clean**
   ```bash
   git add -p  # Stage changes incrementally
   git commit -m "Update [component] to use centralized config"
   ```

2. **Test incrementally**
   - Update one file/component at a time
   - Test before moving to next file
   - Commit working changes

3. **Revert if needed**
   ```bash
   git revert <commit-hash>
   ```

## Common Issues and Solutions

### Issue: Type errors after importing config

**Solution:** Import both value and type
```typescript
import { apiConfig } from '@/lib/config'
import type { ApiConfig } from '@/lib/config'
```

### Issue: Environment variables not working

**Solution:**
1. Ensure variable name starts with `NEXT_PUBLIC_`
2. Restart dev server after changing `.env.local`
3. Clear `.next` cache: `rm -rf .next`

### Issue: Circular dependency errors

**Solution:** Import only what you need
```typescript
// Good
import { marketConfig } from '@/lib/config'

// Avoid
import config from '@/lib/config'  // Only if you need multiple domains
```

### Issue: Configuration values seem cached

**Solution:**
```bash
# Clear all caches
rm -rf .next
rm -rf node_modules/.cache
npm run dev
```

## Completion Checklist

- [ ] All API URLs use `apiConfig`
- [ ] All symbols use `marketConfig.defaultSymbol`
- [ ] All timeframes use `timeframeConfig.defaultTimeframe`
- [ ] All refresh intervals use `refreshIntervals.*`
- [ ] All risk thresholds use `riskConfig.*`
- [ ] All chart heights use `chartConfig.*` or `getChartHeight()`
- [ ] `.env.example` is up to date
- [ ] All imports are optimized (no unused imports)
- [ ] TypeScript builds without errors
- [ ] Application runs without console errors
- [ ] Environment variable overrides work
- [ ] Documentation is updated

## Benefits After Migration

1. **Single source of truth** for all configuration
2. **Easy environment-specific overrides** via env vars
3. **Type safety** for all config values
4. **Better maintainability** - change once, affect everywhere
5. **Clear documentation** of all configurable values
6. **Easier testing** - mock config values easily
7. **Follows Open/Closed Principle** - extend without modifying

## Next Steps

After completing migration:

1. **Update tests** to use mocked config values
2. **Add config validation** in development mode
3. **Document any new config values** in README
4. **Consider adding config presets** for different environments
5. **Set up CI/CD** to validate config on build

## Support

For questions or issues during migration:
- Review `lib/config/README.md`
- Check this migration guide
- Review the configuration files themselves
- Contact the development team
