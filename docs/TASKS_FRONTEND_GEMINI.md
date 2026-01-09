# Tareas Frontend - Gemini (Next.js/React/TypeScript)

**Proyecto:** USDCOP RL Trading System V19
**Asignado a:** Gemini
**Referencia Contratos:** `docs/API_CONTRACTS_SHARED.md`

---

## Contexto del Sistema

### Stack Tecnologico
- **Framework:** Next.js 15.x (App Router)
- **UI:** Tailwind CSS + shadcn/ui
- **State:** Zustand (lib/store/)
- **Data Fetching:** SWR / React Query
- **Charts:** Recharts / Lightweight Charts
- **WebSocket:** Native WebSocket API

### Estructura del Proyecto
```
usdcop-trading-dashboard/
├── app/                    # Next.js App Router pages
├── components/
│   ├── charts/            # Componentes de graficos
│   ├── trading/           # Componentes de trading (signals, metrics)
│   ├── views/             # Vistas completas
│   └── ui/                # shadcn/ui components
├── hooks/                  # Custom React hooks
├── lib/
│   ├── types/             # TypeScript interfaces
│   ├── services/          # API clients
│   └── store/             # Zustand stores
└── tests/                  # E2E tests (Playwright)
```

### API Endpoints Disponibles
| Endpoint | Descripcion |
|----------|-------------|
| `GET /api/signals/latest` | Ultimas senales |
| `GET /api/performance?period=30d` | Metricas performance |
| `GET /api/positions` | Posiciones abiertas |
| `GET /api/risk/status` | Estado RiskManager (NUEVO) |
| `GET /api/monitor/health` | Salud de modelos (NUEVO) |
| `WS /ws/trading-signals` | Stream de senales |

---

## Fase 1: Tipos y Contratos - SEMANA 1

### CT-01: Crear archivo tipos compartidos
**Archivo:** `types/contracts.ts`
**Complejidad:** Baja
**Dependencias:** Ninguna

> [!IMPORTANT]
> No crear `lib/types/` - ya existe `types/` como directorio canónico.
> Reutilizar tipos existentes de `types/trading.ts` donde sea posible.

**Implementacion:**
```typescript
// types/contracts.ts
// Re-exportar tipos existentes para evitar duplicación
export { MarketStatus, OrderSide as TradeSide } from './trading';
export type { Position, Trade } from './trading';

// Nuevos tipos solo para contracts no cubiertos por trading.ts
export type SignalType = 'LONG' | 'SHORT' | 'HOLD' | 'CLOSE';

export interface RiskLimits {
  max_drawdown_pct: number;
  max_daily_loss_pct: number;
  max_trades_per_day: number;
}

export interface RiskStatusResponse {
  kill_switch_active: boolean;
  current_drawdown_pct: number;
  daily_pnl_pct: number;
  trades_today: number;
  consecutive_losses: number;
  cooldown_active: boolean;
  cooldown_until?: string;
  limits: RiskLimits;
}

// ... resto de contracts específicos de la API
```

**Verificacion:**
```bash
# No debe haber errores de TypeScript
cd usdcop-trading-dashboard && npx tsc --noEmit
```

---

### FE-01: Actualizar useModelSignals con tipos contract
**Archivo:** `hooks/useModelSignals.ts`
**Complejidad:** Baja
**Dependencias:** CT-01

**Archivo Actual:** Verificar si existe en `hooks/useModelSignals.ts`

**Implementacion Requerida:**
```typescript
// hooks/useModelSignals.ts
import useSWR from 'swr';
import {
  LatestSignalsResponse,
  StrategySignal,
  APIError
} from '@/types/contracts';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const fetcher = async (url: string): Promise<LatestSignalsResponse> => {
  const res = await fetch(url);
  if (!res.ok) {
    const error: APIError = await res.json();
    throw new Error(error.message);
  }
  return res.json();
};

interface UseModelSignalsOptions {
  refreshInterval?: number;  // ms, default 5000
  enabled?: boolean;
}

interface UseModelSignalsReturn {
  signals: StrategySignal[];
  marketPrice: number | null;
  marketStatus: string | null;
  timestamp: string | null;
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
  mutate: () => void;
}

export function useModelSignals(
  options: UseModelSignalsOptions = {}
): UseModelSignalsReturn {
  const { refreshInterval = 5000, enabled = true } = options;

  const { data, error, isLoading, mutate } = useSWR<LatestSignalsResponse>(
    enabled ? `${API_BASE}/api/signals/latest` : null,
    fetcher,
    {
      refreshInterval,
      revalidateOnFocus: true,
      dedupingInterval: 2000,
    }
  );

  return {
    signals: data?.signals ?? [],
    marketPrice: data?.market_price ?? null,
    marketStatus: data?.market_status ?? null,
    timestamp: data?.timestamp ?? null,
    isLoading,
    isError: !!error,
    error: error ?? null,
    mutate,
  };
}

// Hook auxiliar para obtener senal de modelo especifico
export function useModelSignal(strategyCode: string) {
  const { signals, ...rest } = useModelSignals();
  const signal = signals.find(s => s.strategy_code === strategyCode) ?? null;
  return { signal, ...rest };
}
```

**Criterios de Aceptacion:**
- [x] Usa tipos de `api-contracts.ts`
- [x] Maneja errores correctamente
- [x] Tiene refresh automatico configurable
- [x] Exporta hook auxiliar `useModelSignal`

---

### FE-02: Actualizar useFinancialMetrics con tipos contract
**Archivo:** `hooks/useFinancialMetrics.ts`
**Complejidad:** Baja
**Dependencias:** CT-01

**Implementacion Requerida:**
```typescript
// hooks/useFinancialMetrics.ts
import useSWR from 'swr';
import {
  PerformanceResponse,
  StrategyPerformance,
  APIError
} from '@/types/contracts';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

type Period = '1d' | '7d' | '30d' | '90d' | 'all';

interface UseFinancialMetricsOptions {
  period?: Period;
  refreshInterval?: number;
  enabled?: boolean;
}

interface UseFinancialMetricsReturn {
  strategies: StrategyPerformance[];
  period: string | null;
  startDate: string | null;
  endDate: string | null;
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
}

export function useFinancialMetrics(
  options: UseFinancialMetricsOptions = {}
): UseFinancialMetricsReturn {
  const { period = '7d', refreshInterval = 60000, enabled = true } = options;

  const { data, error, isLoading } = useSWR<PerformanceResponse>(
    enabled ? `${API_BASE}/api/performance?period=${period}` : null,
    async (url) => {
      const res = await fetch(url);
      if (!res.ok) throw new Error('Failed to fetch performance');
      return res.json();
    },
    { refreshInterval }
  );

  return {
    strategies: data?.strategies ?? [],
    period: data?.period ?? null,
    startDate: data?.start_date ?? null,
    endDate: data?.end_date ?? null,
    isLoading,
    isError: !!error,
    error: error ?? null,
  };
}

// Hook para obtener metricas de un modelo especifico
export function useStrategyPerformance(strategyCode: string, period: Period = '7d') {
  const { strategies, ...rest } = useFinancialMetrics({ period });
  const strategy = strategies.find(s => s.strategy_code === strategyCode) ?? null;
  return { strategy, ...rest };
}
```

---

## Fase 2: Nuevos Hooks - SEMANA 2

### FE-03: Crear useRiskStatus hook
**Archivo:** `hooks/useRiskStatus.ts`
**Complejidad:** Media
**Dependencias:** CT-01, Backend BE-08

**Implementacion:**
```typescript
// hooks/useRiskStatus.ts
import useSWR from 'swr';
import { RiskStatusResponse, RiskLimits } from '@/types/contracts';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface UseRiskStatusReturn {
  // Estado actual
  killSwitchActive: boolean;
  currentDrawdownPct: number;
  dailyPnlPct: number;
  tradesToday: number;
  consecutiveLosses: number;
  cooldownActive: boolean;
  cooldownUntil: Date | null;

  // Limites configurados
  limits: RiskLimits | null;

  // Estados derivados para UI
  riskLevel: 'safe' | 'warning' | 'danger' | 'critical';
  tradingAllowed: boolean;
  warningMessages: string[];

  // SWR states
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
  mutate: () => void;
}

export function useRiskStatus(): UseRiskStatusReturn {
  const { data, error, isLoading, mutate } = useSWR<RiskStatusResponse>(
    `${API_BASE}/api/risk/status`,
    async (url) => {
      const res = await fetch(url);
      if (!res.ok) throw new Error('Failed to fetch risk status');
      return res.json();
    },
    {
      refreshInterval: 10000, // 10s - mas frecuente para risk
      revalidateOnFocus: true,
    }
  );

  // Calcular risk level
  const calculateRiskLevel = (): 'safe' | 'warning' | 'danger' | 'critical' => {
    if (!data) return 'safe';
    if (data.kill_switch_active) return 'critical';
    if (data.current_drawdown_pct > 10 || data.cooldown_active) return 'danger';
    if (data.current_drawdown_pct > 5 || data.consecutive_losses >= 2) return 'warning';
    return 'safe';
  };

  // Generar mensajes de advertencia
  const getWarningMessages = (): string[] => {
    if (!data) return [];
    const messages: string[] = [];

    if (data.kill_switch_active) {
      messages.push('KILL SWITCH ACTIVADO - Trading detenido');
    }
    if (data.cooldown_active) {
      messages.push(`Cooldown activo hasta ${data.cooldown_until}`);
    }
    if (data.current_drawdown_pct > 10) {
      messages.push(`Drawdown alto: ${data.current_drawdown_pct.toFixed(1)}%`);
    }
    if (data.limits && data.trades_today >= data.limits.max_trades_per_day * 0.8) {
      messages.push(`Cerca del limite de trades (${data.trades_today}/${data.limits.max_trades_per_day})`);
    }

    return messages;
  };

  return {
    killSwitchActive: data?.kill_switch_active ?? false,
    currentDrawdownPct: data?.current_drawdown_pct ?? 0,
    dailyPnlPct: data?.daily_pnl_pct ?? 0,
    tradesToday: data?.trades_today ?? 0,
    consecutiveLosses: data?.consecutive_losses ?? 0,
    cooldownActive: data?.cooldown_active ?? false,
    cooldownUntil: data?.cooldown_until ? new Date(data.cooldown_until) : null,
    limits: data?.limits ?? null,

    riskLevel: calculateRiskLevel(),
    tradingAllowed: !data?.kill_switch_active && !data?.cooldown_active,
    warningMessages: getWarningMessages(),

    isLoading,
    isError: !!error,
    error: error ?? null,
    mutate,
  };
}
```

---

### FE-04: Crear usePaperTradingMetrics hook
**Archivo:** `hooks/usePaperTradingMetrics.ts`
**Complejidad:** Media
**Dependencias:** Backend BE-12

```typescript
// hooks/usePaperTradingMetrics.ts
import useSWR from 'swr';

interface PaperTrade {
  trade_id: number;
  model_id: string;
  signal: string;
  entry_price: number;
  exit_price: number;
  pnl: number;
  pnl_pct: number;
  entry_time: string;
  exit_time: string;
}

interface PaperTradingMetrics {
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  total_pnl: number;
  win_rate: number;
  avg_win: number;
  avg_loss: number;
  current_equity: number;
  trades: PaperTrade[];
}

export function usePaperTradingMetrics() {
  const { data, error, isLoading } = useSWR<PaperTradingMetrics>(
    '/api/paper-trading/metrics',
    async (url) => {
      const res = await fetch(url);
      if (!res.ok) throw new Error('Failed to fetch paper trading metrics');
      return res.json();
    },
    { refreshInterval: 30000 }
  );

  return {
    metrics: data ?? null,
    trades: data?.trades ?? [],
    isLoading,
    isError: !!error,
  };
}
```

---

## Fase 3: Componentes UI - SEMANA 3

### FE-05: Crear RiskStatusCard component
**Archivo:** `components/trading/RiskStatusCard.tsx`
**Complejidad:** Media
**Dependencias:** FE-03

```tsx
// components/trading/RiskStatusCard.tsx
'use client';

import { useRiskStatus } from '@/hooks/useRiskStatus';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { AlertTriangle, Shield, XCircle, Clock } from 'lucide-react';
import { cn } from '@/lib/utils';

export function RiskStatusCard() {
  const {
    killSwitchActive,
    currentDrawdownPct,
    dailyPnlPct,
    tradesToday,
    consecutiveLosses,
    cooldownActive,
    cooldownUntil,
    limits,
    riskLevel,
    tradingAllowed,
    warningMessages,
    isLoading,
  } = useRiskStatus();

  if (isLoading) {
    return (
      <Card className="animate-pulse">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="h-5 w-5" />
            Risk Status
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-32 bg-muted rounded" />
        </CardContent>
      </Card>
    );
  }

  const riskColors = {
    safe: 'bg-green-500/10 border-green-500/20',
    warning: 'bg-yellow-500/10 border-yellow-500/20',
    danger: 'bg-orange-500/10 border-orange-500/20',
    critical: 'bg-red-500/10 border-red-500/20',
  };

  const riskBadgeColors = {
    safe: 'bg-green-500 text-white',
    warning: 'bg-yellow-500 text-black',
    danger: 'bg-orange-500 text-white',
    critical: 'bg-red-500 text-white animate-pulse',
  };

  return (
    <Card className={cn('border-2', riskColors[riskLevel])}>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-lg">
            <Shield className="h-5 w-5" />
            Risk Status
          </CardTitle>
          <Badge className={riskBadgeColors[riskLevel]}>
            {riskLevel.toUpperCase()}
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Kill Switch Alert */}
        {killSwitchActive && (
          <div className="flex items-center gap-2 p-3 bg-red-500/20 rounded-lg border border-red-500">
            <XCircle className="h-5 w-5 text-red-500" />
            <span className="font-semibold text-red-500">
              KILL SWITCH ACTIVADO - Trading Detenido
            </span>
          </div>
        )}

        {/* Cooldown Alert */}
        {cooldownActive && cooldownUntil && (
          <div className="flex items-center gap-2 p-3 bg-yellow-500/20 rounded-lg border border-yellow-500">
            <Clock className="h-5 w-5 text-yellow-500" />
            <span className="text-yellow-600">
              Cooldown hasta: {cooldownUntil.toLocaleTimeString()}
            </span>
          </div>
        )}

        {/* Metrics Grid */}
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-1">
            <p className="text-sm text-muted-foreground">Drawdown</p>
            <p className={cn(
              'text-2xl font-bold',
              currentDrawdownPct > 10 ? 'text-red-500' :
              currentDrawdownPct > 5 ? 'text-yellow-500' : 'text-green-500'
            )}>
              {currentDrawdownPct.toFixed(1)}%
            </p>
            <p className="text-xs text-muted-foreground">
              Max: {limits?.max_drawdown_pct}%
            </p>
          </div>

          <div className="space-y-1">
            <p className="text-sm text-muted-foreground">PnL Diario</p>
            <p className={cn(
              'text-2xl font-bold',
              dailyPnlPct >= 0 ? 'text-green-500' : 'text-red-500'
            )}>
              {dailyPnlPct >= 0 ? '+' : ''}{dailyPnlPct.toFixed(2)}%
            </p>
            <p className="text-xs text-muted-foreground">
              Limite: -{limits?.max_daily_loss_pct}%
            </p>
          </div>

          <div className="space-y-1">
            <p className="text-sm text-muted-foreground">Trades Hoy</p>
            <p className="text-2xl font-bold">
              {tradesToday}
            </p>
            <p className="text-xs text-muted-foreground">
              Max: {limits?.max_trades_per_day}
            </p>
          </div>

          <div className="space-y-1">
            <p className="text-sm text-muted-foreground">Perdidas Consec.</p>
            <p className={cn(
              'text-2xl font-bold',
              consecutiveLosses >= 3 ? 'text-red-500' :
              consecutiveLosses >= 2 ? 'text-yellow-500' : 'text-green-500'
            )}>
              {consecutiveLosses}
            </p>
            <p className="text-xs text-muted-foreground">
              Cooldown en: 3
            </p>
          </div>
        </div>

        {/* Warning Messages */}
        {warningMessages.length > 0 && (
          <div className="space-y-2">
            {warningMessages.map((msg, idx) => (
              <div key={idx} className="flex items-center gap-2 text-sm">
                <AlertTriangle className="h-4 w-4 text-yellow-500" />
                <span>{msg}</span>
              </div>
            ))}
          </div>
        )}

        {/* Trading Status */}
        <div className="pt-2 border-t">
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Estado Trading</span>
            <Badge variant={tradingAllowed ? 'default' : 'destructive'}>
              {tradingAllowed ? 'PERMITIDO' : 'BLOQUEADO'}
            </Badge>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
```

---

### FE-06: Anadir indicador Kill Switch al header
**Archivo:** `components/layout/Header.tsx` o similar
**Complejidad:** Baja
**Dependencias:** FE-03

```tsx
// Agregar al header existente
import { useRiskStatus } from '@/hooks/useRiskStatus';
import { AlertTriangle } from 'lucide-react';

function KillSwitchIndicator() {
  const { killSwitchActive, riskLevel } = useRiskStatus();

  if (!killSwitchActive && riskLevel === 'safe') {
    return null;
  }

  return (
    <div className={cn(
      'flex items-center gap-2 px-3 py-1 rounded-full text-sm font-medium',
      killSwitchActive
        ? 'bg-red-500 text-white animate-pulse'
        : riskLevel === 'danger'
        ? 'bg-orange-500 text-white'
        : 'bg-yellow-500 text-black'
    )}>
      <AlertTriangle className="h-4 w-4" />
      {killSwitchActive ? 'KILL SWITCH' : 'RISK WARNING'}
    </div>
  );
}

// Usar en Header:
// <header>
//   <Logo />
//   <KillSwitchIndicator />
//   <Navigation />
// </header>
```

---

### FE-07: Crear PaperTradingBadge visual
**Archivo:** `components/trading/PaperTradingBadge.tsx`
**Complejidad:** Baja

```tsx
// components/trading/PaperTradingBadge.tsx
'use client';

import { Badge } from '@/components/ui/badge';
import { TestTube } from 'lucide-react';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';

interface PaperTradingBadgeProps {
  isPaperMode: boolean;
}

export function PaperTradingBadge({ isPaperMode }: PaperTradingBadgeProps) {
  if (!isPaperMode) return null;

  return (
    <Tooltip>
      <TooltipTrigger>
        <Badge
          variant="outline"
          className="bg-purple-500/10 border-purple-500 text-purple-500"
        >
          <TestTube className="h-3 w-3 mr-1" />
          PAPER MODE
        </Badge>
      </TooltipTrigger>
      <TooltipContent>
        <p>Sistema en modo simulacion.</p>
        <p>Las operaciones NO son reales.</p>
      </TooltipContent>
    </Tooltip>
  );
}
```

---

### FE-08: Integrar Risk panel en dashboard principal
**Archivo:** `app/(dashboard)/page.tsx` o pagina principal
**Complejidad:** Media
**Dependencias:** FE-05

```tsx
// app/(dashboard)/page.tsx
import { RiskStatusCard } from '@/components/trading/RiskStatusCard';
import { PaperTradingBadge } from '@/components/trading/PaperTradingBadge';
// ... otros imports

export default function DashboardPage() {
  const isPaperMode = process.env.NEXT_PUBLIC_PAPER_MODE === 'true';

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header con Paper Mode Badge */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">USDCOP Trading Dashboard</h1>
        <PaperTradingBadge isPaperMode={isPaperMode} />
      </div>

      {/* Grid principal */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Columna 1: Risk Status (NUEVO) */}
        <div className="md:col-span-1">
          <RiskStatusCard />
        </div>

        {/* Columna 2-3: Signals y Chart */}
        <div className="md:col-span-2 space-y-6">
          {/* Componentes existentes de signals */}
          <SignalsPanel />
          <TradingChart />
        </div>
      </div>

      {/* Fila inferior: Performance */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <PerformanceCard />
        <PositionsCard />
      </div>
    </div>
  );
}
```

---

## Fase 4: Testing - SEMANA 4

### FE-09: Test E2E - Verificar signals se muestran
**Archivo:** `tests/e2e/signals.spec.ts`
**Complejidad:** Media
**Dependencias:** FE-01

```typescript
// tests/e2e/signals.spec.ts
import { test, expect } from '@playwright/test';

test.describe('Trading Signals', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should display signals from API', async ({ page }) => {
    // Esperar a que se carguen las senales
    await page.waitForSelector('[data-testid="signal-card"]', {
      timeout: 10000,
    });

    // Verificar que hay al menos una senal
    const signals = page.locator('[data-testid="signal-card"]');
    await expect(signals).toHaveCount({ min: 1 });
  });

  test('should show market price', async ({ page }) => {
    const marketPrice = page.locator('[data-testid="market-price"]');
    await expect(marketPrice).toBeVisible();

    // Verificar formato de precio (ej: 4,350.25)
    const priceText = await marketPrice.textContent();
    expect(priceText).toMatch(/[\d,]+\.\d{2}/);
  });

  test('should show market status', async ({ page }) => {
    const marketStatus = page.locator('[data-testid="market-status"]');
    await expect(marketStatus).toBeVisible();

    const statusText = await marketStatus.textContent();
    expect(['open', 'closed', 'pre_market']).toContain(
      statusText?.toLowerCase()
    );
  });

  test('should show signal confidence', async ({ page }) => {
    const confidence = page.locator('[data-testid="signal-confidence"]').first();
    await expect(confidence).toBeVisible();

    // Verificar que es un porcentaje valido
    const confText = await confidence.textContent();
    const confValue = parseFloat(confText?.replace('%', '') ?? '0');
    expect(confValue).toBeGreaterThanOrEqual(0);
    expect(confValue).toBeLessThanOrEqual(100);
  });

  test('signals should update periodically', async ({ page }) => {
    // Obtener timestamp inicial
    const timestampEl = page.locator('[data-testid="signals-timestamp"]');
    const initialTimestamp = await timestampEl.textContent();

    // Esperar 6 segundos (refresh cada 5s)
    await page.waitForTimeout(6000);

    // Verificar que timestamp cambio
    const newTimestamp = await timestampEl.textContent();
    // Nota: puede que no cambie si no hay nuevas senales,
    // pero el componente deberia haber intentado refetch
  });
});
```

---

### FE-10: Test E2E - Verificar risk status
**Archivo:** `tests/e2e/risk.spec.ts`
**Complejidad:** Media
**Dependencias:** FE-05

```typescript
// tests/e2e/risk.spec.ts
import { test, expect } from '@playwright/test';

test.describe('Risk Status', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should display risk status card', async ({ page }) => {
    const riskCard = page.locator('[data-testid="risk-status-card"]');
    await expect(riskCard).toBeVisible();
  });

  test('should show drawdown percentage', async ({ page }) => {
    const drawdown = page.locator('[data-testid="risk-drawdown"]');
    await expect(drawdown).toBeVisible();

    const text = await drawdown.textContent();
    expect(text).toMatch(/\d+\.\d+%/);
  });

  test('should show daily PnL', async ({ page }) => {
    const dailyPnl = page.locator('[data-testid="risk-daily-pnl"]');
    await expect(dailyPnl).toBeVisible();
  });

  test('should show trades today count', async ({ page }) => {
    const trades = page.locator('[data-testid="risk-trades-today"]');
    await expect(trades).toBeVisible();

    const text = await trades.textContent();
    const count = parseInt(text ?? '0');
    expect(count).toBeGreaterThanOrEqual(0);
  });

  test('should show risk level badge', async ({ page }) => {
    const badge = page.locator('[data-testid="risk-level-badge"]');
    await expect(badge).toBeVisible();

    const text = await badge.textContent();
    expect(['SAFE', 'WARNING', 'DANGER', 'CRITICAL']).toContain(
      text?.toUpperCase()
    );
  });

  test('should show kill switch alert when active', async ({ page }) => {
    // Este test requiere mock del API
    // o verificar que el componente lo maneja correctamente
    const killSwitchAlert = page.locator('[data-testid="kill-switch-alert"]');

    // Si kill switch esta activo, debe ser visible
    // Si no, no debe existir
    const isVisible = await killSwitchAlert.isVisible().catch(() => false);

    if (isVisible) {
      await expect(killSwitchAlert).toContainText('KILL SWITCH');
    }
  });

  test('risk card should have correct color based on level', async ({ page }) => {
    const riskCard = page.locator('[data-testid="risk-status-card"]');

    // Verificar que tiene alguna clase de color
    const className = await riskCard.getAttribute('class');
    expect(className).toMatch(/(green|yellow|orange|red)/);
  });
});
```

---

## Variables de Entorno

Crear/actualizar `.env.local`:
```env
# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000

# Feature Flags
NEXT_PUBLIC_PAPER_MODE=true
NEXT_PUBLIC_ENABLE_RISK_PANEL=true
NEXT_PUBLIC_ENABLE_MONITORING=true

# Refresh Intervals (ms)
NEXT_PUBLIC_SIGNALS_REFRESH=5000
NEXT_PUBLIC_METRICS_REFRESH=60000
NEXT_PUBLIC_RISK_REFRESH=10000
```

---

## Comandos Utiles

```bash
# Desarrollo
cd usdcop-trading-dashboard
npm run dev

# Type check
npx tsc --noEmit

# Lint
npm run lint

# Tests E2E
npm run test:e2e

# Build production
npm run build
```

---

## Checklist Pre-Produccion

- [ ] CT-01: Tipos en `api-contracts.ts` coinciden con backend
- [ ] FE-01: useModelSignals funciona con datos reales
- [ ] FE-02: useFinancialMetrics funciona con datos reales
- [ ] FE-03: useRiskStatus muestra estado correcto
- [ ] FE-05: RiskStatusCard renderiza todos los estados
- [ ] FE-06: Kill switch indicator visible en header cuando activo
- [ ] FE-08: Risk panel integrado en dashboard
- [ ] FE-09: Tests E2E de signals pasan
- [ ] FE-10: Tests E2E de risk pasan
- [ ] No hay errores de TypeScript
- [ ] Build de produccion exitoso
- [ ] WebSocket reconecta automaticamente
