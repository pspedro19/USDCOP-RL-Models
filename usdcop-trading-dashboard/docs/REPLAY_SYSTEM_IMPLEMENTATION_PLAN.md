# Sistema de Replay Dinámico - Plan de Implementación v2.0 (Production Ready)

## Resumen Ejecutivo

Este documento detalla el plan completo para implementar un **Sistema de Replay Dinámico** en el dashboard de trading USD/COP. El sistema permitirá reproducir visualmente las decisiones del modelo de trading en períodos de validación y test, sincronizando candlesticks, equity curve, métricas y tabla de trades.

**Versión 2.0** incluye mejoras de robustez para producción:
- Validación runtime con Zod schemas
- Branded Types para prevenir errores de tipos primitivos
- State Machine para transiciones predecibles
- Error handling con estrategias de recuperación
- API client con retry y backoff exponencial
- Performance monitoring con calidad adaptativa
- Keyboard shortcuts para accesibilidad

---

## Fechas del Modelo V20

| Período | Inicio | Fin | Duración | Propósito |
|---------|--------|-----|----------|-----------|
| **Entrenamiento** | 2020-01-01 | 2024-12-31 | 5 años | Aprendizaje del modelo |
| **Validación** | 2025-01-01 | 2025-06-30 | 6 meses | Tuning de hiperparámetros |
| **Test (OOS)** | 2025-07-01 | 2026-01-08 | ~7 meses | Evaluación real |

> **Regla fundamental:** El replay solo permite fechas desde **2025-01-01** en adelante (validación + test), ya que el período de training no debe evaluarse como "performance real".

---

## Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         REPLAY CONTROL BAR                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌─────┐ ┌─────┐ ┌─────┐ ┌───────────┐  │
│  │ Start Date ▼ │  │  End Date ▼  │  │ ▶️  │ │ ⏸️  │ │ ⏹️  │ │ Speed: 2x │  │
│  │ 2025-07-01   │  │  2025-12-31  │  │Play │ │Pause│ │Reset│ │ ▼         │  │
│  └──────────────┘  └──────────────┘  └─────┘ └─────┘ └─────┘ └───────────┘  │
│                                                                              │
│  ══════════════════════════●══════════════════════════════════════════════  │
│  Progress: 2025-09-15 (45%)              [Validation] [Test]                │
│                                                                              │
│  Keyboard: [Space] Play/Pause  [R] Reset  [↑↓] Speed  [1-8] Set Speed      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────────┐   ┌───────────────────────┐   ┌───────────────────────┐
│  CANDLESTICK      │   │   BACKTEST METRICS    │   │    EQUITY CURVE       │
│  CHART            │   │   (Actualiza live)    │   │  (Crece animado)      │
│  (Sincronizado)   │   │                       │   │                       │
│                   │   │ ┌─────┐┌─────┐┌─────┐ │   │ 📈────────●          │
│  📊 ────────●     │   │ │1.24 ││-8.5%││ 58% │ │   │          ↑           │
│            ↑      │   │ │Sharpe│Max DD│WinR  │ │   │   cursor replay      │
│   cursor   │      │   │ └─────┘└─────┘└─────┘ │   │                       │
└───────────────────┘   └───────────────────────┘   └───────────────────────┘
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │       TRADES TABLE            │
                    │   (Agrega en tiempo real)     │
                    │                               │
                    │ #127 │ LONG │ +$30 ← Nuevo!   │
                    │ #126 │SHORT │ +$25            │
                    │ #125 │ LONG │ -$12            │
                    └───────────────────────────────┘
```

---

## Dependencias Requeridas

### Nueva Dependencia (Requiere instalación)

```bash
npm install zod
```

### Dependencias Existentes (No requieren instalación)

| Dependencia | Versión | Uso |
|-------------|---------|-----|
| `zod` | ^3.x | **NUEVO** - Validación runtime de schemas |
| `framer-motion` | 12.23.12 | Animaciones de entrada |
| `recharts` | 3.1.2 | Equity curve chart |
| `lightweight-charts` | 5.0.8 | Candlestick chart |
| `lucide-react` | 0.511.0 | Iconos de controles |
| Tailwind CSS | 3.4.17 | Estilos y animaciones |

---

## Archivos a Crear (10 archivos nuevos)

### 1. `types/replay.ts` (~350 líneas) - SCHEMAS ZOD + BRANDED TYPES

**Propósito:** Contratos de tipos con validación runtime, branded types para prevenir errores.

```typescript
import { z } from 'zod';

// ═══════════════════════════════════════════════════════════════════════════
// CONSTANTES DEL MODELO - Single Source of Truth
// ═══════════════════════════════════════════════════════════════════════════

export const MODEL_CONFIG = {
  VERSION: 'v20',
  DATES: {
    TRAINING_START: '2020-01-01',
    TRAINING_END: '2024-12-31',
    VALIDATION_START: '2025-01-01',
    VALIDATION_END: '2025-06-30',
    TEST_START: '2025-07-01',
    TEST_END: '2026-01-08',
  },
  LIMITS: {
    MAX_TRADES_PER_LOAD: 10_000,
    MAX_CANDLES_PER_LOAD: 50_000,
    MAX_REPLAY_SPEED: 8,
    MIN_TICK_INTERVAL_MS: 16, // ~60fps cap
    MAX_DATE_RANGE_DAYS: 365,
  },
} as const;

// ═══════════════════════════════════════════════════════════════════════════
// BRANDED TYPES - Previenen errores de tipos primitivos
// ═══════════════════════════════════════════════════════════════════════════

declare const __brand: unique symbol;
type Brand<T, B> = T & { [__brand]: B };

export type ISODateString = Brand<string, 'ISODateString'>;
export type TradeId = Brand<string, 'TradeId'>;
export type ModelId = Brand<string, 'ModelId'>;
export type Percentage = Brand<number, 'Percentage'>; // 0-100
export type Ratio = Brand<number, 'Ratio'>;           // 0-1

// Constructores seguros
export const ISODateString = {
  parse: (value: string): ISODateString => {
    if (!/^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2}(\.\d{3})?Z?)?$/.test(value)) {
      throw new Error(`Invalid ISO date string: ${value}`);
    }
    return value as ISODateString;
  },
  fromDate: (date: Date): ISODateString => date.toISOString() as ISODateString,
  unsafe: (value: string): ISODateString => value as ISODateString,
};

export const Percentage = {
  fromRatio: (ratio: number): Percentage => {
    const pct = Math.min(100, Math.max(0, ratio * 100));
    return pct as Percentage;
  },
  clamp: (value: number): Percentage => Math.min(100, Math.max(0, value)) as Percentage,
};

// ═══════════════════════════════════════════════════════════════════════════
// ZOD SCHEMAS - Validación en runtime
// ═══════════════════════════════════════════════════════════════════════════

export const ReplaySpeedSchema = z.union([
  z.literal(1), z.literal(2), z.literal(4), z.literal(8),
]);
export type ReplaySpeed = z.infer<typeof ReplaySpeedSchema>;

export const ReplayModeSchema = z.enum(['validation', 'test', 'both']);
export type ReplayMode = z.infer<typeof ReplayModeSchema>;

export const ReplayStateSchema = z.object({
  startDate: z.date(),
  endDate: z.date(),
  currentDate: z.date(),
  isPlaying: z.boolean(),
  speed: ReplaySpeedSchema,
  mode: ReplayModeSchema,
  progress: z.number().min(0).max(100),
}).refine(
  (data) => data.startDate <= data.endDate,
  { message: 'startDate must be before or equal to endDate' }
).refine(
  (data) => data.currentDate >= data.startDate && data.currentDate <= data.endDate,
  { message: 'currentDate must be within date range' }
);
export type ReplayState = z.infer<typeof ReplayStateSchema>;

export const TradeSideSchema = z.enum(['LONG', 'SHORT']);
export type TradeSide = z.infer<typeof TradeSideSchema>;

export const TradeStatusSchema = z.enum(['OPEN', 'CLOSED', 'CANCELLED']);
export type TradeStatus = z.infer<typeof TradeStatusSchema>;

export const TradeSchema = z.object({
  trade_id: z.string().min(1),
  timestamp: z.string().datetime(),
  side: TradeSideSchema,
  entry_price: z.number().positive(),
  exit_price: z.number().positive().optional(),
  pnl: z.number(),
  pnl_percent: z.number(),
  hold_time_minutes: z.number().int().nonnegative(),
  status: TradeStatusSchema,
  confidence: z.number().min(0).max(1).optional(),
  _meta: z.object({
    model_version: z.string().optional(),
    signal_strength: z.number().optional(),
  }).optional(),
});
export type Trade = z.infer<typeof TradeSchema>;

export const EquityPointSchema = z.object({
  timestamp: z.string().datetime(),
  equity: z.number(),
  drawdown: z.number().max(0),
  cumulative_pnl: z.number(),
});
export type EquityPoint = z.infer<typeof EquityPointSchema>;

export const CandlestickSchema = z.object({
  time: z.number().int().positive(),
  open: z.number().positive(),
  high: z.number().positive(),
  low: z.number().positive(),
  close: z.number().positive(),
  volume: z.number().nonnegative().optional(),
}).refine(
  (data) => data.high >= data.low && data.high >= data.open && data.high >= data.close,
  { message: 'high must be highest value' }
).refine(
  (data) => data.low <= data.open && data.low <= data.close,
  { message: 'low must be lowest value' }
);
export type Candlestick = z.infer<typeof CandlestickSchema>;

export const ReplayMetricsSchema = z.object({
  sharpe_ratio: z.number(),
  max_drawdown: z.number().max(0),
  win_rate: z.number().min(0).max(100),
  avg_hold_time_minutes: z.number().nonnegative(),
  total_trades: z.number().int().nonnegative(),
  winning_trades: z.number().int().nonnegative(),
  losing_trades: z.number().int().nonnegative(),
  total_pnl: z.number(),
  profit_factor: z.number().nonnegative(),
  avg_win: z.number().optional(),
  avg_loss: z.number().optional(),
  largest_win: z.number().optional(),
  largest_loss: z.number().optional(),
  consecutive_wins: z.number().int().nonnegative().optional(),
  consecutive_losses: z.number().int().nonnegative().optional(),
});
export type ReplayMetrics = z.infer<typeof ReplayMetricsSchema>;

// ═══════════════════════════════════════════════════════════════════════════
// API RESPONSE SCHEMAS
// ═══════════════════════════════════════════════════════════════════════════

export const TradesResponseSchema = z.object({
  success: z.literal(true),
  data: z.object({
    trades: z.array(TradeSchema),
    total_count: z.number().int().nonnegative(),
    has_more: z.boolean(),
    next_cursor: z.string().optional(),
  }),
  meta: z.object({
    request_id: z.string(),
    timestamp: z.string().datetime(),
    duration_ms: z.number().nonnegative(),
  }),
});
export type TradesResponse = z.infer<typeof TradesResponseSchema>;

export const EquityCurveResponseSchema = z.object({
  success: z.literal(true),
  data: z.object({
    points: z.array(EquityPointSchema),
    summary: z.object({
      start_equity: z.number(),
      end_equity: z.number(),
      max_equity: z.number(),
      min_equity: z.number(),
    }),
  }),
});
export type EquityCurveResponse = z.infer<typeof EquityCurveResponseSchema>;

// ═══════════════════════════════════════════════════════════════════════════
// RESULT TYPE - Manejo explícito de errores
// ═══════════════════════════════════════════════════════════════════════════

export type Result<T, E = Error> =
  | { success: true; data: T }
  | { success: false; error: E };

export const Result = {
  ok: <T>(data: T): Result<T, never> => ({ success: true, data }),
  err: <E>(error: E): Result<never, E> => ({ success: false, error }),
  map: <T, U, E>(result: Result<T, E>, fn: (data: T) => U): Result<U, E> => {
    if (result.success) return Result.ok(fn(result.data));
    return result;
  },
  unwrapOr: <T, E>(result: Result<T, E>, defaultValue: T): T => {
    if (result.success) return result.data;
    return defaultValue;
  },
};

// ═══════════════════════════════════════════════════════════════════════════
// REPLAY EVENTS - Para logging/analytics
// ═══════════════════════════════════════════════════════════════════════════

export const ReplayEventSchema = z.discriminatedUnion('type', [
  z.object({
    type: z.literal('REPLAY_STARTED'),
    payload: z.object({
      startDate: z.string(),
      endDate: z.string(),
      mode: ReplayModeSchema,
      speed: ReplaySpeedSchema,
    }),
    timestamp: z.string().datetime(),
  }),
  z.object({
    type: z.literal('REPLAY_PAUSED'),
    payload: z.object({ currentDate: z.string(), progress: z.number() }),
    timestamp: z.string().datetime(),
  }),
  z.object({
    type: z.literal('REPLAY_COMPLETED'),
    payload: z.object({ duration_ms: z.number(), trades_replayed: z.number() }),
    timestamp: z.string().datetime(),
  }),
  z.object({
    type: z.literal('TRADE_HIGHLIGHTED'),
    payload: z.object({ trade_id: z.string(), side: TradeSideSchema, pnl: z.number() }),
    timestamp: z.string().datetime(),
  }),
  z.object({
    type: z.literal('REPLAY_ERROR'),
    payload: z.object({ error_code: z.string(), error_message: z.string() }),
    timestamp: z.string().datetime(),
  }),
]);
export type ReplayEvent = z.infer<typeof ReplayEventSchema>;
```

---

### 2. `utils/replayErrors.ts` (~150 líneas) - ERROR HANDLING ROBUSTO

**Propósito:** Tipos de error discriminados, estrategias de recuperación, error boundary hook.

```typescript
// ═══════════════════════════════════════════════════════════════════════════
// ERROR TYPES
// ═══════════════════════════════════════════════════════════════════════════

export class ReplayError extends Error {
  constructor(
    public readonly code: ReplayErrorCode,
    message: string,
    public readonly recoverable: boolean = true,
    public readonly context?: Record<string, unknown>
  ) {
    super(message);
    this.name = 'ReplayError';
  }
}

export enum ReplayErrorCode {
  INVALID_DATE_RANGE = 'INVALID_DATE_RANGE',
  DATA_LOAD_FAILED = 'DATA_LOAD_FAILED',
  DATA_VALIDATION_FAILED = 'DATA_VALIDATION_FAILED',
  NO_TRADES_IN_RANGE = 'NO_TRADES_IN_RANGE',
  INVALID_STATE_TRANSITION = 'INVALID_STATE_TRANSITION',
  ANIMATION_FRAME_ERROR = 'ANIMATION_FRAME_ERROR',
  TOO_MANY_DATA_POINTS = 'TOO_MANY_DATA_POINTS',
  RENDER_TIMEOUT = 'RENDER_TIMEOUT',
  API_TIMEOUT = 'API_TIMEOUT',
  API_ERROR = 'API_ERROR',
  NETWORK_OFFLINE = 'NETWORK_OFFLINE',
}

export const ERROR_MESSAGES: Record<ReplayErrorCode, string> = {
  [ReplayErrorCode.INVALID_DATE_RANGE]:
    'El rango de fechas seleccionado no es válido. Debe estar entre validación y test.',
  [ReplayErrorCode.DATA_LOAD_FAILED]:
    'No se pudieron cargar los datos del replay. Por favor, intente de nuevo.',
  [ReplayErrorCode.DATA_VALIDATION_FAILED]:
    'Los datos recibidos no tienen el formato esperado.',
  [ReplayErrorCode.NO_TRADES_IN_RANGE]:
    'No hay trades en el rango de fechas seleccionado.',
  [ReplayErrorCode.INVALID_STATE_TRANSITION]:
    'Operación no permitida en el estado actual del replay.',
  [ReplayErrorCode.ANIMATION_FRAME_ERROR]:
    'Error en la animación. El replay será pausado.',
  [ReplayErrorCode.TOO_MANY_DATA_POINTS]:
    'Demasiados datos para mostrar. Por favor, seleccione un rango más pequeño.',
  [ReplayErrorCode.RENDER_TIMEOUT]:
    'El renderizado está tardando demasiado. Reduciendo calidad visual.',
  [ReplayErrorCode.API_TIMEOUT]:
    'El servidor tardó demasiado en responder.',
  [ReplayErrorCode.API_ERROR]:
    'Error del servidor. Por favor, intente más tarde.',
  [ReplayErrorCode.NETWORK_OFFLINE]:
    'Sin conexión a internet. Verifique su conexión.',
};

// ═══════════════════════════════════════════════════════════════════════════
// RECOVERY STRATEGIES
// ═══════════════════════════════════════════════════════════════════════════

export interface RecoveryStrategy {
  action: 'retry' | 'reduce_range' | 'reduce_speed' | 'pause' | 'reset' | 'none';
  delay_ms?: number;
  message?: string;
}

export function getRecoveryStrategy(error: ReplayError): RecoveryStrategy {
  switch (error.code) {
    case ReplayErrorCode.API_TIMEOUT:
    case ReplayErrorCode.DATA_LOAD_FAILED:
      return { action: 'retry', delay_ms: 2000, message: 'Reintentando...' };
    case ReplayErrorCode.TOO_MANY_DATA_POINTS:
      return { action: 'reduce_range', message: 'Reduciendo rango de fechas...' };
    case ReplayErrorCode.RENDER_TIMEOUT:
      return { action: 'reduce_speed', message: 'Reduciendo velocidad...' };
    case ReplayErrorCode.ANIMATION_FRAME_ERROR:
      return { action: 'pause', message: 'Replay pausado debido a un error.' };
    case ReplayErrorCode.NETWORK_OFFLINE:
      return { action: 'none', message: 'Esperando conexión...' };
    default:
      return { action: 'reset' };
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// ERROR BOUNDARY HOOK
// ═══════════════════════════════════════════════════════════════════════════

export function useReplayError() {
  const [error, setErrorState] = useState<ReplayError | null>(null);
  const [recovery, setRecovery] = useState<RecoveryStrategy | null>(null);

  const handleError = useCallback((err: unknown) => {
    let replayError: ReplayError;

    if (err instanceof ReplayError) {
      replayError = err;
    } else if (err instanceof Error) {
      if (err.message.includes('network') || err.message.includes('fetch')) {
        replayError = new ReplayError(
          ReplayErrorCode.NETWORK_OFFLINE,
          ERROR_MESSAGES[ReplayErrorCode.NETWORK_OFFLINE],
          true
        );
      } else {
        replayError = new ReplayError(
          ReplayErrorCode.API_ERROR,
          err.message,
          true,
          { originalError: err.name }
        );
      }
    } else {
      replayError = new ReplayError(ReplayErrorCode.API_ERROR, 'Unknown error', true);
    }

    setErrorState(replayError);
    setRecovery(getRecoveryStrategy(replayError));
    console.error('[ReplayError]', replayError);
  }, []);

  const clearError = useCallback(() => {
    setErrorState(null);
    setRecovery(null);
  }, []);

  return { error, hasError: error !== null, handleError, clearError, recovery };
}
```

---

### 3. `hooks/useReplayStateMachine.ts` (~250 líneas) - STATE MACHINE

**Propósito:** Estado predecible con transiciones explícitas, previene estados inválidos.

```typescript
import { useReducer, useCallback } from 'react';
import { ReplaySpeed, ReplayMode, MODEL_CONFIG } from '@/types/replay';

// ═══════════════════════════════════════════════════════════════════════════
// TIPOS DE ESTADO (Discriminated Union)
// ═══════════════════════════════════════════════════════════════════════════

type ReplayStatus = 'idle' | 'loading' | 'ready' | 'playing' | 'paused' | 'completed' | 'error';

interface ReplayMachineState {
  status: ReplayStatus;
  startDate: Date;
  endDate: Date;
  currentDate: Date;
  speed: ReplaySpeed;
  mode: ReplayMode;
  progress: number;
  error: string | null;
  _lastTickTime: number;
  _tickCount: number;
}

// ═══════════════════════════════════════════════════════════════════════════
// ACCIONES
// ═══════════════════════════════════════════════════════════════════════════

type ReplayAction =
  | { type: 'LOAD_DATA' }
  | { type: 'DATA_LOADED' }
  | { type: 'PLAY' }
  | { type: 'PAUSE' }
  | { type: 'RESET' }
  | { type: 'TICK'; payload: { nextDate: Date } }
  | { type: 'SEEK'; payload: { targetDate: Date } }
  | { type: 'SET_SPEED'; payload: { speed: ReplaySpeed } }
  | { type: 'SET_MODE'; payload: { mode: ReplayMode } }
  | { type: 'SET_DATE_RANGE'; payload: { startDate: Date; endDate: Date } }
  | { type: 'COMPLETE' }
  | { type: 'ERROR'; payload: { message: string } }
  | { type: 'CLEAR_ERROR' };

// ═══════════════════════════════════════════════════════════════════════════
// TRANSICIONES VÁLIDAS
// ═══════════════════════════════════════════════════════════════════════════

const VALID_TRANSITIONS: Record<ReplayStatus, ReplayAction['type'][]> = {
  idle: ['LOAD_DATA', 'ERROR'],
  loading: ['DATA_LOADED', 'ERROR'],
  ready: ['PLAY', 'SET_DATE_RANGE', 'SET_MODE', 'SET_SPEED', 'LOAD_DATA', 'ERROR'],
  playing: ['PAUSE', 'TICK', 'SEEK', 'SET_SPEED', 'COMPLETE', 'ERROR'],
  paused: ['PLAY', 'RESET', 'SEEK', 'SET_DATE_RANGE', 'SET_MODE', 'SET_SPEED', 'ERROR'],
  completed: ['RESET', 'SET_DATE_RANGE', 'SET_MODE', 'ERROR'],
  error: ['CLEAR_ERROR', 'RESET', 'LOAD_DATA'],
};

function replayReducer(state: ReplayMachineState, action: ReplayAction): ReplayMachineState {
  const validActions = VALID_TRANSITIONS[state.status];
  if (!validActions.includes(action.type)) {
    console.warn(`[ReplayMachine] Invalid transition: ${action.type} from ${state.status}`);
    return state;
  }
  // ... reducer implementation
}

export function useReplayStateMachine(): [ReplayMachineState, ReplayMachineActions] {
  const [state, dispatch] = useReducer(replayReducer, initialState);

  const actions: ReplayMachineActions = {
    loadData: useCallback(() => dispatch({ type: 'LOAD_DATA' }), []),
    dataLoaded: useCallback(() => dispatch({ type: 'DATA_LOADED' }), []),
    play: useCallback(() => dispatch({ type: 'PLAY' }), []),
    pause: useCallback(() => dispatch({ type: 'PAUSE' }), []),
    reset: useCallback(() => dispatch({ type: 'RESET' }), []),
    tick: useCallback((nextDate: Date) => dispatch({ type: 'TICK', payload: { nextDate } }), []),
    seek: useCallback((targetDate: Date) => dispatch({ type: 'SEEK', payload: { targetDate } }), []),
    setSpeed: useCallback((speed: ReplaySpeed) => dispatch({ type: 'SET_SPEED', payload: { speed } }), []),
    setMode: useCallback((mode: ReplayMode) => dispatch({ type: 'SET_MODE', payload: { mode } }), []),
    setDateRange: useCallback((startDate: Date, endDate: Date) =>
      dispatch({ type: 'SET_DATE_RANGE', payload: { startDate, endDate } }), []),
    clearError: useCallback(() => dispatch({ type: 'CLEAR_ERROR' }), []),
  };

  return [state, actions];
}
```

---

### 4. `lib/replayApiClient.ts` (~200 líneas) - API CLIENT CON RETRY

**Propósito:** Fetch con retry, backoff exponencial, timeout, validación Zod.

```typescript
import { z } from 'zod';
import { TradesResponseSchema, EquityCurveResponseSchema, Result, MODEL_CONFIG } from '@/types/replay';
import { ReplayError, ReplayErrorCode, ERROR_MESSAGES } from '@/utils/replayErrors';

const API_CONFIG = {
  BASE_URL: process.env.NEXT_PUBLIC_API_URL || '',
  TIMEOUT_MS: 30_000,
  MAX_RETRIES: 3,
  RETRY_DELAY_MS: 1000,
  RETRY_BACKOFF: 2,
};

async function fetchWithRetry<T>(
  url: string,
  schema: z.ZodSchema<T>,
  options: { retries?: number; timeout?: number; signal?: AbortSignal } = {}
): Promise<Result<T, ReplayError>> {
  const { retries = API_CONFIG.MAX_RETRIES, timeout = API_CONFIG.TIMEOUT_MS } = options;
  let lastError: ReplayError | null = null;
  let delay = API_CONFIG.RETRY_DELAY_MS;

  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeout);

      const response = await fetch(url, {
        signal: controller.signal,
        headers: { 'Content-Type': 'application/json', 'X-Request-ID': crypto.randomUUID() },
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new ReplayError(
          ReplayErrorCode.API_ERROR,
          `HTTP ${response.status}`,
          response.status >= 500,
          { status: response.status, url }
        );
      }

      const data = await response.json();
      const parsed = schema.safeParse(data);

      if (!parsed.success) {
        throw new ReplayError(
          ReplayErrorCode.DATA_VALIDATION_FAILED,
          `Invalid response: ${parsed.error.message}`,
          false,
          { zodError: parsed.error.flatten() }
        );
      }

      return Result.ok(parsed.data);
    } catch (error) {
      // Handle errors and retry logic...
      if (attempt < retries) {
        await new Promise(resolve => setTimeout(resolve, delay));
        delay *= API_CONFIG.RETRY_BACKOFF;
      }
    }
  }

  return Result.err(lastError || new ReplayError(ReplayErrorCode.DATA_LOAD_FAILED, 'Failed', false));
}

export async function fetchReplayData(params: FetchReplayDataParams): Promise<Result<ReplayData, ReplayError>> {
  // Validate inputs, fetch in parallel, return combined result
}
```

---

### 5. `utils/replayMetrics.ts` (~200 líneas) - CÁLCULO PRECISO DE MÉTRICAS

**Propósito:** Fórmulas precisas con anualización correcta, calculador incremental.

```typescript
import { Trade, EquityPoint, ReplayMetrics } from '@/types/replay';

const TRADING_DAYS_PER_YEAR = 252;
const RISK_FREE_RATE_ANNUAL = 0.05;

function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((sum, v) => sum + v, 0) / values.length;
}

function standardDeviation(values: number[]): number {
  if (values.length < 2) return 0;
  const avg = mean(values);
  const squaredDiffs = values.map(v => Math.pow(v - avg, 2));
  return Math.sqrt(mean(squaredDiffs));
}

export function calculateReplayMetrics(trades: Trade[], equityCurve: EquityPoint[]): ReplayMetrics {
  if (trades.length === 0) return emptyMetrics;

  const wins = trades.filter(t => t.pnl > 0);
  const losses = trades.filter(t => t.pnl < 0);
  const pnls = trades.map(t => t.pnl);
  const totalPnL = pnls.reduce((sum, p) => sum + p, 0);
  const grossProfit = wins.reduce((sum, t) => sum + t.pnl, 0);
  const grossLoss = Math.abs(losses.reduce((sum, t) => sum + t.pnl, 0));

  // Sharpe Ratio (annualized)
  const avgReturn = mean(pnls);
  const stdDev = standardDeviation(pnls);
  const avgHoldDays = mean(trades.map(t => t.hold_time_minutes)) / 1440;
  const tradesPerYear = avgHoldDays > 0 ? TRADING_DAYS_PER_YEAR / avgHoldDays : TRADING_DAYS_PER_YEAR;
  const sharpeRatio = stdDev > 0
    ? ((avgReturn - (RISK_FREE_RATE_ANNUAL / tradesPerYear)) / stdDev) * Math.sqrt(tradesPerYear)
    : 0;

  // Max Drawdown from equity curve
  let maxDrawdown = 0;
  if (equityCurve.length > 0) {
    let peak = equityCurve[0].equity;
    for (const point of equityCurve) {
      if (point.equity > peak) peak = point.equity;
      const drawdown = (peak - point.equity) / peak;
      maxDrawdown = Math.max(maxDrawdown, drawdown);
    }
  }

  return {
    sharpe_ratio: Number(sharpeRatio.toFixed(3)),
    max_drawdown: Number((-maxDrawdown * 100).toFixed(2)),
    win_rate: Number(((wins.length / trades.length) * 100).toFixed(2)),
    // ... rest of metrics
  };
}

// Calculador incremental para replay en tiempo real
export class IncrementalMetricsCalculator {
  private trades: Trade[] = [];
  private equityPoints: EquityPoint[] = [];
  private cachedMetrics: ReplayMetrics | null = null;

  addTrade(trade: Trade): void {
    this.trades.push(trade);
    this.cachedMetrics = null;
  }

  getMetrics(): ReplayMetrics {
    if (this.cachedMetrics) return this.cachedMetrics;
    this.cachedMetrics = calculateReplayMetrics(this.trades, this.equityPoints);
    return this.cachedMetrics;
  }

  reset(): void {
    this.trades = [];
    this.equityPoints = [];
    this.cachedMetrics = null;
  }
}
```

---

### 6. `hooks/useReplayKeyboard.ts` (~80 líneas) - KEYBOARD SHORTCUTS

**Propósito:** Accesibilidad con atajos de teclado.

```typescript
import { useEffect, useCallback } from 'react';
import { ReplayMachineActions, ReplaySpeed } from './useReplayStateMachine';

const SPEED_SEQUENCE: ReplaySpeed[] = [1, 2, 4, 8];

export function useReplayKeyboard({
  enabled = true,
  actions,
  currentSpeed,
  isPlaying,
}: UseReplayKeyboardOptions): void {

  const handleKeyDown = useCallback((event: KeyboardEvent) => {
    if (event.target instanceof HTMLInputElement) return;

    switch (event.key) {
      case ' ': // Space - Play/Pause
        event.preventDefault();
        isPlaying ? actions.pause() : actions.play();
        break;
      case 'r':
      case 'R':
        if (!event.metaKey && !event.ctrlKey) {
          event.preventDefault();
          actions.reset();
        }
        break;
      case 'ArrowUp':
      case '+':
        event.preventDefault();
        const nextUp = SPEED_SEQUENCE[Math.min(SPEED_SEQUENCE.indexOf(currentSpeed) + 1, 3)];
        actions.setSpeed(nextUp);
        break;
      case 'ArrowDown':
      case '-':
        event.preventDefault();
        const nextDown = SPEED_SEQUENCE[Math.max(SPEED_SEQUENCE.indexOf(currentSpeed) - 1, 0)];
        actions.setSpeed(nextDown);
        break;
      case '1': case '2': case '4': case '8':
        event.preventDefault();
        actions.setSpeed(parseInt(event.key) as ReplaySpeed);
        break;
      case 'Escape':
        event.preventDefault();
        actions.pause();
        break;
    }
  }, [actions, currentSpeed, isPlaying]);

  useEffect(() => {
    if (!enabled) return;
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [enabled, handleKeyDown]);
}

export function KeyboardShortcutsHelp(): JSX.Element {
  return (
    <div className="text-xs text-gray-500 space-y-1">
      <div><kbd className="px-1 bg-gray-700 rounded">Space</kbd> Play/Pause</div>
      <div><kbd className="px-1 bg-gray-700 rounded">R</kbd> Reset</div>
      <div><kbd className="px-1 bg-gray-700 rounded">↑</kbd>/<kbd className="px-1 bg-gray-700 rounded">↓</kbd> Speed</div>
      <div><kbd className="px-1 bg-gray-700 rounded">1-8</kbd> Set speed</div>
      <div><kbd className="px-1 bg-gray-700 rounded">Esc</kbd> Pause</div>
    </div>
  );
}
```

---

### 7. `utils/replayPerformance.ts` (~120 líneas) - PERFORMANCE MONITORING

**Propósito:** Monitoreo de rendimiento, calidad adaptativa.

```typescript
export const PERF_THRESHOLDS = {
  FRAME_BUDGET_MS: 16.67,    // 60fps target
  FRAME_WARNING_MS: 33.33,   // 30fps warning
  FRAME_CRITICAL_MS: 100,    // Very slow
};

class ReplayPerformanceMonitor {
  private samples: PerformanceSample[] = [];

  measure<T>(operation: string, fn: () => T): T {
    const start = performance.now();
    const result = fn();
    const duration = performance.now() - start;
    this.addSample({ timestamp: Date.now(), duration, operation });
    this.checkThresholds(operation, duration);
    return result;
  }

  getStats(): PerformanceStats {
    const frameSamples = this.samples.filter(s => s.operation === 'frame').map(s => s.duration);
    if (frameSamples.length === 0) return { avgFrameTime: 0, maxFrameTime: 0, p95FrameTime: 0 };

    const sorted = [...frameSamples].sort((a, b) => a - b);
    return {
      avgFrameTime: frameSamples.reduce((a, b) => a + b, 0) / frameSamples.length,
      maxFrameTime: Math.max(...frameSamples),
      p95FrameTime: sorted[Math.floor(sorted.length * 0.95)],
    };
  }
}

export function getAdaptiveQuality(avgFrameTime: number): QualitySettings {
  if (avgFrameTime < PERF_THRESHOLDS.FRAME_BUDGET_MS) {
    return { maxVisibleCandles: 500, animationEnabled: true, highlightDuration: 2000 };
  }
  if (avgFrameTime < PERF_THRESHOLDS.FRAME_WARNING_MS) {
    return { maxVisibleCandles: 300, animationEnabled: true, highlightDuration: 1500 };
  }
  return { maxVisibleCandles: 150, animationEnabled: false, highlightDuration: 1000 };
}

export const replayPerfMonitor = new ReplayPerformanceMonitor();
```

---

### 8. `hooks/useReplayAnimation.ts` (~180 líneas)

**Propósito:** Lógica de animación con performance monitoring integrado.

---

### 9. `components/trading/ReplayControlBar.tsx` (~250 líneas)

**Propósito:** UI de controles con keyboard shortcuts helper y error display.

---

### 10. `__tests__/replay/fixtures.ts` (~150 líneas) - TEST FIXTURES

**Propósito:** Factories, generators, edge cases para testing.

```typescript
export function createMockTrade(overrides: Partial<Trade> = {}): Trade {
  return {
    trade_id: `trade_${++tradeIdCounter}`,
    timestamp: new Date('2025-07-15T10:30:00Z').toISOString(),
    side: 'LONG',
    entry_price: 4250.50,
    exit_price: 4280.25,
    pnl: 29.75,
    pnl_percent: 0.70,
    hold_time_minutes: 45,
    status: 'CLOSED',
    ...overrides,
  };
}

export function generateTradeSeries(count: number, options = {}): Trade[] {
  // Generate realistic trade sequences
}

export const EDGE_CASES = {
  emptyTrades: [] as Trade[],
  singleWinningTrade: [createMockTrade({ pnl: 100 })],
  singleLosingTrade: [createMockTrade({ pnl: -50 })],
  allWinners: generateTradeSeries(10, { winRate: 1.0 }),
  allLosers: generateTradeSeries(10, { winRate: 0.0 }),
  highFrequency: generateTradeSeries(100, { avgHoldMinutes: 5 }),
};
```

---

## Archivos a Modificar (7 archivos existentes)

### 1-7. (Sin cambios respecto a la versión anterior)

Los archivos a modificar permanecen iguales:
- `app/api/models/[modelId]/metrics/route.ts` - +from/to params
- `app/api/models/[modelId]/equity-curve/route.ts` - +from/to params
- `app/api/trading/trades/history/route.ts` - +from/to params
- `app/dashboard/page.tsx` - Integración completa
- `components/trading/TradesTable.tsx` - Highlight animation
- `components/charts/TradingChartWithSignals.tsx` - endDate prop
- `tailwind.config.ts` - Animación highlight

---

## Resumen de Impacto v2.0

| Categoría | Archivos | Líneas Estimadas |
|-----------|----------|------------------|
| **Nuevos** | 10 archivos | ~1,730 líneas |
| **Modificados** | 7 archivos | ~145 líneas |
| **Total** | 17 archivos | ~1,875 líneas |

---

## Comparación: Propuesta Original vs v2.0

| Área | Propuesta Original | v2.0 Production Ready |
|------|-------------------|------------------------|
| **Tipos** | Solo TypeScript compile-time | Zod schemas + Branded Types + Runtime validation |
| **Estado** | useState simple | State Machine con transiciones explícitas |
| **Errores** | No especificado | Error types + Recovery strategies + Error boundaries |
| **API** | Fetch básico | Retry con backoff + Timeout + Validation |
| **Métricas** | Cálculo básico | Fórmulas precisas + Incremental calculator |
| **Testing** | Plan general | Fixtures + Factories + Edge cases |
| **Accesibilidad** | No especificado | Keyboard shortcuts + ARIA |
| **Performance** | No especificado | Monitoring + Adaptive quality |

---

## Checklist de Implementación

### Fase 0: Setup (Antes de empezar)
- [ ] Instalar zod: `npm install zod`
- [ ] Crear estructura de carpetas
- [ ] Configurar paths en tsconfig.json

### Fase 1: Tipos y Contratos
- [ ] `types/replay.ts` - Schemas Zod + Branded Types
- [ ] `utils/replayErrors.ts` - Error types + recovery
- [ ] `__tests__/replay/fixtures.ts` - Factories + edge cases

### Fase 2: Core Logic
- [ ] `hooks/useReplayStateMachine.ts` - State machine
- [ ] `utils/replayMetrics.ts` - Cálculo de métricas
- [ ] `lib/replayApiClient.ts` - API client con retry

### Fase 3: UI y Accesibilidad
- [ ] `components/trading/ReplayControlBar.tsx`
- [ ] `hooks/useReplayKeyboard.ts` - Shortcuts
- [ ] Actualizar `tailwind.config.ts`

### Fase 4: Performance
- [ ] `utils/replayPerformance.ts` - Monitoring
- [ ] Implementar adaptive quality

### Fase 5: Integración
- [ ] Modificar APIs (from/to params)
- [ ] Integrar en dashboard
- [ ] E2E testing

### Fase 6: Documentación
- [ ] README del sistema de replay
- [ ] Storybook stories (opcional)

---

## Plan de Testing Detallado

### Unit Tests
```
__tests__/
├── replay/
│   ├── fixtures.ts              # Factories + edge cases
│   ├── types.test.ts            # Zod schema validation
│   ├── errors.test.ts           # Error handling
│   ├── metrics.test.ts          # Metrics calculation
│   └── stateMachine.test.ts     # State transitions
```

### Integration Tests
```
tests/integration/
└── replay-workflow.test.ts      # Complete replay flow
```

### E2E Tests
```
tests/e2e/
└── replay.spec.ts               # Playwright tests
```

---

## Notas de Backward Compatibility

- Todos los parámetros de API son **opcionales**
- Sin `from/to`, las APIs mantienen comportamiento actual
- Componentes sin props de replay funcionan igual
- No se requieren migraciones de datos
- Zod validation es opt-in (puede degradarse gracefully)

---

## Autor y Versión

- **Generado por:** Claude Opus 4.5 con análisis de 10 agentes paralelos
- **Fecha:** 2026-01-10
- **Versión:** 2.0 - Production Ready
- **Changelog:**
  - v1.0: Plan inicial básico
  - v2.0: Añadido robustez para producción (Zod, State Machine, Error Handling, Performance Monitoring)
