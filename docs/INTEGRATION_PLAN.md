# Plan de Integración: SignalBridge → USDCOP Trading Dashboard

## Resumen Ejecutivo

Este documento presenta el plan profesional para integrar el módulo **SignalBridge** (plataforma de ejecución de trading automatizado) al **USDCOP Trading Dashboard** existente. La integración sigue principios de arquitectura empresarial: **SSOT**, **DRY**, **Contract-Driven Development**, y patrones de diseño probados.

---

## 1. Análisis de Compatibilidad

### 1.1 Stack Tecnológico Comparativo

| Aspecto | Dashboard Actual | SignalBridge | Acción |
|---------|------------------|--------------|--------|
| **Framework** | Next.js 15 (App Router) | Vite + React 18 | ⚠️ Migrar a Next.js App Router |
| **TypeScript** | 5.x | 5.3.3 | ✓ Compatible |
| **Styling** | Tailwind CSS | Tailwind CSS (tema terminal) | 🔄 Unificar temas |
| **State Management** | React Context + Hooks | Zustand + React Query | ⚠️ Consolidar en Zustand |
| **Validation** | Zod | Zod | ✓ Unificar contratos |
| **API Client** | Axios + SWR | Axios + React Query | 🔄 Unificar en React Query |
| **UI Components** | Radix + ShadCN | ShadCN | ✓ Compatible |
| **Icons** | Lucide | Lucide | ✓ Compatible |
| **Animations** | Framer Motion | Framer Motion | ✓ Compatible |

### 1.2 Incompatibilidades Críticas

1. **Routing**: SignalBridge usa `react-router-dom`, Dashboard usa Next.js App Router
2. **State Management**: Dashboard usa Context API, SignalBridge usa Zustand
3. **Temas**: Dashboard tiene tema oscuro genérico, SignalBridge tiene tema "terminal"
4. **Auth**: Dashboard usa NextAuth, SignalBridge tiene auth custom con JWT

---

## 2. Arquitectura de Contratos Unificados (SSOT)

### 2.1 Estructura Propuesta de Contratos

```
lib/contracts/
├── core/                          # Contratos fundamentales (SSOT)
│   ├── ssot.contract.ts           # ✓ Existente - Feature order, actions
│   ├── auth.contract.ts           # 🆕 Unificar auth contracts
│   └── api-response.contract.ts   # 🆕 Generic API wrapper
│
├── trading/                       # Dominio Trading (existente)
│   ├── model.contract.ts          # ✓ Existente
│   ├── backtest.contract.ts       # ✓ Existente
│   └── signal.contract.ts         # 🔄 Extender con SignalBridge
│
├── execution/                     # 🆕 Dominio Ejecución (SignalBridge)
│   ├── exchange.contract.ts       # Exchanges conectados
│   ├── execution.contract.ts      # Órdenes ejecutadas
│   └── trading-config.contract.ts # Configuración de trading
│
└── shared/                        # 🆕 Tipos compartidos
    ├── pagination.contract.ts     # Paginación genérica
    ├── timestamp.contract.ts      # ISO dates, branded types
    └── money.contract.ts          # Currency, amounts
```

### 2.2 Contrato Auth Unificado

```typescript
// lib/contracts/core/auth.contract.ts
import { z } from 'zod';

// ============================================
// SSOT: Authentication Contract v1.0.0
// Last sync: [DATE]
// ============================================

export const SubscriptionTierSchema = z.enum(['FREE', 'PRO', 'ENTERPRISE']);
export type SubscriptionTier = z.infer<typeof SubscriptionTierSchema>;

export const UserSchema = z.object({
  id: z.string().uuid(),
  email: z.string().email(),
  name: z.string().optional(),
  subscription_tier: SubscriptionTierSchema,
  trading_enabled: z.boolean(),
  created_at: z.string().datetime(),
  updated_at: z.string().datetime(),
});
export type User = z.infer<typeof UserSchema>;

export const AuthTokensSchema = z.object({
  access_token: z.string(),
  refresh_token: z.string(),
  token_type: z.literal('Bearer'),
  expires_in: z.number().int().positive(),
});
export type AuthTokens = z.infer<typeof AuthTokensSchema>;

export const LoginRequestSchema = z.object({
  email: z.string().email(),
  password: z.string().min(8),
});
export type LoginRequest = z.infer<typeof LoginRequestSchema>;

export const RegisterRequestSchema = LoginRequestSchema.extend({
  name: z.string().min(2).optional(),
  confirm_password: z.string(),
}).refine((data) => data.password === data.confirm_password, {
  message: 'Passwords do not match',
  path: ['confirm_password'],
});
export type RegisterRequest = z.infer<typeof RegisterRequestSchema>;

// Validation helpers
export const validateLoginRequest = (data: unknown) => LoginRequestSchema.safeParse(data);
export const validateUser = (data: unknown) => UserSchema.safeParse(data);
```

### 2.3 Contrato Exchange Unificado

```typescript
// lib/contracts/execution/exchange.contract.ts
import { z } from 'zod';

// ============================================
// SSOT: Exchange Contract v1.0.0
// Mirrors backend: backend/app/contracts/exchange.py
// ============================================

export const SupportedExchangeSchema = z.enum(['BINANCE', 'MEXC']);
export type SupportedExchange = z.infer<typeof SupportedExchangeSchema>;

export const ExchangeStatusSchema = z.enum(['CONNECTED', 'DISCONNECTED', 'ERROR', 'VALIDATING']);
export type ExchangeStatus = z.infer<typeof ExchangeStatusSchema>;

export const ConnectedExchangeSchema = z.object({
  id: z.string().uuid(),
  exchange: SupportedExchangeSchema,
  status: ExchangeStatusSchema,
  label: z.string().optional(),
  is_testnet: z.boolean(),
  last_validated: z.string().datetime().optional(),
  created_at: z.string().datetime(),
  // NOTE: API keys are NEVER exposed in frontend contracts
});
export type ConnectedExchange = z.infer<typeof ConnectedExchangeSchema>;

export const ConnectExchangeRequestSchema = z.object({
  exchange: SupportedExchangeSchema,
  api_key: z.string().min(10),
  api_secret: z.string().min(10),
  label: z.string().optional(),
  is_testnet: z.boolean().default(false),
});
export type ConnectExchangeRequest = z.infer<typeof ConnectExchangeRequestSchema>;

export const ExchangeBalanceSchema = z.object({
  exchange: SupportedExchangeSchema,
  asset: z.string(),
  free: z.number().nonnegative(),
  locked: z.number().nonnegative(),
  total: z.number().nonnegative(),
  usd_value: z.number().optional(),
});
export type ExchangeBalance = z.infer<typeof ExchangeBalanceSchema>;

// Exchange metadata (for UI display)
export const EXCHANGE_METADATA: Record<SupportedExchange, {
  name: string;
  logo: string;
  color: string;
  docsUrl: string;
}> = {
  BINANCE: {
    name: 'Binance',
    logo: '/exchanges/binance.svg',
    color: '#F0B90B',
    docsUrl: 'https://binance-docs.github.io/apidocs/',
  },
  MEXC: {
    name: 'MEXC',
    logo: '/exchanges/mexc.svg',
    color: '#00B897',
    docsUrl: 'https://mexcdevelop.github.io/apidocs/',
  },
};
```

### 2.4 Contrato Execution Unificado

```typescript
// lib/contracts/execution/execution.contract.ts
import { z } from 'zod';
import { SupportedExchangeSchema } from './exchange.contract';
import { Action, ACTION_NAMES } from '../core/ssot.contract';

// ============================================
// SSOT: Execution Contract v1.0.0
// Mirrors backend: backend/app/contracts/execution.py
// ============================================

export const ExecutionStatusSchema = z.enum([
  'PENDING',
  'SUBMITTED',
  'PARTIALLY_FILLED',
  'FILLED',
  'CANCELLED',
  'REJECTED',
  'EXPIRED',
]);
export type ExecutionStatus = z.infer<typeof ExecutionStatusSchema>;

export const ExecutionSideSchema = z.enum(['BUY', 'SELL']);
export type ExecutionSide = z.infer<typeof ExecutionSideSchema>;

export const ExecutionSchema = z.object({
  id: z.string().uuid(),
  signal_id: z.string().uuid(),
  exchange: SupportedExchangeSchema,
  symbol: z.string(),
  side: ExecutionSideSchema,
  quantity: z.number().positive(),
  price: z.number().positive().optional(), // Market orders may not have price
  status: ExecutionStatusSchema,
  exchange_order_id: z.string().optional(),
  filled_quantity: z.number().nonnegative(),
  avg_fill_price: z.number().positive().optional(),
  commission: z.number().nonnegative().optional(),
  commission_asset: z.string().optional(),
  pnl: z.number().optional(),
  pnl_percent: z.number().optional(),
  error_message: z.string().optional(),
  created_at: z.string().datetime(),
  updated_at: z.string().datetime(),
  filled_at: z.string().datetime().optional(),
});
export type Execution = z.infer<typeof ExecutionSchema>;

// Map from model action to execution side
export function actionToExecutionSide(action: number): ExecutionSide | null {
  switch (action) {
    case Action.BUY:
      return 'BUY';
    case Action.SELL:
      return 'SELL';
    case Action.HOLD:
      return null; // No execution for HOLD
    default:
      return null;
  }
}

// Status colors for UI
export const EXECUTION_STATUS_COLORS: Record<ExecutionStatus, string> = {
  PENDING: 'text-yellow-500',
  SUBMITTED: 'text-blue-500',
  PARTIALLY_FILLED: 'text-cyan-500',
  FILLED: 'text-green-500',
  CANCELLED: 'text-gray-500',
  REJECTED: 'text-red-500',
  EXPIRED: 'text-orange-500',
};
```

---

## 3. Plan de Migración de State Management

### 3.1 Consolidación en Zustand + React Query

**Principio**: Zustand para estado cliente (UI, auth), React Query para estado servidor (API data).

```typescript
// lib/stores/index.ts
export { useAuthStore } from './authStore';
export { useUIStore } from './uiStore';
export { useTradingStore } from './tradingStore';

// lib/stores/authStore.ts
import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import { User, AuthTokens } from '@/lib/contracts/core/auth.contract';

interface AuthState {
  user: User | null;
  tokens: AuthTokens | null;
  isAuthenticated: boolean;
  isLoading: boolean;

  // Actions
  setAuth: (user: User, tokens: AuthTokens) => void;
  logout: () => void;
  updateUser: (user: Partial<User>) => void;
  refreshTokens: (tokens: AuthTokens) => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      tokens: null,
      isAuthenticated: false,
      isLoading: true,

      setAuth: (user, tokens) => set({
        user,
        tokens,
        isAuthenticated: true,
        isLoading: false,
      }),

      logout: () => set({
        user: null,
        tokens: null,
        isAuthenticated: false,
        isLoading: false,
      }),

      updateUser: (updates) => set((state) => ({
        user: state.user ? { ...state.user, ...updates } : null,
      })),

      refreshTokens: (tokens) => set({ tokens }),
    }),
    {
      name: 'usdcop-auth',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        user: state.user,
        tokens: state.tokens,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
);
```

### 3.2 Migración de Context Existente

| Context Actual | Migración | Justificación |
|----------------|-----------|---------------|
| `ModelContext` | Mantener como Context | Usado por muchos componentes, bajo churn |
| `LanguageContext` | Mantener como Context | Simple, sin side effects |
| `NotificationContext` | → `useUIStore` | Consolidar con toasts de SignalBridge |

---

## 4. Integración de Servicios (DRY)

### 4.1 Capa de Servicios Unificada

```
lib/services/
├── api/
│   ├── client.ts              # ✓ Existente - Base API client
│   ├── endpoints.ts           # 🆕 Centralized endpoint definitions
│   └── interceptors.ts        # 🆕 Auth, error handling interceptors
│
├── trading/                   # ✓ Existente
│   ├── backtest.service.ts
│   ├── market-data/
│   └── signals.service.ts
│
├── execution/                 # 🆕 SignalBridge services
│   ├── exchange.service.ts
│   ├── execution.service.ts
│   └── trading-config.service.ts
│
├── auth/                      # 🔄 Unificar
│   ├── auth.service.ts        # Consolidar dashboard + SignalBridge
│   └── token.service.ts       # Token refresh logic
│
└── shared/                    # 🆕 Servicios compartidos
    ├── cache.service.ts       # Caching layer
    ├── notification.service.ts
    └── websocket.service.ts
```

### 4.2 Service Factory Pattern

```typescript
// lib/services/execution/exchange.service.ts
import { apiClient } from '../api/client';
import {
  ConnectedExchange,
  ConnectedExchangeSchema,
  ConnectExchangeRequest,
  ExchangeBalance,
  ExchangeBalanceSchema,
  SupportedExchange,
} from '@/lib/contracts/execution/exchange.contract';
import { z } from 'zod';

class ExchangeService {
  private readonly baseUrl = '/api/v1/exchanges';

  async getConnectedExchanges(): Promise<ConnectedExchange[]> {
    const response = await apiClient.get(`${this.baseUrl}/credentials`);
    return z.array(ConnectedExchangeSchema).parse(response.data);
  }

  async connectExchange(request: ConnectExchangeRequest): Promise<ConnectedExchange> {
    const response = await apiClient.post(`${this.baseUrl}/credentials`, request);
    return ConnectedExchangeSchema.parse(response.data);
  }

  async disconnectExchange(exchangeId: string): Promise<void> {
    await apiClient.delete(`${this.baseUrl}/credentials/${exchangeId}`);
  }

  async validateConnection(exchangeId: string): Promise<{ valid: boolean; error?: string }> {
    const response = await apiClient.post(`${this.baseUrl}/credentials/${exchangeId}/validate`);
    return response.data;
  }

  async getBalances(exchange: SupportedExchange): Promise<ExchangeBalance[]> {
    const response = await apiClient.get(`${this.baseUrl}/credentials/${exchange}/balances`);
    return z.array(ExchangeBalanceSchema).parse(response.data);
  }

  async getAllBalances(): Promise<Record<SupportedExchange, ExchangeBalance[]>> {
    const response = await apiClient.get(`${this.baseUrl}/balances`);
    // Validate each exchange's balances
    const validated: Record<SupportedExchange, ExchangeBalance[]> = {} as any;
    for (const [exchange, balances] of Object.entries(response.data)) {
      validated[exchange as SupportedExchange] = z.array(ExchangeBalanceSchema).parse(balances);
    }
    return validated;
  }
}

// Singleton export
export const exchangeService = new ExchangeService();
```

### 4.3 React Query Hooks

```typescript
// lib/hooks/useExchanges.ts
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { exchangeService } from '@/lib/services/execution/exchange.service';
import { ConnectExchangeRequest, SupportedExchange } from '@/lib/contracts/execution/exchange.contract';
import { toast } from '@/lib/utils/toast';

// Query keys factory
export const exchangeKeys = {
  all: ['exchanges'] as const,
  lists: () => [...exchangeKeys.all, 'list'] as const,
  balances: (exchange?: SupportedExchange) => [...exchangeKeys.all, 'balances', exchange] as const,
  detail: (id: string) => [...exchangeKeys.all, 'detail', id] as const,
};

export function useConnectedExchanges() {
  return useQuery({
    queryKey: exchangeKeys.lists(),
    queryFn: () => exchangeService.getConnectedExchanges(),
    staleTime: 30_000, // 30 seconds
  });
}

export function useExchangeBalances(exchange?: SupportedExchange) {
  return useQuery({
    queryKey: exchangeKeys.balances(exchange),
    queryFn: () => exchange
      ? exchangeService.getBalances(exchange)
      : exchangeService.getAllBalances(),
    staleTime: 10_000, // 10 seconds - balances change frequently
    refetchInterval: 30_000, // Auto-refresh every 30s
  });
}

export function useConnectExchange() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: ConnectExchangeRequest) =>
      exchangeService.connectExchange(request),
    onSuccess: (newExchange) => {
      queryClient.invalidateQueries({ queryKey: exchangeKeys.lists() });
      toast.success(`${newExchange.exchange} connected successfully`);
    },
    onError: (error: Error) => {
      toast.error(`Failed to connect exchange: ${error.message}`);
    },
  });
}

export function useDisconnectExchange() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (exchangeId: string) =>
      exchangeService.disconnectExchange(exchangeId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: exchangeKeys.lists() });
      toast.success('Exchange disconnected');
    },
    onError: (error: Error) => {
      toast.error(`Failed to disconnect: ${error.message}`);
    },
  });
}
```

---

## 5. Plan de Routing (Next.js App Router)

### 5.1 Nueva Estructura de Rutas

```
app/
├── (public)/                      # Rutas públicas (no auth)
│   ├── login/
│   │   └── page.tsx
│   ├── register/
│   │   └── page.tsx
│   └── forgot-password/
│       └── page.tsx
│
├── (authenticated)/               # Rutas protegidas
│   ├── layout.tsx                 # Auth check + sidebar layout
│   │
│   ├── dashboard/                 # ✓ Existente
│   │   └── page.tsx
│   │
│   ├── forecasting/               # ✓ Existente
│   │   └── page.tsx
│   │
│   ├── hub/                       # ✓ Existente
│   │   └── page.tsx
│   │
│   ├── execution/                 # 🆕 SignalBridge module
│   │   ├── page.tsx               # Execution dashboard
│   │   ├── exchanges/
│   │   │   ├── page.tsx           # Connected exchanges list
│   │   │   └── connect/
│   │   │       └── [exchange]/
│   │   │           └── page.tsx   # Connect specific exchange
│   │   ├── signals/
│   │   │   └── page.tsx           # Signal history
│   │   ├── orders/
│   │   │   └── page.tsx           # Order/execution history
│   │   ├── portfolio/
│   │   │   └── page.tsx           # Portfolio overview
│   │   └── settings/
│   │       └── page.tsx           # Trading config
│   │
│   └── settings/                  # 🆕 User settings
│       ├── page.tsx               # Settings overview
│       ├── profile/
│       │   └── page.tsx
│       └── security/
│           └── page.tsx
│
├── api/                           # API routes
│   ├── auth/
│   │   ├── [...nextauth]/
│   │   │   └── route.ts           # NextAuth (existente)
│   │   └── signalbridge/          # 🆕 SignalBridge auth proxy
│   │       ├── login/route.ts
│   │       ├── register/route.ts
│   │       └── refresh/route.ts
│   │
│   └── execution/                 # 🆕 Execution API proxy
│       ├── exchanges/route.ts
│       ├── orders/route.ts
│       └── signals/route.ts
│
└── layout.tsx                     # Root layout
```

### 5.2 Middleware de Autenticación

```typescript
// middleware.ts
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import { getToken } from 'next-auth/jwt';

const PUBLIC_PATHS = [
  '/login',
  '/register',
  '/forgot-password',
  '/api/auth',
];

const EXECUTION_PATHS = [
  '/execution',
  '/api/execution',
];

export async function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // Allow public paths
  if (PUBLIC_PATHS.some(path => pathname.startsWith(path))) {
    return NextResponse.next();
  }

  // Check NextAuth session
  const token = await getToken({ req: request });

  if (!token) {
    const loginUrl = new URL('/login', request.url);
    loginUrl.searchParams.set('callbackUrl', pathname);
    return NextResponse.redirect(loginUrl);
  }

  // For execution paths, also check SignalBridge auth
  if (EXECUTION_PATHS.some(path => pathname.startsWith(path))) {
    const sbToken = request.cookies.get('sb_access_token');
    if (!sbToken) {
      // Redirect to SignalBridge login or show upgrade prompt
      const upgradeUrl = new URL('/execution/setup', request.url);
      return NextResponse.redirect(upgradeUrl);
    }
  }

  return NextResponse.next();
}

export const config = {
  matcher: [
    '/((?!_next/static|_next/image|favicon.ico|public/).*)',
  ],
};
```

---

## 6. Plan de Migración de Componentes

### 6.1 Componentes a Migrar de SignalBridge

| Componente | Ubicación Original | Ubicación Destino | Cambios Necesarios |
|------------|-------------------|-------------------|-------------------|
| `ConnectExchangeForm` | `components/exchanges/` | `components/execution/exchanges/` | Adaptar a Next.js form handling |
| `ExchangeCard` | `components/exchanges/` | `components/execution/exchanges/` | Usar Card de ShadCN existente |
| `BalanceDisplay` | `components/dashboard/` | `components/execution/dashboard/` | Unificar con TradingSummaryCard |
| `ExecutionTable` | `components/executions/` | `components/execution/orders/` | Usar TradesTable como base |
| `TradingStatusCard` | `components/dashboard/` | `components/execution/dashboard/` | Integrar con ModelContext |
| `SignalCard` | `components/signals/` | `components/trading/` | Alinear con TradingSignalSchema |

### 6.2 Patrón de Componente Migrado

```typescript
// components/execution/exchanges/ConnectExchangeForm.tsx
'use client';

import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { useRouter } from 'next/navigation';
import {
  ConnectExchangeRequest,
  ConnectExchangeRequestSchema,
  SupportedExchange,
  EXCHANGE_METADATA,
} from '@/lib/contracts/execution/exchange.contract';
import { useConnectExchange } from '@/lib/hooks/useExchanges';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardHeader, CardTitle, CardContent, CardFooter } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Loader2, AlertCircle, CheckCircle2 } from 'lucide-react';

interface Props {
  exchange: SupportedExchange;
}

export function ConnectExchangeForm({ exchange }: Props) {
  const router = useRouter();
  const connectMutation = useConnectExchange();
  const metadata = EXCHANGE_METADATA[exchange];

  const form = useForm<ConnectExchangeRequest>({
    resolver: zodResolver(ConnectExchangeRequestSchema),
    defaultValues: {
      exchange,
      api_key: '',
      api_secret: '',
      label: '',
      is_testnet: false,
    },
  });

  const onSubmit = async (data: ConnectExchangeRequest) => {
    try {
      await connectMutation.mutateAsync(data);
      router.push('/execution/exchanges');
    } catch (error) {
      // Error handled by mutation
    }
  };

  return (
    <Card className="max-w-lg">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <img src={metadata.logo} alt={metadata.name} className="h-6 w-6" />
          Connect {metadata.name}
        </CardTitle>
      </CardHeader>

      <form onSubmit={form.handleSubmit(onSubmit)}>
        <CardContent className="space-y-4">
          {connectMutation.isError && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                {connectMutation.error?.message || 'Failed to connect'}
              </AlertDescription>
            </Alert>
          )}

          <div className="space-y-2">
            <Label htmlFor="api_key">API Key</Label>
            <Input
              id="api_key"
              type="password"
              {...form.register('api_key')}
              placeholder="Enter your API key"
            />
            {form.formState.errors.api_key && (
              <p className="text-sm text-destructive">
                {form.formState.errors.api_key.message}
              </p>
            )}
          </div>

          <div className="space-y-2">
            <Label htmlFor="api_secret">API Secret</Label>
            <Input
              id="api_secret"
              type="password"
              {...form.register('api_secret')}
              placeholder="Enter your API secret"
            />
            {form.formState.errors.api_secret && (
              <p className="text-sm text-destructive">
                {form.formState.errors.api_secret.message}
              </p>
            )}
          </div>

          <div className="space-y-2">
            <Label htmlFor="label">Label (optional)</Label>
            <Input
              id="label"
              {...form.register('label')}
              placeholder="e.g., Main Trading Account"
            />
          </div>
        </CardContent>

        <CardFooter className="flex justify-between">
          <Button
            type="button"
            variant="outline"
            onClick={() => router.back()}
          >
            Cancel
          </Button>
          <Button
            type="submit"
            disabled={connectMutation.isPending}
          >
            {connectMutation.isPending ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Connecting...
              </>
            ) : (
              <>
                <CheckCircle2 className="mr-2 h-4 w-4" />
                Connect Exchange
              </>
            )}
          </Button>
        </CardFooter>
      </form>
    </Card>
  );
}
```

---

## 7. Integración de Temas

### 7.1 Extensión del Tema Existente

```typescript
// tailwind.config.ts
import type { Config } from 'tailwindcss';

const config: Config = {
  // ... existing config
  theme: {
    extend: {
      colors: {
        // Existing dashboard colors...

        // 🆕 Terminal theme from SignalBridge (namespace prefixed)
        terminal: {
          bg: '#030712',
          surface: '#0A0E27',
          elevated: '#0F141B',
          accent: '#06B6D4',
        },

        // 🆕 Market colors (shared)
        market: {
          up: '#00D395',
          down: '#FF3B69',
          neutral: '#8B92A8',
        },

        // 🆕 Execution status colors
        execution: {
          pending: '#EAB308',
          submitted: '#3B82F6',
          filled: '#22C55E',
          cancelled: '#6B7280',
          rejected: '#EF4444',
        },
      },

      // 🆕 Terminal-specific animations
      animation: {
        'pulse-glow': 'pulse-glow 2s ease-in-out infinite',
        'price-flash-up': 'price-flash-up 0.5s ease-out',
        'price-flash-down': 'price-flash-down 0.5s ease-out',
      },

      keyframes: {
        'pulse-glow': {
          '0%, 100%': { opacity: '1', transform: 'scale(1)' },
          '50%': { opacity: '0.8', transform: 'scale(1.02)' },
        },
        'price-flash-up': {
          '0%': { backgroundColor: 'rgb(34 197 94 / 0.3)' },
          '100%': { backgroundColor: 'transparent' },
        },
        'price-flash-down': {
          '0%': { backgroundColor: 'rgb(239 68 68 / 0.3)' },
          '100%': { backgroundColor: 'transparent' },
        },
      },
    },
  },
};

export default config;
```

### 7.2 Theme Provider Extendido

```typescript
// components/providers/ThemeProvider.tsx
'use client';

import { createContext, useContext, useState, useEffect } from 'react';

type Theme = 'light' | 'dark' | 'terminal';

interface ThemeContextType {
  theme: Theme;
  setTheme: (theme: Theme) => void;
  isTerminalMode: boolean;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<Theme>('dark');

  useEffect(() => {
    const stored = localStorage.getItem('usdcop-theme') as Theme | null;
    if (stored) setTheme(stored);
  }, []);

  useEffect(() => {
    document.documentElement.classList.remove('light', 'dark', 'terminal');
    document.documentElement.classList.add(theme);
    localStorage.setItem('usdcop-theme', theme);
  }, [theme]);

  return (
    <ThemeContext.Provider value={{
      theme,
      setTheme,
      isTerminalMode: theme === 'terminal',
    }}>
      {children}
    </ThemeContext.Provider>
  );
}

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) throw new Error('useTheme must be used within ThemeProvider');
  return context;
};
```

---

## 8. Plan de Navegación

### 8.1 Sidebar Extendido

```typescript
// lib/config/navigation.ts
import {
  LayoutDashboard,
  TrendingUp,
  GitCompare,
  Wallet,
  ArrowRightLeft,
  Signal,
  ClipboardList,
  PieChart,
  Settings,
  type LucideIcon,
} from 'lucide-react';

interface NavItem {
  title: string;
  titleEs: string;
  href: string;
  icon: LucideIcon;
  badge?: string;
  children?: NavItem[];
}

interface NavSection {
  title: string;
  titleEs: string;
  items: NavItem[];
}

export const navigation: NavSection[] = [
  {
    title: 'Analytics',
    titleEs: 'Analítica',
    items: [
      {
        title: 'Dashboard',
        titleEs: 'Panel Principal',
        href: '/dashboard',
        icon: LayoutDashboard,
      },
      {
        title: 'Forecasting',
        titleEs: 'Pronósticos',
        href: '/forecasting',
        icon: TrendingUp,
      },
      {
        title: 'Model Hub',
        titleEs: 'Hub de Modelos',
        href: '/hub',
        icon: GitCompare,
      },
    ],
  },
  {
    title: 'Execution',
    titleEs: 'Ejecución',
    items: [
      {
        title: 'Execution Dashboard',
        titleEs: 'Panel de Ejecución',
        href: '/execution',
        icon: Wallet,
        badge: 'NEW',
      },
      {
        title: 'Exchanges',
        titleEs: 'Exchanges',
        href: '/execution/exchanges',
        icon: ArrowRightLeft,
      },
      {
        title: 'Signals',
        titleEs: 'Señales',
        href: '/execution/signals',
        icon: Signal,
      },
      {
        title: 'Orders',
        titleEs: 'Órdenes',
        href: '/execution/orders',
        icon: ClipboardList,
      },
      {
        title: 'Portfolio',
        titleEs: 'Portafolio',
        href: '/execution/portfolio',
        icon: PieChart,
      },
    ],
  },
  {
    title: 'Account',
    titleEs: 'Cuenta',
    items: [
      {
        title: 'Settings',
        titleEs: 'Configuración',
        href: '/settings',
        icon: Settings,
      },
    ],
  },
];
```

---

## 9. Backend Integration

### 9.1 API Route Proxies

```typescript
// app/api/execution/exchanges/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth/options';
import { z } from 'zod';

const SIGNALBRIDGE_API_URL = process.env.SIGNALBRIDGE_API_URL;

export async function GET(request: NextRequest) {
  try {
    const session = await getServerSession(authOptions);
    if (!session?.user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    // Get SignalBridge token from session or cookie
    const sbToken = request.cookies.get('sb_access_token')?.value;
    if (!sbToken) {
      return NextResponse.json({ error: 'SignalBridge auth required' }, { status: 401 });
    }

    const response = await fetch(`${SIGNALBRIDGE_API_URL}/v1/exchanges/credentials`, {
      headers: {
        'Authorization': `Bearer ${sbToken}`,
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`SignalBridge API error: ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);

  } catch (error) {
    console.error('[API] Exchange fetch error:', error);
    return NextResponse.json(
      { error: 'Failed to fetch exchanges' },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const session = await getServerSession(authOptions);
    if (!session?.user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const sbToken = request.cookies.get('sb_access_token')?.value;
    if (!sbToken) {
      return NextResponse.json({ error: 'SignalBridge auth required' }, { status: 401 });
    }

    const body = await request.json();

    const response = await fetch(`${SIGNALBRIDGE_API_URL}/v1/exchanges/credentials`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${sbToken}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const error = await response.json();
      return NextResponse.json(error, { status: response.status });
    }

    const data = await response.json();
    return NextResponse.json(data);

  } catch (error) {
    console.error('[API] Exchange connect error:', error);
    return NextResponse.json(
      { error: 'Failed to connect exchange' },
      { status: 500 }
    );
  }
}
```

---

## 10. Checklist de Implementación

### Fase 1: Fundamentos (Semana 1)

- [ ] **1.1** Crear estructura de contratos unificados
  - [ ] `lib/contracts/core/auth.contract.ts`
  - [ ] `lib/contracts/core/api-response.contract.ts`
  - [ ] `lib/contracts/execution/exchange.contract.ts`
  - [ ] `lib/contracts/execution/execution.contract.ts`
  - [ ] `lib/contracts/execution/trading-config.contract.ts`

- [ ] **1.2** Configurar Zustand stores
  - [ ] `lib/stores/authStore.ts`
  - [ ] `lib/stores/uiStore.ts`
  - [ ] Migrar notificaciones a store

- [ ] **1.3** Extender tema Tailwind
  - [ ] Añadir colores terminal/execution
  - [ ] Añadir animaciones

### Fase 2: Servicios (Semana 2)

- [ ] **2.1** Implementar servicios de ejecución
  - [ ] `lib/services/execution/exchange.service.ts`
  - [ ] `lib/services/execution/execution.service.ts`
  - [ ] `lib/services/execution/trading-config.service.ts`

- [ ] **2.2** Crear React Query hooks
  - [ ] `lib/hooks/useExchanges.ts`
  - [ ] `lib/hooks/useExecutions.ts`
  - [ ] `lib/hooks/useTradingConfig.ts`

- [ ] **2.3** Implementar API route proxies
  - [ ] `/api/execution/exchanges/`
  - [ ] `/api/execution/orders/`
  - [ ] `/api/execution/signals/`

### Fase 3: Componentes (Semana 3)

- [ ] **3.1** Migrar componentes core
  - [ ] `ConnectExchangeForm`
  - [ ] `ExchangeCard`
  - [ ] `BalanceDisplay`
  - [ ] `ExecutionTable`

- [ ] **3.2** Crear páginas de ejecución
  - [ ] `/execution` (dashboard)
  - [ ] `/execution/exchanges`
  - [ ] `/execution/exchanges/connect/[exchange]`
  - [ ] `/execution/orders`
  - [ ] `/execution/signals`
  - [ ] `/execution/portfolio`

- [ ] **3.3** Actualizar navegación
  - [ ] Extender Sidebar
  - [ ] Añadir badges "NEW"

### Fase 4: Integración (Semana 4)

- [ ] **4.1** Conectar con backend SignalBridge
  - [ ] Configurar variables de entorno
  - [ ] Probar conexión exchange
  - [ ] Probar ejecución de órdenes

- [ ] **4.2** Integrar autenticación
  - [ ] Dual auth (NextAuth + SignalBridge JWT)
  - [ ] Token refresh handling
  - [ ] Protected routes

- [ ] **4.3** Testing E2E
  - [ ] Flow de conexión exchange
  - [ ] Flow de visualización de señales
  - [ ] Flow de ejecución de orden

### Fase 5: Polish (Semana 5)

- [ ] **5.1** Optimizaciones
  - [ ] Bundle analysis
  - [ ] Lazy loading de módulo execution
  - [ ] Caching strategies

- [ ] **5.2** Documentación
  - [ ] README actualizado
  - [ ] Storybook para componentes nuevos
  - [ ] API documentation

- [ ] **5.3** Deployment
  - [ ] Environment variables
  - [ ] Docker compose update
  - [ ] CI/CD pipeline

---

## 11. Patrones de Diseño Aplicados

| Patrón | Aplicación | Beneficio |
|--------|------------|-----------|
| **SSOT** | Contratos centralizados en `lib/contracts/` | Una fuente de verdad para tipos |
| **DRY** | Servicios reutilizables, hooks compartidos | Elimina duplicación |
| **Factory** | `createFrontendModelConfig()`, service singletons | Encapsula creación |
| **Adapter** | API route proxies, data transformers | Desacopla backend |
| **Observer** | React Query mutations, websockets | Estado reactivo |
| **Facade** | Servicios que ocultan API complexity | API simple |
| **Strategy** | Theme provider, auth middleware | Comportamiento intercambiable |
| **Composition** | Zod schemas extending each other | Tipos componibles |

---

## 12. Riesgos y Mitigaciones

| Riesgo | Impacto | Mitigación |
|--------|---------|------------|
| Conflicto de auth systems | Alto | Dual auth con fallback a NextAuth |
| Breaking changes en API | Medio | Versionamiento de contratos + Zod validation |
| Performance overhead | Medio | Lazy loading, bundle splitting |
| UX inconsistency | Medio | Design system unificado, Storybook |
| State management complexity | Medio | Clear separation: Zustand (client) vs React Query (server) |

---

## Conclusión

Este plan proporciona una ruta clara para integrar SignalBridge al dashboard existente manteniendo:

1. **Type Safety**: Contratos Zod end-to-end
2. **SSOT**: Una fuente de verdad para configuración
3. **DRY**: Servicios y hooks reutilizables
4. **Escalabilidad**: Arquitectura modular por dominios
5. **Mantenibilidad**: Patrones de diseño probados

La integración se puede realizar de forma incremental, permitiendo releases parciales y rollbacks seguros.

---

*Documento generado: 2026-01-22*
*Versión: 1.0.0*
