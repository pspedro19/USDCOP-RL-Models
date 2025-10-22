# EJEMPLOS DE CÓDIGO: CÓMO ELIMINAR VALORES HARDCODED

Esta guía muestra ejemplos específicos de cómo reemplazar valores hardcoded con llamadas a APIs reales.

---

## EJEMPLO 1: EnhancedTradingTerminal.tsx

### ❌ ANTES (Hardcoded)

```typescript
// Líneas 14-63
const generateKPIData = async () => {
  let pnlIntraday = 0;
  let pnlPercent = 0;

  try {
    const ANALYTICS_API_URL = process.env.NEXT_PUBLIC_ANALYTICS_API_URL || 'http://localhost:8001';
    const response = await fetch(`${ANALYTICS_API_URL}/session-pnl?symbol=USDCOP`);

    if (response.ok) {
      const data = await response.json();
      pnlIntraday = data.session_pnl || 0;
      pnlPercent = data.session_pnl_percent || 0;
    }
  } catch (error) {
    console.error('Error fetching session P&L:', error);
  }

  return {
    session: {
      timeRange: '08:00–12:55',
      pnlIntraday,
      pnlPercent,
      tradesEpisode: 7,  // ❌ HARDCODED
      targetRange: '2–10',
      avgHolding: 12,  // ❌ HARDCODED
      holdingRange: '5–25 barras',
      actionBalance: { sell: 45, buy: 55 },  // ❌ HARDCODED
      drawdownIntraday: -2.1  // ❌ HARDCODED
    },
    execution: {
      vwapVsFill: 1.2,  // ❌ HARDCODED
      spreadEffective: 4.8,  // ❌ HARDCODED
      slippage: 2.1,  // ❌ HARDCODED
      turnCost: 8.5,  // ❌ HARDCODED
      fillRatio: 94.2  // ❌ HARDCODED
    },
    latency: {
      p50: 45,  // ❌ HARDCODED
      p95: 78,  // ❌ HARDCODED
      p99: 95,  // ❌ HARDCODED
      onnxP99: 12  // ❌ HARDCODED
    },
    // ...
  };
};
```

### ✅ DESPUÉS (Conectado a API)

```typescript
// Nueva interfaz para respuesta completa del API
interface SessionKPIsResponse {
  session: {
    timeRange: string;
    pnlIntraday: number;
    pnlPercent: number;
    tradesEpisode: number;
    targetRange: string;
    avgHolding: number;
    holdingRange: string;
    actionBalance: { sell: number; buy: number };
    drawdownIntraday: number;
  };
  execution: {
    vwapVsFill: number;
    spreadEffective: number;
    slippage: number;
    turnCost: number;
    fillRatio: number;
  };
  latency: {
    p50: number;
    p95: number;
    p99: number;
    onnxP99: number;
  };
  marketState: {
    status: string;
    latency: string;
    clockSkew: string;
    lastUpdate: string;
  };
}

const fetchSessionKPIs = async (): Promise<SessionKPIsResponse | null> => {
  try {
    const ANALYTICS_API_URL = process.env.NEXT_PUBLIC_ANALYTICS_API_URL || 'http://localhost:8001';
    const response = await fetch(`${ANALYTICS_API_URL}/api/analytics/session-kpis?symbol=USDCOP`);

    if (!response.ok) {
      throw new Error(`API responded with status ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching session KPIs:', error);
    return null;
  }
};

// En el componente
const [kpiData, setKpiData] = useState<SessionKPIsResponse | null>(null);
const [loading, setLoading] = useState(true);
const [error, setError] = useState<string | null>(null);

useEffect(() => {
  const loadData = async () => {
    setLoading(true);
    const data = await fetchSessionKPIs();

    if (data) {
      setKpiData(data);
      setError(null);
    } else {
      setError('Failed to load session KPIs');
    }

    setLoading(false);
  };

  loadData();
  const interval = setInterval(loadData, 5000); // Update every 5s

  return () => clearInterval(interval);
}, []);

// Renderizado con estados de loading/error
{loading && <div>Loading KPIs...</div>}
{error && <div className="text-red-400">{error}</div>}
{kpiData && (
  <div className="bg-slate-800/50 rounded-lg p-3">
    <div className="text-sm text-slate-400">Trades/Episodio</div>
    <div className="text-xl font-bold text-white">{kpiData.session.tradesEpisode}</div>
  </div>
)}
```

---

## EJEMPLO 2: DataPipelineQuality.tsx

### ❌ ANTES (Hook con datos mock)

```typescript
const useDataPipelineQuality = () => {
  return {
    l0: {
      coverage: 95.8,  // ❌ HARDCODED
      ohlcInvariants: 0,
      crossSourceDelta: 6.2,
      duplicates: 0,
      gaps: 0,
      staleRate: 1.2,
      acquisitionLatency: 340,
      volumeDataPoints: 84455
    },
    l1: {
      gridPerfection: 100,  // ❌ HARDCODED
      terminalCorrectness: 100,
      hodBaselines: 100,
      // ...
    },
    // ...
  };
};
```

### ✅ DESPUÉS (Conectado a Pipeline API)

```typescript
interface PipelineQualityMetrics {
  l0: {
    coverage: number;
    ohlcInvariants: number;
    crossSourceDelta: number;
    duplicates: number;
    gaps: number;
    staleRate: number;
    acquisitionLatency: number;
    volumeDataPoints: number;
  };
  l1: {
    gridPerfection: number;
    terminalCorrectness: number;
    hodBaselines: number;
  };
  l2: {
    winsorizationRate: number;
    outlierRate: number;
  };
  l3: {
    forwardIC: number;
    featureCount: number;
  };
  l4: {
    observationFeatures: number;
    clipRate: number;
    zeroRateT33: number;
  };
  l5: {
    actionSpaceObs: number;
    rewardConsistency: number;
  };
}

const usePipelineQualityMetrics = () => {
  const [metrics, setMetrics] = useState<PipelineQualityMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const PIPELINE_API_URL = process.env.NEXT_PUBLIC_PIPELINE_API_URL || 'http://localhost:8002';

        // Fetch all layers in parallel
        const [l0, l1, l2, l3, l4, l5] = await Promise.all([
          fetch(`${PIPELINE_API_URL}/api/pipeline/l0/quality-metrics`).then(r => r.json()),
          fetch(`${PIPELINE_API_URL}/api/pipeline/l1/quality-metrics`).then(r => r.json()),
          fetch(`${PIPELINE_API_URL}/api/pipeline/l2/quality-metrics`).then(r => r.json()),
          fetch(`${PIPELINE_API_URL}/api/pipeline/l3/quality-metrics`).then(r => r.json()),
          fetch(`${PIPELINE_API_URL}/api/pipeline/l4/quality-metrics`).then(r => r.json()),
          fetch(`${PIPELINE_API_URL}/api/pipeline/l5/quality-metrics`).then(r => r.json())
        ]);

        setMetrics({ l0, l1, l2, l3, l4, l5 });
        setError(null);
      } catch (err) {
        console.error('Error fetching pipeline metrics:', err);
        setError('Failed to load pipeline quality metrics');
      } finally {
        setLoading(false);
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 10000); // Update every 10s

    return () => clearInterval(interval);
  }, []);

  return { metrics, loading, error };
};

// En el componente
const { metrics, loading, error } = usePipelineQualityMetrics();

if (loading) return <LoadingSpinner />;
if (error) return <ErrorMessage message={error} />;
if (!metrics) return <NoDataMessage />;

return (
  <div>
    <div className="metric-card">
      <span>L0 Coverage</span>
      <span className="font-bold">{metrics.l0.coverage.toFixed(2)}%</span>
    </div>
    {/* ... más métricas ... */}
  </div>
);
```

---

## EJEMPLO 3: ModelPerformance.tsx

### ❌ ANTES (Mock data completo)

```typescript
const [metrics] = useState<ModelMetrics>({
  sharpeRatio: 2.34,  // ❌ HARDCODED
  maxDrawdown: 0.087,  // ❌ HARDCODED
  winRate: 0.643,  // ❌ HARDCODED
  totalReturn: 0.156,
  volatility: 0.089,
  // ... más valores hardcoded
});
```

### ✅ DESPUÉS (Fetch desde API)

```typescript
interface ModelMetrics {
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  totalReturn: number;
  volatility: number;
  sortinoRatio: number;
  calmarRatio: number;
  profitFactor: number;
}

const useModelMetrics = (symbol: string, days: number = 30) => {
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const ANALYTICS_API_URL = process.env.NEXT_PUBLIC_ANALYTICS_API_URL || 'http://localhost:8001';
        const response = await fetch(
          `${ANALYTICS_API_URL}/api/analytics/model-metrics?symbol=${symbol}&days=${days}`
        );

        if (!response.ok) {
          throw new Error('Failed to fetch model metrics');
        }

        const data = await response.json();
        setMetrics(data);
        setError(null);
      } catch (err) {
        console.error('Error fetching model metrics:', err);
        setError('Failed to load model performance metrics');
      } finally {
        setLoading(false);
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 30000); // Update every 30s

    return () => clearInterval(interval);
  }, [symbol, days]);

  return { metrics, loading, error };
};

// En el componente
const { metrics, loading, error } = useModelMetrics('USDCOP', 30);
```

---

## EJEMPLO 4: PortfolioExposureAnalysis.tsx

### ❌ ANTES (Mock exposure data)

```typescript
const generateMockExposureData = (): ExposureBreakdown => {
  return {
    countryExposure: [
      { country: 'Colombia', exposure: 8500000, percentage: 85, risk: 'medium' },  // ❌ HARDCODED
      { country: 'United States', exposure: 1000000, percentage: 10, risk: 'low' },
      // ...
    ],
    currencyExposure: [
      { currency: 'COP', exposure: 8500000, percentage: 85, hedged: false },  // ❌ HARDCODED
      // ...
    ],
    // ...
  };
};
```

### ✅ DESPUÉS (Fetch desde Portfolio API)

```typescript
interface ExposureBreakdown {
  countryExposure: Array<{
    country: string;
    exposure: number;
    percentage: number;
    risk: 'low' | 'medium' | 'high';
  }>;
  currencyExposure: Array<{
    currency: string;
    exposure: number;
    percentage: number;
    hedged: boolean;
  }>;
  sectorExposure: Array<{
    sector: string;
    exposure: number;
    percentage: number;
    beta: number;
  }>;
  // ...
}

const usePortfolioExposure = (symbol: string) => {
  const [exposure, setExposure] = useState<ExposureBreakdown | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchExposure = async () => {
      try {
        const ANALYTICS_API_URL = process.env.NEXT_PUBLIC_ANALYTICS_API_URL || 'http://localhost:8001';

        // Fetch complete exposure breakdown
        const response = await fetch(
          `${ANALYTICS_API_URL}/api/analytics/portfolio-exposure?symbol=${symbol}`
        );

        if (!response.ok) {
          throw new Error('Failed to fetch portfolio exposure');
        }

        const data = await response.json();
        setExposure(data);
        setError(null);
      } catch (err) {
        console.error('Error fetching portfolio exposure:', err);
        setError('Failed to load portfolio exposure data');
      } finally {
        setLoading(false);
      }
    };

    fetchExposure();
    const interval = setInterval(fetchExposure, 60000); // Update every 60s

    return () => clearInterval(interval);
  }, [symbol]);

  return { exposure, loading, error };
};

// En el componente
const { exposure, loading, error } = usePortfolioExposure('USDCOP');

if (loading) return <LoadingState />;
if (error) return <ErrorState message={error} />;
if (!exposure) return <NoDataState />;

return (
  <div className="grid grid-cols-2 gap-4">
    {exposure.countryExposure.map((country) => (
      <ExposureCard
        key={country.country}
        title={country.country}
        exposure={country.exposure}
        percentage={country.percentage}
        risk={country.risk}
      />
    ))}
  </div>
);
```

---

## EJEMPLO 5: RLModelHealth.tsx

### ❌ ANTES (Estado inicial hardcoded)

```typescript
const [modelHealth, setModelHealth] = useState({
  production: {
    model: 'PPO-LSTM',
    version: 'v2.1.5',
    tradesPerEpisode: 6,  // ❌ HARDCODED
    policyEntropy: 0.34,  // ❌ HARDCODED
    klDivergence: 0.019,  // ❌ HARDCODED
    // ...
  },
  ppo: {
    policyLoss: 0.0023,  // ❌ HARDCODED
    valueLoss: 0.045,  // ❌ HARDCODED
    // ...
  },
  // ...
});
```

### ✅ DESPUÉS (No inicializar con valores hardcoded)

```typescript
interface RLModelHealth {
  production: {
    model: string;
    version: string;
    tradesPerEpisode: number;
    policyEntropy: number;
    klDivergence: number;
    actionDistribution: {
      sell: number;
      hold: number;
      buy: number;
    };
  };
  ppo: {
    policyLoss: number;
    valueLoss: number;
    explainedVariance: number;
    clipFraction: number;
  };
  // ...
}

const useRLModelHealth = () => {
  // ✅ Inicializar como null, no con valores hardcoded
  const [modelHealth, setModelHealth] = useState<RLModelHealth | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchRLMetrics = async () => {
      try {
        const ML_API_URL = process.env.NEXT_PUBLIC_ML_ANALYTICS_API_URL || 'http://localhost:8004';

        // Fetch all RL metrics in parallel
        const [production, ppo, lstm, qrdqn, reward, performance] = await Promise.all([
          fetch(`${ML_API_URL}/api/ml/production-metrics`).then(r => r.json()),
          fetch(`${ML_API_URL}/api/ml/ppo-metrics`).then(r => r.json()),
          fetch(`${ML_API_URL}/api/ml/lstm-metrics`).then(r => r.json()),
          fetch(`${ML_API_URL}/api/ml/qrdqn-metrics`).then(r => r.json()),
          fetch(`${ML_API_URL}/api/ml/reward-metrics`).then(r => r.json()),
          fetch(`${ML_API_URL}/api/ml/system-performance`).then(r => r.json())
        ]);

        setModelHealth({
          production,
          ppo,
          lstm,
          qrdqn,
          reward,
          performance
        });
        setError(null);
      } catch (err) {
        console.error('Error fetching RL metrics:', err);
        setError('Failed to load RL model health metrics');
      } finally {
        setLoading(false);
      }
    };

    fetchRLMetrics();
    const interval = setInterval(fetchRLMetrics, 5000); // Update every 5s

    return () => clearInterval(interval);
  }, []);

  return { modelHealth, loading, error };
};

// En el componente
const { modelHealth, loading, error } = useRLModelHealth();

if (loading) return <SkeletonLoader />;
if (error) return <ErrorDisplay error={error} />;
if (!modelHealth) return <NoDataAvailable />;

return (
  <div>
    <MetricCard
      title="Trades/Episode"
      value={modelHealth.production.tradesPerEpisode}
      subtitle="Full & t≤35 episodes"
    />
  </div>
);
```

---

## PATRÓN GENERAL RECOMENDADO

### 1. Crear Custom Hook para cada fuente de datos

```typescript
// hooks/useSessionKPIs.ts
export const useSessionKPIs = (symbol: string) => {
  const [data, setData] = useState<SessionKPIs | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const response = await fetch(`${API_URL}/endpoint?symbol=${symbol}`);

        if (!response.ok) throw new Error('API error');

        const json = await response.json();
        setData(json);
        setError(null);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, REFRESH_INTERVAL);
    return () => clearInterval(interval);
  }, [symbol]);

  return { data, loading, error };
};
```

### 2. Usar el hook en el componente

```typescript
const MyComponent = () => {
  const { data, loading, error } = useSessionKPIs('USDCOP');

  if (loading) return <LoadingState />;
  if (error) return <ErrorState message={error} />;
  if (!data) return <NoDataState />;

  return <DataDisplay data={data} />;
};
```

### 3. NUNCA inicializar estado con valores hardcoded

```typescript
// ❌ INCORRECTO
const [price, setPrice] = useState(4010.91);

// ✅ CORRECTO
const [price, setPrice] = useState<number | null>(null);
```

---

## COMPONENTES DE UI PARA ESTADOS

### LoadingState

```typescript
const LoadingState = () => (
  <div className="flex items-center justify-center p-8">
    <div className="animate-spin rounded-full h-12 w-12 border-2 border-cyan-400 border-t-transparent" />
    <span className="ml-3 text-slate-400">Loading data...</span>
  </div>
);
```

### ErrorState

```typescript
const ErrorState: React.FC<{ message: string }> = ({ message }) => (
  <div className="bg-red-900/20 border border-red-500/30 rounded-lg p-4">
    <div className="flex items-center gap-2 text-red-400">
      <AlertTriangle className="h-5 w-5" />
      <span className="font-semibold">Error loading data</span>
    </div>
    <p className="text-sm text-slate-400 mt-2">{message}</p>
  </div>
);
```

### NoDataState

```typescript
const NoDataState = () => (
  <div className="text-center p-8 text-slate-400">
    <Database className="h-12 w-12 mx-auto mb-3 opacity-50" />
    <p>No data available</p>
    <p className="text-xs mt-1">Check API connection</p>
  </div>
);
```

---

## CHECKLIST DE IMPLEMENTACIÓN

Cuando limpies un componente, asegúrate de:

- [ ] Crear custom hook para fetch de datos
- [ ] Eliminar TODOS los valores hardcoded del estado inicial
- [ ] Usar `null` o `undefined` para estado inicial
- [ ] Implementar estados de loading, error y no-data
- [ ] Agregar auto-refresh con `setInterval`
- [ ] Limpiar interval en cleanup (`return () => clearInterval()`)
- [ ] Tipear correctamente las interfaces de respuesta API
- [ ] Usar variables de entorno para URLs de API
- [ ] Agregar logs de error apropiados
- [ ] Probar con API real antes de commit

---

**Siguiente paso:** Implementar las APIs faltantes en el backend
**Ver:** `HARDCODED_VALUES_ANALYSIS_REPORT.md` para lista completa de APIs necesarias
