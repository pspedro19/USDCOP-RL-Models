# Frontend-Backend Integration Guide

## Overview
This guide details how each frontend dashboard component integrates with the backend API endpoints, including data fetching strategies, real-time updates, and error handling.

## Integration Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API    │    │   Data Sources  │
│   Dashboard     │◄──►│   (FastAPI)      │◄──►│   (PostgreSQL,  │
│   (Next.js)     │    │                  │    │    InfluxDB,    │
│                 │    │                  │    │    MinIO, etc.) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │
         │              ┌──────────────────┐
         └──────────────►│   WebSocket      │
                         │   Server         │
                         └──────────────────┘
```

## 1. Executive Overview Integration

### Component: `ExecutiveOverview.tsx`
**Primary Endpoints**:
- `GET /executive/kpis` - Real-time KPI data
- `GET /executive/performance-chart` - Historical performance
- `WS /ws/real-time-feed` - Live updates

**Integration Pattern**:
```typescript
// hooks/useExecutiveData.ts
export const useExecutiveData = () => {
  const [kpis, setKpis] = useState<KPIData | null>(null);
  const [loading, setLoading] = useState(true);

  // Initial data fetch
  useEffect(() => {
    const fetchKPIs = async () => {
      try {
        const response = await fetch('/api/v1/executive/kpis');
        const data = await response.json();
        if (data.success) {
          setKpis(data.data);
        }
      } catch (error) {
        console.error('Failed to fetch KPIs:', error);
      } finally {
        setLoading(false);
      }
    };
    
    fetchKPIs();
  }, []);

  // WebSocket for real-time updates
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/real-time-feed');
    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.topic === 'kpi_update') {
        setKpis(prev => ({ ...prev, ...message.data }));
      }
    };
    return () => ws.close();
  }, []);

  return { kpis, loading };
};
```

**Data Refresh Strategy**:
- Initial load: REST API call
- Real-time updates: WebSocket subscription
- Fallback polling: Every 30 seconds if WebSocket fails
- Cache duration: 5 minutes for historical data

---

## 2. Live Trading Terminal Integration

### Component: `LiveTradingTerminal.tsx`
**Primary Endpoints**:
- `GET /trading/real-time-data` - Current price and indicators
- `GET /trading/historical-data` - OHLCV chart data
- `GET /trading/rl-actions` - RL model decisions
- `POST /trading/manual-override` - Emergency controls

**Integration Pattern**:
```typescript
// hooks/useTradingData.ts
export const useTradingData = () => {
  const [priceData, setPriceData] = useState<PriceData | null>(null);
  const [chartData, setChartData] = useState<OHLCVData[]>([]);
  const [rlActions, setRlActions] = useState<RLAction | null>(null);

  // Real-time price updates
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/real-time-feed');
    ws.onopen = () => {
      ws.send(JSON.stringify({ 
        action: 'subscribe', 
        topics: ['price_feed', 'rl_decisions'] 
      }));
    };
    
    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      switch (message.topic) {
        case 'price_feed':
          setPriceData(message.data);
          break;
        case 'rl_decisions':
          setRlActions(message.data);
          break;
      }
    };
    
    return () => ws.close();
  }, []);

  // Historical chart data
  const fetchChartData = useCallback(async (interval: string, limit: number) => {
    const params = new URLSearchParams({
      symbol: 'USDCOP',
      interval,
      limit: limit.toString()
    });
    
    try {
      const response = await fetch(`/api/v1/trading/historical-data?${params}`);
      const data = await response.json();
      if (data.success) {
        setChartData(data.data);
      }
    } catch (error) {
      console.error('Failed to fetch chart data:', error);
    }
  }, []);

  // Manual override function
  const executeManualOverride = useCallback(async (action: string, reason: string) => {
    try {
      const response = await fetch('/api/v1/trading/manual-override', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${getToken()}`
        },
        body: JSON.stringify({ action, reason, duration_minutes: 60 })
      });
      return response.json();
    } catch (error) {
      console.error('Manual override failed:', error);
      throw error;
    }
  }, []);

  return { 
    priceData, 
    chartData, 
    rlActions, 
    fetchChartData, 
    executeManualOverride 
  };
};
```

**Chart Integration**:
```typescript
// components/TradingChart.tsx
const TradingChart: React.FC = () => {
  const { chartData, priceData } = useTradingData();
  
  const chartOptions = useMemo(() => ({
    chart: { type: 'candlestick' },
    series: [{
      name: 'USDCOP',
      data: chartData.map(d => [
        new Date(d.timestamp).getTime(),
        d.open, d.high, d.low, d.close
      ])
    }],
    plotOptions: {
      candlestick: {
        color: '#DC2626', // market-down
        upColor: '#0D9E75' // market-up
      }
    }
  }), [chartData]);

  return <HighchartsReact highcharts={Highcharts} options={chartOptions} />;
};
```

---

## 3. RL Model Health Integration

### Component: `RLModelHealth.tsx`
**Primary Endpoints**:
- `GET /models/health-summary` - Model status overview
- `GET /models/training-metrics` - Training convergence data
- `GET /models/action-distribution` - Action heatmap data

**Integration Pattern**:
```typescript
// hooks/useModelHealth.ts
export const useModelHealth = () => {
  const [modelHealth, setModelHealth] = useState<ModelHealthData | null>(null);
  const [trainingMetrics, setTrainingMetrics] = useState<TrainingMetrics | null>(null);
  const [actionDistribution, setActionDistribution] = useState<ActionDistribution | null>(null);

  // Fetch model health data
  const fetchModelHealth = useCallback(async () => {
    try {
      const [healthRes, metricsRes, actionRes] = await Promise.all([
        fetch('/api/v1/models/health-summary'),
        fetch('/api/v1/models/training-metrics?model=ensemble&period=7d'),
        fetch('/api/v1/models/action-distribution')
      ]);

      const [healthData, metricsData, actionData] = await Promise.all([
        healthRes.json(),
        metricsRes.json(),
        actionRes.json()
      ]);

      if (healthData.success) setModelHealth(healthData.data);
      if (metricsData.success) setTrainingMetrics(metricsData.data);
      if (actionData.success) setActionDistribution(actionData.data);
    } catch (error) {
      console.error('Failed to fetch model data:', error);
    }
  }, []);

  useEffect(() => {
    fetchModelHealth();
    
    // Refresh every 2 minutes
    const interval = setInterval(fetchModelHealth, 120000);
    return () => clearInterval(interval);
  }, [fetchModelHealth]);

  return { modelHealth, trainingMetrics, actionDistribution };
};
```

**Heatmap Visualization**:
```typescript
// components/ActionHeatmap.tsx
const ActionHeatmap: React.FC<{ data: ActionDistribution }> = ({ data }) => {
  const heatmapData = useMemo(() => {
    return data.hourly_distribution.map((hour, hourIndex) => 
      Object.entries(hour).filter(([key]) => key !== 'hour').map(([action, value], actionIndex) => [
        hourIndex,
        actionIndex,
        Math.round(value * 100)
      ])
    ).flat();
  }, [data]);

  const options = {
    chart: { type: 'heatmap' },
    series: [{
      name: 'Action Probability',
      borderWidth: 1,
      data: heatmapData,
      dataLabels: { enabled: true, color: '#FFFFFF' }
    }],
    colorAxis: {
      min: 0,
      max: 100,
      stops: [
        [0, '#1E293B'],
        [0.5, '#0891B2'],
        [1, '#0D9E75']
      ]
    }
  };

  return <HighchartsReact highcharts={Highcharts} options={options} />;
};
```

---

## 4. Risk Management Integration

### Component: `RiskManagement.tsx`
**Primary Endpoints**:
- `GET /risk/var-analysis` - VaR/CVaR calculations
- `GET /risk/stress-tests` - Stress testing results
- `GET /risk/exposure-analysis` - Portfolio exposure
- `WS /ws/real-time-feed` - Risk alerts

**Integration Pattern**:
```typescript
// hooks/useRiskData.ts
export const useRiskData = () => {
  const [varData, setVarData] = useState<VaRData | null>(null);
  const [stressTests, setStressTests] = useState<StressTestData | null>(null);
  const [exposure, setExposure] = useState<ExposureData | null>(null);
  const [alerts, setAlerts] = useState<RiskAlert[]>([]);

  // Fetch risk data with different confidence levels
  const fetchRiskData = useCallback(async (confidenceLevel = 95) => {
    try {
      const [varRes, stressRes, exposureRes] = await Promise.all([
        fetch(`/api/v1/risk/var-analysis?confidence_level=${confidenceLevel}`),
        fetch('/api/v1/risk/stress-tests'),
        fetch('/api/v1/risk/exposure-analysis')
      ]);

      const results = await Promise.all([
        varRes.json(),
        stressRes.json(),
        exposureRes.json()
      ]);

      results.forEach((result, index) => {
        if (result.success) {
          switch (index) {
            case 0: setVarData(result.data); break;
            case 1: setStressTests(result.data); break;
            case 2: setExposure(result.data); break;
          }
        }
      });
    } catch (error) {
      console.error('Failed to fetch risk data:', error);
    }
  }, []);

  // Risk alerts via WebSocket
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/real-time-feed');
    ws.onopen = () => {
      ws.send(JSON.stringify({ 
        action: 'subscribe', 
        topics: ['risk_alerts'] 
      }));
    };
    
    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.topic === 'risk_alerts') {
        setAlerts(prev => [message.data, ...prev.slice(0, 9)]);
      }
    };
    
    return () => ws.close();
  }, []);

  return { 
    varData, 
    stressTests, 
    exposure, 
    alerts, 
    fetchRiskData 
  };
};
```

---

## 5. Data Pipeline Quality Integration

### Component: `DataPipelineQuality.tsx`
**Primary Endpoints**:
- `GET /pipeline/quality-gates` - L0-L4 quality status
- `GET /pipeline/anti-leakage-checks` - Leakage prevention
- `GET /pipeline/system-resources` - Resource monitoring

**Integration Pattern**:
```typescript
// hooks/usePipelineData.ts
export const usePipelineData = () => {
  const [qualityGates, setQualityGates] = useState<QualityGateData | null>(null);
  const [antiLeakage, setAntiLeakage] = useState<AntiLeakageData | null>(null);
  const [systemResources, setSystemResources] = useState<SystemResourceData | null>(null);

  const fetchPipelineData = useCallback(async () => {
    try {
      const [qualityRes, leakageRes, resourcesRes] = await Promise.all([
        fetch('/api/v1/pipeline/quality-gates'),
        fetch('/api/v1/pipeline/anti-leakage-checks'),
        fetch('/api/v1/pipeline/system-resources')
      ]);

      const results = await Promise.all([
        qualityRes.json(),
        leakageRes.json(),
        resourcesRes.json()
      ]);

      results.forEach((result, index) => {
        if (result.success) {
          switch (index) {
            case 0: setQualityGates(result.data); break;
            case 1: setAntiLeakage(result.data); break;
            case 2: setSystemResources(result.data); break;
          }
        }
      });
    } catch (error) {
      console.error('Failed to fetch pipeline data:', error);
    }
  }, []);

  useEffect(() => {
    fetchPipelineData();
    
    // Refresh every minute for pipeline monitoring
    const interval = setInterval(fetchPipelineData, 60000);
    return () => clearInterval(interval);
  }, [fetchPipelineData]);

  return { qualityGates, antiLeakage, systemResources, refetch: fetchPipelineData };
};
```

**Pipeline Visualization**:
```typescript
// components/PipelineFlowChart.tsx
const PipelineFlowChart: React.FC<{ data: QualityGateData }> = ({ data }) => {
  const pipelineStages = ['l0_raw_data', 'l1_cleaned', 'l2_features', 'l3_model_ready', 'l4_production'];
  
  return (
    <div className="flex items-center justify-between">
      {pipelineStages.map((stage, index) => {
        const stageData = data[stage];
        const statusColor = stageData.status === 'PASS' ? 'bg-market-up' :
                          stageData.status === 'WARNING' ? 'bg-yellow-500' : 'bg-market-down';
        
        return (
          <React.Fragment key={stage}>
            <div className={`w-16 h-16 rounded-full ${statusColor} flex items-center justify-center text-white font-bold`}>
              L{index}
            </div>
            {index < pipelineStages.length - 1 && (
              <div className="flex-1 h-0.5 bg-fintech-dark-700 mx-2">
                <motion.div
                  className="h-full bg-fintech-cyan-500"
                  initial={{ width: 0 }}
                  animate={{ width: stageData.status === 'PASS' ? '100%' : '0%' }}
                  transition={{ duration: 1 }}
                />
              </div>
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
};
```

---

## 6. Audit & Compliance Integration

### Component: `AuditCompliance.tsx`
**Primary Endpoints**:
- `GET /audit/traceability` - SHA256 hash chains
- `GET /audit/regulatory-compliance` - SFC/Basel III status
- `GET /audit/security-compliance` - Security frameworks
- `GET /audit/audit-history` - Historical audits

**Integration Pattern**:
```typescript
// hooks/useAuditData.ts
export const useAuditData = () => {
  const [traceability, setTraceability] = useState<TraceabilityData | null>(null);
  const [regulatory, setRegulatory] = useState<RegulatoryData | null>(null);
  const [security, setSecurity] = useState<SecurityData | null>(null);
  const [auditHistory, setAuditHistory] = useState<AuditHistoryData[]>([]);

  const fetchAuditData = useCallback(async () => {
    const today = new Date();
    const thirtyDaysAgo = new Date(today.getTime() - 30 * 24 * 60 * 60 * 1000);
    
    try {
      const [traceRes, regRes, secRes, historyRes] = await Promise.all([
        fetch(`/api/v1/audit/traceability?entity_type=trade&start_date=${thirtyDaysAgo.toISOString()}&end_date=${today.toISOString()}`),
        fetch('/api/v1/audit/regulatory-compliance'),
        fetch('/api/v1/audit/security-compliance'),
        fetch(`/api/v1/audit/audit-history?start_date=${thirtyDaysAgo.toISOString()}&limit=50`)
      ]);

      const results = await Promise.all([
        traceRes.json(),
        regRes.json(),
        secRes.json(),
        historyRes.json()
      ]);

      results.forEach((result, index) => {
        if (result.success) {
          switch (index) {
            case 0: setTraceability(result.data); break;
            case 1: setRegulatory(result.data); break;
            case 2: setSecurity(result.data); break;
            case 3: setAuditHistory(result.data); break;
          }
        }
      });
    } catch (error) {
      console.error('Failed to fetch audit data:', error);
    }
  }, []);

  return { traceability, regulatory, security, auditHistory, refetch: fetchAuditData };
};
```

---

## Error Handling Strategy

### Global Error Handler
```typescript
// utils/errorHandler.ts
export class APIError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
    public details?: any
  ) {
    super(message);
    this.name = 'APIError';
  }
}

export const handleAPIError = (response: Response, data: any): never => {
  if (!response.ok) {
    throw new APIError(
      response.status,
      data.error?.code || 'UNKNOWN_ERROR',
      data.error?.message || 'An unknown error occurred',
      data.error?.details
    );
  }
  throw new Error('Unexpected error');
};

// Error boundary for dashboard components
export const DashboardErrorBoundary: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <ErrorBoundary
      FallbackComponent={({ error, resetErrorBoundary }) => (
        <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-4">
          <h3 className="text-red-400 font-medium mb-2">Dashboard Error</h3>
          <p className="text-red-300 text-sm mb-4">{error.message}</p>
          <button
            onClick={resetErrorBoundary}
            className="px-4 py-2 bg-red-500/20 text-red-300 rounded hover:bg-red-500/30"
          >
            Retry
          </button>
        </div>
      )}
      onError={(error) => console.error('Dashboard error:', error)}
    >
      {children}
    </ErrorBoundary>
  );
};
```

## Authentication Integration

### JWT Token Management
```typescript
// utils/auth.ts
export const getToken = (): string | null => {
  return localStorage.getItem('access_token');
};

export const setToken = (token: string): void => {
  localStorage.setItem('access_token', token);
};

export const refreshToken = async (): Promise<string> => {
  const refresh = localStorage.getItem('refresh_token');
  if (!refresh) throw new Error('No refresh token available');
  
  const response = await fetch('/api/v1/auth/refresh', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ refresh_token: refresh })
  });
  
  const data = await response.json();
  if (data.success) {
    setToken(data.data.access_token);
    return data.data.access_token;
  }
  
  throw new Error('Token refresh failed');
};

// Axios interceptor for automatic token refresh
axios.interceptors.response.use(
  (response) => response,
  async (error) => {
    if (error.response?.status === 401 && !error.config._retry) {
      error.config._retry = true;
      try {
        const newToken = await refreshToken();
        error.config.headers.Authorization = `Bearer ${newToken}`;
        return axios.request(error.config);
      } catch (refreshError) {
        window.location.href = '/login';
        return Promise.reject(refreshError);
      }
    }
    return Promise.reject(error);
  }
);
```

## Performance Optimization

### Data Caching Strategy
```typescript
// utils/cache.ts
import { QueryClient, QueryCache } from '@tanstack/react-query';

export const queryClient = new QueryClient({
  queryCache: new QueryCache(),
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      cacheTime: 10 * 60 * 1000, // 10 minutes
      retry: 3,
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000)
    }
  }
});

// Custom hook for cached API calls
export const useApiQuery = <T>(
  key: string[],
  fetcher: () => Promise<T>,
  options?: UseQueryOptions<T>
) => {
  return useQuery({
    queryKey: key,
    queryFn: fetcher,
    ...options
  });
};
```

This integration guide ensures that all frontend components are properly connected to their respective backend endpoints with robust error handling, real-time updates, and performance optimization.