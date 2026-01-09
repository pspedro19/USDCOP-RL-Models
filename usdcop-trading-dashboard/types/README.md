# TypeScript Types System

Sistema de tipos centralizado y profesional para el USD/COP Trading Dashboard.

## Estructura

```
types/
├── index.ts       - Exportación central
├── common.ts      - Tipos utilitarios
├── trading.ts     - Trading & OHLCV
├── pipeline.ts    - Pipeline L0-L6
├── api.ts         - API & HTTP
├── websocket.ts   - WebSocket
├── charts.ts      - Charts & Visualizations
└── README.md      - Documentación
```

## Uso Básico

```typescript
import { TradingSignal, PipelineStatus, ApiResponse } from '@/types'
```

Ver ejemplos completos en la documentación del proyecto.
