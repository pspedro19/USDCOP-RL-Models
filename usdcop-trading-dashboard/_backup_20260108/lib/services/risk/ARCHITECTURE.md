# Risk Module Architecture

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     RealTimeRiskEngine                          │
│                      (Orchestrator)                             │
│                                                                 │
│  Responsibilities:                                              │
│  - Coordinate components                                        │
│  - Manage subscription lifecycle                                │
│  - Handle real-time updates                                     │
│  - Provide unified API                                          │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Portfolio  │  │     Risk     │  │     Risk     │         │
│  │   Tracker    │  │   Metrics    │  │    Alert     │         │
│  │              │  │  Calculator  │  │   System     │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
         │                   │                   │
         │                   │                   │
         ▼                   ▼                   ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│              │  │              │  │              │
│  Position    │  │   Metrics    │  │    Alert     │
│  Management  │  │ Calculation  │  │  Generation  │
│              │  │              │  │              │
└──────────────┘  └──────────────┘  └──────────────┘
```

## Data Flow

```
1. Position Update Flow:
   ┌────────┐         ┌──────────────────┐
   │ Client │────────>│ RealTimeRiskEng. │
   └────────┘         └──────────────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │ PortfolioTracker │
                     └──────────────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │ RiskMetricsCalc. │
                     └──────────────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │  RiskAlertSys.   │
                     └──────────────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │   Subscribers    │
                     └──────────────────┘

2. API Metrics Refresh Flow:
   ┌──────────────────┐
   │   Interval       │
   │   Timer (30s)    │
   └──────────────────┘
            │
            ▼
   ┌──────────────────┐         ┌──────────────────┐
   │ RealTimeRiskEng. │────────>│ RiskMetricsCalc. │
   └──────────────────┘         └──────────────────┘
            │                            │
            │                            ▼
            │                   ┌──────────────────┐
            │                   │ Analytics API    │
            │                   │ /risk-metrics    │
            │                   └──────────────────┘
            │                            │
            ▼                            │
   ┌──────────────────┐                 │
   │  RiskAlertSys.   │<────────────────┘
   └──────────────────┘
            │
            ▼
   ┌──────────────────┐
   │   Subscribers    │
   └──────────────────┘
```

## Class Relationships

```
ISubscribable<T>
       ▲
       │ implements
       │
RealTimeRiskEngine
       │
       │ has-a (composition)
       │
       ├───> PortfolioTracker
       │
       ├───> RiskMetricsCalculator
       │
       └───> RiskAlertSystem

Types Module (used by all)
       │
       ├───> Position
       ├───> RiskAlert
       ├───> RealTimeRiskMetrics
       ├───> RiskThresholds
       └───> RiskMetricsApiResponse
```

## Responsibility Matrix

| Component               | Positions | Metrics | Alerts | Orchestration | API | Subscriptions |
|------------------------|-----------|---------|--------|---------------|-----|---------------|
| PortfolioTracker       |     ✓     |         |        |               |     |               |
| RiskMetricsCalculator  |           |    ✓    |        |               |  ✓  |               |
| RiskAlertSystem        |           |         |   ✓    |               |     |               |
| RealTimeRiskEngine     |           |         |        |       ✓       |     |       ✓       |

## File Organization

```
lib/services/risk/
│
├── types.ts                     [125 lines]
│   └── All shared type definitions
│
├── PortfolioTracker.ts          [115 lines]
│   └── Position tracking logic
│
├── RiskMetricsCalculator.ts     [197 lines]
│   ├── API integration
│   └── Metric calculations
│
├── RiskAlertSystem.ts           [287 lines]
│   ├── Alert generation
│   ├── Alert management
│   └── Alert statistics
│
├── RealTimeRiskEngine.ts        [299 lines]
│   ├── Component coordination
│   ├── Subscription management
│   └── Update lifecycle
│
├── index.ts                     [70 lines]
│   └── Public API exports
│
└── README.md                    [389 lines]
    └── Documentation
```

## SOLID Principles Applied

### Single Responsibility Principle (SRP)
Each class has exactly ONE reason to change:

- **PortfolioTracker**: Changes only if position tracking requirements change
- **RiskMetricsCalculator**: Changes only if calculation logic changes
- **RiskAlertSystem**: Changes only if alert rules or management changes
- **RealTimeRiskEngine**: Changes only if orchestration logic changes

### Open/Closed Principle (OCP)
Components are open for extension but closed for modification:

- New alert types can be added without modifying existing RiskAlertSystem code
- New metrics can be calculated without changing core calculator logic
- New position types can be tracked without changing tracker implementation

### Liskov Substitution Principle (LSP)
- RealTimeRiskEngine implements ISubscribable<RealTimeRiskMetrics>
- Any component expecting ISubscribable can use RealTimeRiskEngine
- Derived classes could extend base functionality without breaking contracts

### Interface Segregation Principle (ISP)
- Each component exposes only methods relevant to its responsibility
- Clients depend only on the interfaces they actually use
- No fat interfaces forcing clients to depend on unused methods

### Dependency Inversion Principle (DIP)
- High-level module (RealTimeRiskEngine) depends on abstractions (ISubscribable)
- Low-level modules implement specific functionality
- Both depend on shared type definitions (abstractions)

## Testing Strategy

```
Unit Tests:
├── PortfolioTracker.test.ts
│   ├── Position CRUD operations
│   └── Portfolio summary calculations
│
├── RiskMetricsCalculator.test.ts
│   ├── API integration (mocked)
│   ├── Metric calculations
│   └── Response mapping
│
├── RiskAlertSystem.test.ts
│   ├── Alert generation
│   ├── Threshold checking
│   └── Alert management
│
└── RealTimeRiskEngine.test.ts
    ├── Component coordination
    ├── Subscription management
    └── Update lifecycle

Integration Tests:
└── RiskModule.integration.test.ts
    ├── End-to-end flow
    ├── API integration
    └── Component interaction
```

## Performance Characteristics

| Operation                  | Time Complexity | Space Complexity | Notes                    |
|---------------------------|----------------|------------------|--------------------------|
| Update Position           | O(1)           | O(n)             | HashMap lookup/insert    |
| Remove Position           | O(1)           | O(n)             | HashMap deletion         |
| Calculate Metrics         | O(n)           | O(1)             | Linear scan of positions |
| Check Alerts              | O(1)           | O(m)             | Fixed threshold checks   |
| Notify Subscribers        | O(s)           | O(1)             | Iterate subscribers      |
| Get Alerts (filtered)     | O(m)           | O(m)             | Filter alert array       |

Where:
- n = number of positions
- m = number of alerts
- s = number of subscribers

## Memory Management

- Positions: Stored in Map<string, Position> for O(1) access
- Alerts: Array limited to 100 most recent alerts (configurable)
- Subscribers: Array of callback functions (cleaned up on unsubscribe)
- Update interval: Cleared on destroy() to prevent memory leaks

## Error Handling

```
Each component uses scoped logger:

PortfolioTracker        → logger('PortfolioTracker')
RiskMetricsCalculator   → logger('RiskMetricsCalculator')
RiskAlertSystem         → logger('RiskAlertSystem')
RealTimeRiskEngine      → logger('RealTimeRiskEngine')

Error levels:
- DEBUG: Detailed operation info (dev only)
- INFO:  Important state changes
- WARN:  Recoverable issues
- ERROR: Critical failures
```

## Configuration Points

1. **Risk Thresholds** (constructor parameter)
   ```typescript
   const engine = new RealTimeRiskEngine(customThresholds);
   ```

2. **Update Interval** (internal constant)
   ```typescript
   private readonly UPDATE_INTERVAL_MS = 30000;
   ```

3. **Max Alerts** (internal constant)
   ```typescript
   private readonly MAX_ALERTS = 100;
   ```

4. **API Endpoints** (internal constant)
   ```typescript
   private readonly ANALYTICS_API_URL = '/api/analytics';
   ```

## Migration Path

### Phase 1: Backwards Compatibility (Current)
```typescript
// Old file redirects to new structure
// lib/services/real-time-risk-engine.ts
export * from './risk';
```

### Phase 2: Update Imports (Next)
```typescript
// Update all imports across codebase
// FROM: import { ... } from '@/lib/services/real-time-risk-engine'
// TO:   import { ... } from '@/lib/services/risk'
```

### Phase 3: Remove Old File (Future)
```bash
# Once all imports updated
rm lib/services/real-time-risk-engine.ts
```

## Extension Points

Future enhancements can be added without modifying existing code:

1. **New Alert Types**: Extend RiskAlertSystem with new threshold checks
2. **New Metrics**: Add calculation methods to RiskMetricsCalculator
3. **Position Filters**: Add filter methods to PortfolioTracker
4. **Custom Strategies**: Inject custom threshold checking logic
5. **Persistence**: Add storage adapters for metrics/alerts
6. **WebSockets**: Add real-time streaming adapter

## Dependencies

```
External:
└── @/lib/utils/logger
    └── Centralized logging utility

Internal (within risk module):
├── types.ts
│   └── Shared by all components
│
└── Components use each other only via RealTimeRiskEngine
    (no direct cross-dependencies)
```

## Code Metrics

```
Original monolithic file:  416 lines
New modular structure:    1093 lines (code only)
Documentation:             389 lines (README)
Architecture docs:         TBD (this file)

Total improvement:
- Better organization ✓
- Single responsibility ✓
- Easier testing ✓
- Clearer dependencies ✓
- Better maintainability ✓
```

---

**Last Updated:** 2025-12-17
**Version:** 1.0.0
**Status:** Production Ready
