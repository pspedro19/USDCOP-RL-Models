# P2 PENDING TASKS - Mejoras de Prioridad Media
# Proyecto: USDCOP-RL-Models

**Version**: 1.0
**Fecha**: 2026-01-11
**Estado**: BACKLOG - Ejecutar después de completar P0 + P1
**Metodología**: Spec-Driven + AI-Augmented TDD

---

## RESUMEN EJECUTIVO

Este documento contiene las tareas P2 (prioridad media) y tareas de cleanup adicionales que fueron diferidas para enfocarse en P0 + P1. Estas tareas deben ejecutarse **después de completar al menos 80% de CLAUDE_TASKS.md y GEMINI_TASKS.md**.

### Estadísticas

| Categoría | Items | Prioridad Sugerida |
|-----------|-------|-------------------|
| P2 Alta Prioridad | 8 | Sprint siguiente |
| P2 Media Prioridad | 10 | Sprint +2 |
| P2 Baja Prioridad | 5 | Cuando haya tiempo |
| Cleanup Adicional | 5 | Paralelo a desarrollo |
| **TOTAL** | **28** | - |

---

## P2 ALTA PRIORIDAD (Ejecutar primero)

### P2-1: Dynamic Slippage Model

**Audit Ref**: TC-01
**Impacto**: Precisión de backtesting
**Asignar a**: GEMINI (inferencia/trading)

**Problema**: Slippage fijo de 5bps no refleja condiciones reales del mercado.

**Solución sugerida**:
```python
# lib/trading/slippage_model.py

class DynamicSlippageModel:
    """
    Modelo de slippage dinámico basado en:
    - Volatilidad del mercado
    - Spread bid-ask actual
    - Tamaño de la orden
    - Hora del día (liquidez)
    """

    def __init__(self, base_bps: float = 5.0):
        self.base_bps = base_bps

    def calculate_slippage(
        self,
        order_size: float,
        bid_ask_spread: float,
        volatility: float,
        hour_utc: int
    ) -> float:
        """
        Calcula slippage dinámico.

        Returns:
            Slippage en bps
        """
        # Base
        slippage = self.base_bps

        # Ajuste por spread
        slippage += bid_ask_spread * 0.5

        # Ajuste por volatilidad (ATR normalizado)
        if volatility > 0.02:  # Alta volatilidad
            slippage *= 1.5

        # Ajuste por hora (menos liquidez fuera de mercado)
        if hour_utc < 13 or hour_utc > 19:
            slippage *= 1.3

        return min(slippage, 50.0)  # Cap en 50bps
```

---

### P2-7: Documentar V19 vs V20

**Audit Ref**: DOC-01
**Impacto**: Mantenibilidad
**Asignar a**: CLAUDE (documentación)

**Entregable**: Crear `docs/MODEL_VERSIONS.md` con:
- Diferencias de features entre V19 y V20
- Razones de los cambios
- Métricas comparativas
- Guía de migración

---

### P2-17: ModelFactory Pattern

**Audit Ref**: ARCH-01
**Impacto**: Mantenibilidad, testabilidad
**Asignar a**: CLAUDE (arquitectura)

**Problema**: Modelos se instancian directamente, dificulta testing y swapping.

**Solución**:
```python
# lib/models/factory.py

class ModelFactory:
    """Factory para crear modelos de forma consistente."""

    _registry: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str, model_class: Type[BaseModel]):
        cls._registry[name] = model_class

    @classmethod
    def create(
        cls,
        model_type: str,
        version: str,
        config: Optional[dict] = None
    ) -> BaseModel:
        if model_type not in cls._registry:
            raise ValueError(f"Modelo {model_type} no registrado")

        model_class = cls._registry[model_type]
        return model_class(version=version, config=config)

# Uso
ModelFactory.register("ppo", PPOModel)
model = ModelFactory.create("ppo", "v20")
```

---

### P2-18: Dependency Injection / ServiceContainer

**Audit Ref**: ARCH-02
**Impacto**: Testabilidad, modularidad
**Asignar a**: GEMINI (infraestructura)

**Problema**: Dependencias hardcodeadas dificultan testing y configuración.

**Solución**:
```python
# lib/core/container.py

class ServiceContainer:
    """Container para inyección de dependencias."""

    _services: Dict[str, Any] = {}
    _factories: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str, service: Any):
        cls._services[name] = service

    @classmethod
    def register_factory(cls, name: str, factory: Callable):
        cls._factories[name] = factory

    @classmethod
    def get(cls, name: str) -> Any:
        if name in cls._services:
            return cls._services[name]
        if name in cls._factories:
            service = cls._factories[name]()
            cls._services[name] = service
            return service
        raise KeyError(f"Servicio {name} no registrado")

# Configuración
container = ServiceContainer()
container.register_factory("feature_builder", lambda: FeatureBuilder("v20"))
container.register_factory("model", lambda: ModelFactory.create("ppo", "v20"))

# Uso
builder = container.get("feature_builder")
```

---

### P2-20: Trade History Sync + Dual Filter Deprecation

**Audit Ref**: DATA-01
**Impacto**: Consistencia de datos
**Asignar a**: GEMINI (data pipeline)

**Problema**:
- Trade history puede desincronizarse entre BD y frontend
- Dual filtering causa inconsistencias

**Solución**:
- Crear endpoint `/api/trades/sync` que valida consistencia
- Deprecar filtros duplicados (ya iniciado en GEMINI-T10)
- Agregar checksums de trades por sesión

---

### P2-21: Prometheus/Grafana Observability

**Audit Ref**: OPS-01
**Impacto**: Operaciones, debugging
**Asignar a**: GEMINI (observabilidad)

**Entregables**:
```yaml
# docker-compose.observability.yml
services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:latest
    volumes:
      - ./config/grafana/dashboards:/var/lib/grafana/dashboards
    ports:
      - "3001:3000"
```

**Dashboards a crear**:
1. Trading Performance (PnL, Sharpe, Drawdown)
2. Model Inference (latencia, throughput, errores)
3. System Health (CPU, memoria, disk)
4. Feature Drift (PSI, KS test results)

---

### P2-23: Walk-Forward Backtesting

**Audit Ref**: ML-03
**Impacto**: Validación de modelo
**Asignar a**: CLAUDE (ML pipeline)

**Problema**: Backtesting actual puede tener look-ahead bias.

**Solución**:
```python
# lib/backtesting/walk_forward.py

class WalkForwardBacktest:
    """
    Walk-forward backtesting con retraining periódico.

    Evita look-ahead bias al:
    - Entrenar solo con datos pasados
    - Validar en ventana siguiente
    - Re-entrenar periódicamente
    """

    def __init__(
        self,
        train_window_days: int = 365,
        test_window_days: int = 30,
        retrain_frequency_days: int = 30
    ):
        self.train_window = train_window_days
        self.test_window = test_window_days
        self.retrain_freq = retrain_frequency_days

    def run(
        self,
        data: pd.DataFrame,
        model_factory: Callable
    ) -> pd.DataFrame:
        """Ejecuta walk-forward backtest."""
        results = []

        for train_end in self._get_retrain_dates(data):
            # Entrenar con datos hasta train_end
            train_data = data[data.index < train_end].tail(
                self.train_window * 288  # 288 barras/día
            )
            model = model_factory()
            model.train(train_data)

            # Testear en ventana siguiente
            test_start = train_end
            test_end = test_start + pd.Timedelta(days=self.test_window)
            test_data = data[(data.index >= test_start) & (data.index < test_end)]

            period_results = model.backtest(test_data)
            results.append(period_results)

        return pd.concat(results)
```

---

### P2-24: Property-Based Testing

**Audit Ref**: TEST-01
**Impacto**: Cobertura de tests
**Asignar a**: CLAUDE (testing)

**Solución**:
```python
# tests/property/test_feature_builder.py
from hypothesis import given, strategies as st
import numpy as np

@given(
    close_prices=st.lists(
        st.floats(min_value=3000, max_value=5000),
        min_size=100,
        max_size=1000
    ),
    position=st.floats(min_value=-1, max_value=1)
)
def test_feature_builder_never_produces_nan(close_prices, position):
    """FeatureBuilder NUNCA debe producir NaN para inputs válidos."""
    builder = FeatureBuilder("v20")
    ohlcv = create_ohlcv_from_close(close_prices)
    macro = create_mock_macro(len(close_prices))

    for bar_idx in range(14, len(close_prices)):
        obs = builder.build_observation(
            ohlcv, macro, position,
            ohlcv.index[bar_idx], bar_idx
        )
        assert not np.isnan(obs).any()
        assert not np.isinf(obs).any()
```

---

## P2 MEDIA PRIORIDAD

| ID | Descripción | Asignar a |
|----|-------------|-----------|
| P2-3 | Consolidate Constants | CLAUDE |
| P2-5 | Optimize Trade Filtering Performance | GEMINI |
| P2-6 | Model Registry Pattern | GEMINI |
| P2-8 | Reconciliación Datos Históricos | GEMINI |
| P2-9 | Pipeline CI/CD | GEMINI |
| P2-10 | Performance Benchmarks | GEMINI |
| P2-11 | Structured Logging | GEMINI |
| P2-12 | Data Quality Flags | CLAUDE |
| P2-15 | Dataset Checksums (extender GEMINI-T8) | GEMINI |
| P2-19 | DAG Versioning | GEMINI |

---

## P2 BAJA PRIORIDAD

| ID | Descripción | Nota |
|----|-------------|------|
| P2-4 | Refactor Legacy Scripts | Solo si se necesitan |
| P2-13 | Documentation Updates | Continuo |
| P2-14 | Drift Detection adicional | Ya cubierto en P1-16 |
| P2-16 | Cold-Start Warmup | Edge case |
| P2-22 | Feature Store PIT | Requiere infraestructura |

---

## CLEANUP ADICIONAL

### CLEAN-1: Archivar Notebooks Obsoletos

**Ubicación**: `notebooks/`
**Acción**: Mover a `archive/notebooks/`

```
Mover:
- Entrneamiento PPOV1/
- Entrneamiento PPO_V2/
- exploratory_*.ipynb (más de 6 meses)
```

---

### CLEAN-2: Eliminar Servicios Duplicados

**Ubicación**: `services/`
**Acción**: Eliminar versiones obsoletas

```
Eliminar (después de verificar no uso):
- paper_trading_v1/
- paper_trading_v2/
- (mantener solo paper_trading_v3/ o renombrar a paper_trading/)
```

---

### CLEAN-3: Consolidar Documentación

**Acción**: Mover docs redundantes a `archive/docs/`

```
Archivar:
- CLEANUP_COMPLETE_SUMMARY.md
- NEXT_STEPS_COMPLETE.md
- REPLAY_SYSTEM_*.md (excepto el actual)
- OLD_*.md
```

---

### CLEAN-4: Actualizar .gitignore Completo

**Agregar patrones faltantes**:
```gitignore
# IDE
.idea/
.vscode/
*.swp

# Build
dist/
build/
*.egg-info/

# Logs
logs/
*.log

# Data (grandes)
*.parquet
*.h5
data/raw/
data/processed/

# Models (grandes)
models/*.zip
models/*.onnx
!models/.gitkeep

# Secrets
.env
.env.*
*.pem
*.key
credentials.json
```

---

### CLEAN-5: Crear Estructura de Directorios Estándar

```
mkdir -p archive/docs
mkdir -p archive/notebooks
mkdir -p archive/scripts
mkdir -p logs
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/test_fixtures
mkdir -p models
touch models/.gitkeep
```

---

## ORDEN DE EJECUCIÓN SUGERIDO

### Sprint N+1 (Después de P0+P1)

1. **P2-17** (ModelFactory) - Habilitador para otros
2. **P2-18** (DI Container) - Habilitador para otros
3. **P2-21** (Observability) - Crítico para producción
4. **CLEAN-1 a CLEAN-5** - Paralelo

### Sprint N+2

1. **P2-1** (Dynamic Slippage) - Mejora accuracy
2. **P2-23** (Walk-Forward) - Validación robusta
3. **P2-20** (Trade Sync) - Consistencia datos
4. **P2-7** (Docs V19/V20) - Mantenibilidad

### Sprint N+3

1. **P2-24** (Property Tests) - Cobertura
2. **P2 Media Prioridad** - Según necesidad

---

## MATRIZ DE DEPENDENCIAS P2

```
P2-17 (Factory) ──┬──→ P2-6 (Model Registry)
                  │
P2-18 (DI) ───────┼──→ P2-17
                  │
P2-21 (Observ) ───┴──→ [independiente]

P2-23 (Walk-Fwd) ──→ P2-1 (Slippage)

P2-7 (Docs) ──→ [independiente]

P2-24 (Property) ──→ CLAUDE-T1 (FeatureBuilder completo)
```

---

## NOTAS IMPORTANTES

1. **No iniciar P2 hasta que P0+P1 estén al 80%+**
2. **CLEAN-* puede ejecutarse en paralelo** sin afectar desarrollo
3. **P2-21 (Observability) es casi P1** - priorizar si hay problemas en producción
4. **P2-17 y P2-18 son habilitadores** - hacerlos primero desbloquea otros P2

---

*Documento generado: 2026-01-11*
*Próxima revisión: Cuando P0+P1 alcancen 80%*
*Metodología: Spec-Driven + AI-Augmented TDD*
