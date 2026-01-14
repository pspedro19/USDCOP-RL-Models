# Recomendaciones de Limpieza y Refactorización
# USDCOP-RL-Models

**Fecha**: 2026-01-12 (Actualizado)
**Versión**: 2.0
**Basado en**: ARCHITECTURE_CONTRACTS.md v1.3 + IMPLEMENTATION_PLAN.md v3.6
**Objetivo**: Alinear el proyecto con Clean Code, SOLID, Feature Contract Pattern

---

## ACTUALIZACIÓN 2026-01-12 - Análisis Verificado

### Acciones Completadas (Fase 1)
- [x] Eliminados 75+ directorios `tmpclaude-*`
- [x] Eliminado archivo `nul` (artefacto Windows)
- [x] Actualizado `.gitignore` con exclusiones adicionales
- [x] Verificado: 280 tests passing

### Hallazgos Críticos

#### FeatureBuilder Duplicado (ALTO RIESGO)
```
| Import Path                    | Source File                              | Version |
|-------------------------------|------------------------------------------|---------|
| from src import FeatureBuilder | src/core/services/feature_builder.py     | V2.1    |
| from src.features import FB    | src/features/builder.py                  | V20     |
```
**Problema:** Dos implementaciones diferentes exportadas simultáneamente.

#### ConfigLoader - NO Duplicado (Verificado)
Arquitectura correcta: Interface + Implementation + Adapter (patrón SOLID).

---

## Resumen Ejecutivo

| Categoría | Cantidad | Impacto |
|-----------|----------|---------|
| **ELIMINADO** | 75+ temp dirs | Completado |
| **ARCHIVAR** | 20+ archivos V16/V17/V19 | Listo para ejecutar |
| **CONSOLIDAR** | 4 FeatureBuilders | Alto riesgo - requiere aprobación |
| **MANTENER** | Estructura V20 | - |

---

## 1. ARCHIVOS A ELIMINAR (Prioridad Alta)

### 1.1 Archivos Temporales de Claude (168 directorios)

**Ubicación**: Raíz del proyecto y `usdcop-trading-dashboard/`

```bash
# ELIMINAR INMEDIATAMENTE - Sin valor, ocupan espacio
tmpclaude-*-cwd/   # 155+ en raíz
usdcop-trading-dashboard/tmpclaude-*-cwd/  # 9+ en dashboard
nul                # Archivo vacío (artefacto Windows)
```

**Comando de limpieza**:
```bash
# PowerShell
Get-ChildItem -Path . -Directory -Recurse -Filter "tmpclaude-*-cwd" | Remove-Item -Recurse -Force
Remove-Item -Path ".\nul" -Force -ErrorAction SilentlyContinue
```

**Agregar a .gitignore**:
```gitignore
tmpclaude-*-cwd/
nul
```

---

### 1.2 Documentación Redundante/Obsoleta

**Eliminar** (información ya consolidada en ARCHITECTURE_CONTRACTS.md e IMPLEMENTATION_PLAN.md):

```
# Planes de implementación parciales/obsoletos
HYBRID_METADATA_IMPLEMENTATION_PLAN.md    # Cubierto por IMPLEMENTATION_PLAN P0-8
HYBRID_REPLAY_IMPLEMENTATION_PLAN.md      # Cubierto por ARCHITECTURE_CONTRACTS
DEPLOYMENT_AUDIT_GUIDE.md                 # Consolidar en IMPLEMENTATION_PLAN
VERIFICATION_CHECKLIST.md                 # Mover a docs/ o consolidar

# En docs/ - revisar y consolidar
docs/ARCHITECTURE_DIAGRAMS.md             # Fusionar con ARCHITECTURE.md
docs/ARQUITECTURA_INTEGRAL_V3.md          # Obsoleto si hay versión más nueva
docs/PIPELINE_REDUNDANCY_SUMMARY.md       # Consolidar en IMPLEMENTATION_PLAN
docs/MULTISTRATEGY_READINESS_ASSESSMENT.md  # Revisar si aún aplica
docs/PRODUCTION_PLAN_V19.md               # Obsoleto - modelo actual es V20

# En diagnostica/ - archivar después de resolver
diagnostica/01_queries_diagnostico.sql    # Mover a archive/ después de resolver
diagnostica/03_P0_FIXES.sql               # Aplicar y archivar
```

**Acción**: Mover a `archive/docs-deprecated/` en lugar de eliminar permanentemente.

---

### 1.3 Servicios Duplicados/Obsoletos

**En `services/`**:

```python
# ELIMINAR - Múltiples versiones del mismo servicio
services/paper_trading_v1.py          # Mantener solo la versión más reciente
services/paper_trading_v2.py
services/paper_trading_v3.py
services/paper_trading_v3b.py
# MANTENER: services/paper_trading_v4.py (si es la versión actual)

# ELIMINAR - Servicios legacy si no se usan
services/trading_api_multi_model.py   # Duplica multi_model_trading_api.py
services/trading_api_realtime.py      # Revisar si se usa

# REVISAR - Potencialmente redundantes
services/analyze_v20_actions.py       # Mover a scripts/ o notebooks/
```

**Verificar uso antes de eliminar**:
```bash
grep -r "paper_trading_v1" --include="*.py" .
grep -r "trading_api_realtime" --include="*.py" .
```

---

### 1.4 Notebooks/Scripts de Entrenamiento Obsoletos

**Estructura actual fragmentada**:
```
notebooks/
├── Entrneamiento PPOV1/    # Typo: "Entrneamiento" → Obsoleto V1
├── Entrneamiento PPOV2/    # Typo: "Entrneamiento" → Obsoleto V2
├── pipeline entrenamiento/ # Actual pero sin Feature Contract
└── train_v20_production_parity.py  # Único script V20 actual
```

**Acción**:
```bash
# Mover versiones obsoletas a archive/
mv "notebooks/Entrneamiento PPOV1" "archive/notebooks/training_v1_deprecated"
mv "notebooks/Entrneamiento PPOV2" "archive/notebooks/training_v2_deprecated"
```

---

### 1.5 Archivos de Debug/Screenshots de Test

```bash
# Eliminar capturas de test obsoletas
usdcop-trading-dashboard/test-results/replay-api-test-*/  # Resultados de test
usdcop-trading-dashboard/tests/e2e/screenshots/debug-*.png

debug/                    # Revisar y limpiar periódicamente
debug/landing_review/
debug/logs/
debug/screenshots/
```

---

## 2. ARCHIVOS A EDITAR/REFACTORIZAR (Crítico)

### 2.1 Configuración de Inference API (P0-1)

**Archivo**: `services/inference_api/config.py`

```python
# ACTUAL (INCORRECTO):
norm_stats_path: str = "config/v19_norm_stats.json"

# CORREGIR A:
norm_stats_path: str = "config/v20_norm_stats.json"
```

**Impacto**: CRÍTICO - Modelo V20 usando stats de V19 causa divergencia.

---

### 2.2 ADX Hardcoded (P0-3)

**Archivo**: `airflow/dags/l5_multi_model_inference.py:371-373`

```python
# ACTUAL (BROKEN):
return 25.0  # Placeholder

# CORREGIR: Implementar cálculo real de ADX
# Usar lib/features/calculators/adx.py (a crear)
```

---

### 2.3 Observation Builder - Migrar a Feature Contract (P1-13)

**Archivo**: `services/inference_api/core/observation_builder.py` (si existe)
**O**: `src/core/builders/observation_builder.py`

**Estado actual**: Cálculos manuales de RSI, ATR, ADX
**Requerido**: Usar `lib/features/builder.py` (Feature Contract)

```python
# ANTES (3 implementaciones separadas):
# 1. 01_build_5min_datasets.py - ta.rsi(), ta.atr()
# 2. observation_builder.py - cálculos manuales
# 3. feature_calculator_factory.py - factory no usada

# DESPUÉS (una sola fuente):
from lib.features.builder import FeatureBuilder
builder = FeatureBuilder(version="v20")
obs = builder.build_observation(ohlcv, macro, position, timestamp)
```

---

### 2.4 Dataset Builder - Migrar a Feature Contract

**Archivo**: `data/pipeline/06_rl_dataset_builder/01_build_5min_datasets.py`

**Problemas identificados**:
1. Línea ~751: `ffill()` sin límite (P0-10)
2. Líneas 225-231: `merge_asof` con tolerance de 1 día (P0-11)
3. Cálculo de features separado del contrato (P1-13)

**Cambios requeridos**:

```python
# 1. AGREGAR límite a ffill
MAX_FFILL_LIMIT = 144  # 12 horas de barras de 5 min
df[col] = df[col].ffill(limit=MAX_FFILL_LIMIT)

# 2. ELIMINAR tolerance de merge_asof
df = pd.merge_asof(
    df_ohlcv.sort_values('datetime'),
    df_macro.sort_values('datetime'),
    on='datetime',
    direction='backward'
    # SIN tolerance para evitar data leakage
)

# 3. USAR FeatureBuilder
from lib.features.builder import FeatureBuilder
builder = FeatureBuilder(version="v20")
```

---

### 2.5 Frontend - Confidence Hardcodeada (P1-6)

**Archivo**: `usdcop-trading-dashboard/components/charts/TradingChartWithSignals.tsx:168-180`

```typescript
// ACTUAL (INCORRECTO):
confidence: 75,  // Hardcoded!

// CORREGIR A:
confidence: trade.entry_confidence
  ?? trade.confidence
  ?? trade.model_metadata?.confidence
  ?? 75,  // Fallback solo si no hay datos
```

---

### 2.6 Replay API Client - Model ID Hardcodeado (P0-6)

**Archivo**: `usdcop-trading-dashboard/lib/replayApiClient.ts:518`

```typescript
// ACTUAL:
const modelId = 'ppo_v20';  // Hardcoded

// CORREGIR A:
const modelId = options?.modelId || 'ppo_v20';
```

---

### 2.7 Unificar Tipos de Replay

**Archivos a consolidar**:
```
usdcop-trading-dashboard/lib/types/replay.ts     # 26KB
usdcop-trading-dashboard/types/replay.ts         # Duplicado
```

**Acción**: Mantener solo `lib/types/replay.ts` y actualizar imports.

---

### 2.8 Factory de Calculadoras - Conectar o Eliminar

**Archivo**: `src/core/factories/feature_calculator_factory.py`

**Estado**: Existe pero NO se usa (según ARCHITECTURE_CONTRACTS.md)

**Opciones**:
1. **ELIMINAR** - Si se implementa Feature Contract en `lib/features/`
2. **REFACTORIZAR** - Hacer que use el nuevo Feature Contract

**Recomendación**: ELIMINAR y reemplazar con `lib/features/builder.py`

---

### 2.9 Calculadoras Individuales - Migrar a lib/features

**Archivos en `src/core/calculators/`**:
```
adx_calculator.py
atr_calculator.py
base_calculator.py
macro_change_calculator.py
macro_zscore_calculator.py
returns_calculator.py
rsi_calculator.py
```

**Acción**: Migrar lógica a `lib/features/calculators/` según el contrato.

---

## 3. ARCHIVOS A CREAR (Nueva Arquitectura)

### 3.1 Feature Contract (P1-13)

```
lib/
├── __init__.py
├── features/
│   ├── __init__.py
│   ├── contract.py           # FeatureSpec, FEATURE_CONTRACT_V20
│   ├── builder.py            # FeatureBuilder class
│   └── calculators/
│       ├── __init__.py
│       ├── returns.py        # log_ret_5m, log_ret_1h, log_ret_4h
│       ├── rsi.py            # rsi_9 (period=9)
│       ├── atr.py            # atr_pct (period=10)
│       ├── adx.py            # adx_14 (period=14)
│       └── macro.py          # dxy_z, vix_z, embi_z, etc.
├── model_registry.py         # Model → Contract → Hash linking
├── dataset_registry.py       # Dataset checksums (P1-9)
├── config_loader.py          # YAML config loader (P1-10)
├── inference/
│   ├── __init__.py
│   ├── onnx_converter.py     # PPO to ONNX (P1-14)
│   └── onnx_engine.py        # Production inference
├── risk/
│   ├── __init__.py
│   ├── circuit_breakers.py   # Circuit breaker pattern (P1-15)
│   ├── trading_breakers.py   # Pre-configured breakers
│   └── engine.py             # Risk engine (P1-17)
└── observability/
    ├── __init__.py
    ├── drift_detector.py     # Feature drift (P1-16)
    └── alerts.py             # Alert generation
```

---

### 3.2 Configuración V20 Externalizada (P1-10)

**Crear**: `config/v20_config.yaml`

```yaml
model:
  name: ppo_v20
  version: "20"

training:
  learning_rate: 3.0e-4
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  ent_coef: 0.01
  clip_range: 0.2

thresholds:
  long: 0.30
  short: -0.30

features:
  observation_dim: 15
  norm_stats_path: "config/v20_norm_stats.json"

trading:
  initial_capital: 10000
  transaction_cost_bps: 25
  slippage_bps: 5
```

---

### 3.3 Norm Stats V20 (P0-1)

**Crear si no existe**: `config/v20_norm_stats.json`

```json
{
  "log_ret_5m": {"mean": 0.0, "std": 0.001},
  "log_ret_1h": {"mean": 0.0, "std": 0.003},
  "log_ret_4h": {"mean": 0.0, "std": 0.006},
  "rsi_9": {"mean": 50.0, "std": 15.0},
  "atr_pct": {"mean": 0.005, "std": 0.002},
  "adx_14": {"mean": 25.0, "std": 10.0},
  "dxy_z": {"mean": 0.0, "std": 1.0},
  "dxy_change_1d": {"mean": 0.0, "std": 0.005},
  "vix_z": {"mean": 0.0, "std": 1.0},
  "embi_z": {"mean": 0.0, "std": 1.0},
  "brent_change_1d": {"mean": 0.0, "std": 0.02},
  "rate_spread": {"mean": 6.0, "std": 1.5},
  "usdmxn_change_1d": {"mean": 0.0, "std": 0.01}
}
```

---

### 3.4 Migraciones SQL (P0-8)

**Crear**: `database/migrations/002_add_traceability_columns.sql`

```sql
ALTER TABLE trades_history
ADD COLUMN IF NOT EXISTS features_snapshot JSONB;

ALTER TABLE trades_history
ADD COLUMN IF NOT EXISTS model_hash VARCHAR(64);

ALTER TABLE trades_history
ADD COLUMN IF NOT EXISTS model_version VARCHAR(20);

ALTER TABLE trades_history
ADD COLUMN IF NOT EXISTS norm_stats_version VARCHAR(20);

ALTER TABLE trades_history
ADD COLUMN IF NOT EXISTS config_snapshot JSONB;

CREATE INDEX idx_trades_model_hash ON trades_history(model_hash);
CREATE INDEX idx_trades_model_version ON trades_history(model_version);
```

**Crear**: `database/migrations/003_dataset_registry.sql` (P1-9)

**Crear**: `database/migrations/004_model_registry.sql` (P1-11)

---

### 3.5 Tests de Paridad (P1-12)

**Crear**: `tests/integration/test_feature_contract_parity.py`

```python
def test_training_inference_parity():
    """Features MUST be identical between training and inference"""
    builder = FeatureBuilder(version="v20")
    # ... test implementation

def test_contract_v20_is_valid():
    """Contract v20 passes all validations"""
    contract = get_contract("v20")
    assert validate_contract(contract)
```

---

## 4. ARCHIVOS A MANTENER (Sin cambios mayores)

### 4.1 Estructura Base

```
├── .env.example              # Template de configuración
├── .gitignore                # Reglas de git (actualizar)
├── docker-compose.yml        # Orquestación Docker
├── pytest.ini                # Config de pytest
├── README.md                 # Documentación principal
```

### 4.2 Frontend Core

```
usdcop-trading-dashboard/
├── app/                      # App router Next.js
├── components/ui/            # Componentes UI base
├── lib/auth/                 # Sistema de autenticación
├── contexts/                 # React contexts
├── public/                   # Archivos estáticos
├── styles/                   # Estilos CSS
├── package.json
├── tsconfig.json
└── next.config.js
```

### 4.3 Database Schemas

```
database/
├── schemas/                  # Definiciones de esquemas
└── seed/                     # Datos iniciales
```

### 4.4 Modelos Entrenados

```
models/
├── ppo_v20_production/       # Modelo actual de producción
└── onnx/                     # Versiones ONNX
```

---

## 5. PLAN DE EJECUCIÓN

### Fase 0: Limpieza Inmediata (1 día)

| Tarea | Comando/Acción | Impacto |
|-------|----------------|---------|
| Eliminar tmpclaude-* | `rm -rf tmpclaude-*-cwd` | Libera espacio |
| Actualizar .gitignore | Agregar patrones | Previene futuros |
| Archivar docs obsoletos | `mv` a archive/ | Orden |

### Fase 1: Fixes Críticos P0 (2-3 días)

| Item | Archivo | Cambio |
|------|---------|--------|
| P0-1 | services/inference_api/config.py | v19 → v20 |
| P0-3 | airflow/dags/l5_multi_model_inference.py | ADX real |
| P0-6 | lib/replayApiClient.ts | modelId dinámico |
| P0-8 | database/migrations/ | features_snapshot |
| P0-10 | 01_build_5min_datasets.py | ffill limit |
| P0-11 | 01_build_5min_datasets.py | merge_asof sin tolerance |

### Fase 2: Feature Contract (3-5 días)

| Item | Entregable |
|------|------------|
| P1-13 | lib/features/contract.py |
| P1-13 | lib/features/builder.py |
| P1-13 | lib/features/calculators/* |
| P1-11 | lib/model_registry.py |

### Fase 3: Migración de Componentes (3-5 días)

| Componente | De | A |
|------------|-----|-----|
| Training | 01_build_5min_datasets.py | FeatureBuilder |
| Inference | observation_builder.py | FeatureBuilder |
| Airflow | l5_multi_model_inference.py | FeatureBuilder |

### Fase 4: Industry Grade (5-7 días)

| Item | Entregable |
|------|------------|
| P1-14 | lib/inference/onnx_converter.py |
| P1-15 | lib/risk/circuit_breakers.py |
| P1-16 | lib/observability/drift_detector.py |
| P1-17 | lib/risk/engine.py |

---

## 6. SCRIPTS DE LIMPIEZA

### 6.1 Script PowerShell para Windows

```powershell
# cleanup_project.ps1

# 1. Eliminar archivos temporales
Write-Host "Eliminando archivos temporales..."
Get-ChildItem -Path . -Directory -Recurse -Filter "tmpclaude-*-cwd" | Remove-Item -Recurse -Force
Remove-Item -Path ".\nul" -Force -ErrorAction SilentlyContinue

# 2. Mover docs obsoletos a archive
$obsoleteDocs = @(
    "HYBRID_METADATA_IMPLEMENTATION_PLAN.md",
    "HYBRID_REPLAY_IMPLEMENTATION_PLAN.md",
    "DEPLOYMENT_AUDIT_GUIDE.md",
    "VERIFICATION_CHECKLIST.md"
)

New-Item -ItemType Directory -Force -Path "archive\docs-deprecated" | Out-Null
foreach ($doc in $obsoleteDocs) {
    if (Test-Path $doc) {
        Move-Item $doc "archive\docs-deprecated\" -Force
    }
}

# 3. Archivar notebooks obsoletos
New-Item -ItemType Directory -Force -Path "archive\notebooks" | Out-Null
if (Test-Path "notebooks\Entrneamiento PPOV1") {
    Move-Item "notebooks\Entrneamiento PPOV1" "archive\notebooks\training_v1_deprecated" -Force
}
if (Test-Path "notebooks\Entrneamiento PPOV2") {
    Move-Item "notebooks\Entrneamiento PPOV2" "archive\notebooks\training_v2_deprecated" -Force
}

Write-Host "Limpieza completada!"
```

### 6.2 Script Bash para Linux/Mac

```bash
#!/bin/bash
# cleanup_project.sh

# 1. Eliminar archivos temporales
echo "Eliminando archivos temporales..."
find . -type d -name "tmpclaude-*-cwd" -exec rm -rf {} +
rm -f ./nul

# 2. Mover docs obsoletos
mkdir -p archive/docs-deprecated
for doc in HYBRID_METADATA_IMPLEMENTATION_PLAN.md HYBRID_REPLAY_IMPLEMENTATION_PLAN.md \
           DEPLOYMENT_AUDIT_GUIDE.md VERIFICATION_CHECKLIST.md; do
    [ -f "$doc" ] && mv "$doc" archive/docs-deprecated/
done

# 3. Archivar notebooks obsoletos
mkdir -p archive/notebooks
[ -d "notebooks/Entrneamiento PPOV1" ] && mv "notebooks/Entrneamiento PPOV1" archive/notebooks/training_v1_deprecated
[ -d "notebooks/Entrneamiento PPOV2" ] && mv "notebooks/Entrneamiento PPOV2" archive/notebooks/training_v2_deprecated

echo "Limpieza completada!"
```

---

## 7. CHECKLIST DE VERIFICACIÓN

### Pre-Limpieza

- [ ] Backup completo del proyecto
- [ ] Verificar que no hay trabajo sin commitear
- [ ] Revisar archivos a eliminar manualmente

### Post-Limpieza

- [ ] Ejecutar tests: `pytest tests/`
- [ ] Verificar build del frontend: `cd usdcop-trading-dashboard && npm run build`
- [ ] Verificar Docker compose: `docker-compose config`
- [ ] Verificar imports en Python: `python -c "from lib.features.builder import FeatureBuilder"`

### Validación Feature Contract

- [ ] `config/v20_norm_stats.json` existe
- [ ] `lib/features/contract.py` define FEATURE_CONTRACT_V20
- [ ] `lib/features/builder.py` exporta FeatureBuilder
- [ ] Test de paridad pasa: `pytest tests/integration/test_feature_contract_parity.py`

---

## 8. DEPENDENCIAS ENTRE ITEMS

```
P0-1 (norm_stats) ──┬──► P1-13 (Feature Contract) ──► P1-12 (Parity Tests)
P0-3 (ADX)         ─┤
P0-8 (features_snapshot) ─┤
                    │
                    └──► P1-14 (ONNX) ──┬──► P1-15 (Circuit Breakers)
                                        ├──► P1-16 (Drift Detection)
                                        └──► P1-17 (Risk Engine)
```

---

## 9. MÉTRICAS DE ÉXITO

| Métrica | Antes | Después |
|---------|-------|---------|
| Archivos temporales | 168+ | 0 |
| Implementaciones de features | 3 | 1 (FeatureBuilder) |
| Documentos redundantes | 10+ | Consolidados |
| Tests de paridad | 0 | 5+ |
| Cobertura Feature Contract | 0% | 100% |

---

## 10. NOTAS ADICIONALES

### Principios SOLID Aplicados

1. **S (Single Responsibility)**: FeatureBuilder solo construye features
2. **O (Open/Closed)**: Nuevas versiones (v21, v22) se agregan sin modificar código existente
3. **L (Liskov)**: Todos los calculators implementan IFeatureCalculator
4. **I (Interface Segregation)**: Interfaces pequeñas y específicas
5. **D (Dependency Inversion)**: FeatureBuilder depende de abstracciones (FeatureSpec)

### Factory Pattern

- `FeatureCalculatorFactory` reemplazada por `FeatureBuilder` con contratos
- Cada versión de modelo tiene su contrato inmutable

### Manejo de Contratos

- `FEATURE_CONTRACT_V20` es inmutable (frozen dataclass)
- Nuevas versiones crean nuevos contratos sin modificar existentes
- Hash de modelo + contrato + norm_stats garantiza trazabilidad

---

*Documento generado: 2026-01-11*
*Alineado con: ARCHITECTURE_CONTRACTS.md v1.3, IMPLEMENTATION_PLAN.md v3.6*
