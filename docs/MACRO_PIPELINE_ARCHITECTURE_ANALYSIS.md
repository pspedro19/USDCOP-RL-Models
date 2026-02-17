# Análisis de Arquitectura: Pipeline de Datos Macro

## 1. Estado Actual del Código

### 1.1 `generate_master_consolidated.py` - Cómo Funciona

```
┌─────────────────────────────────────────────────────────────────┐
│                    FLUJO ACTUAL                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  DICCIONARIO_MACROECONOMICOS_FINAL.csv                          │
│         │                                                        │
│         ▼                                                        │
│  load_dictionary() ──────────────────────┐                      │
│         │                                 │                      │
│         ▼                                 ▼                      │
│  MACRO_DAILY_CLEAN.csv ──► build_header_rows() ──► save_csv()   │
│  MACRO_MONTHLY_CLEAN.csv ─►               │      ─► save_excel()│
│  MACRO_QUARTERLY_CLEAN.csv─►              │      ─► save_parquet│
│                                           │                      │
│                                           ▼                      │
│                                  9 ARCHIVOS MASTER               │
└─────────────────────────────────────────────────────────────────┘
```

**Problemas Identificados:**

1. **No es un "regenerador"** - Solo formatea archivos ya limpios
2. **No hace scraping** - Lee de archivos pre-existentes
3. **Acoplamiento alto** - Rutas hardcodeadas
4. **Sin validación** - No verifica freshness de datos
5. **Sin caché** - Re-procesa todo cada vez

### 1.2 Scrapers Actuales - Cómo Funcionan

| Scraper | Método | Problema |
|---------|--------|----------|
| `scraper_suameca_api.py` | REST API directo | ✅ Óptimo - descubierto hoy |
| `scraper_suameca_full.py` | Selenium | ❌ Lento, frágil |
| `full_macro_regeneration.py` | Mixto (FRED + BCRP) | ⚠️ Sin Investing.com |
| `scraper_investing_selenium.py` | Cloudscraper API | ⚠️ Rate limiting agresivo |

### 1.3 DAGs L0 Actuales

| DAG | Propósito | Estado |
|-----|-----------|--------|
| `l0_macro_unified.py` | Extracción diaria 41+ vars | ⚠️ Usa Strategy Pattern pero SUAMECA requiere Selenium |
| `l0_data_initialization.py` | Restore desde backups | ✅ Funcional |
| `l0_ohlcv_realtime.py` | OHLCV en tiempo real | ✅ Independiente |
| `l0_ohlcv_backfill.py` | Backfill OHLCV | ✅ Independiente |

---

## 2. Propuesta de Arquitectura Mejorada

### 2.1 Patrones de Diseño a Aplicar

#### A) **Strategy Pattern** (ya existe parcialmente)
```python
from abc import ABC, abstractmethod

class MacroExtractor(ABC):
    """Interfaz base para todos los extractores."""

    @abstractmethod
    def extract(self, start_date: str, end_date: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_source_name(self) -> str:
        pass

    @abstractmethod
    def get_variables(self) -> List[str]:
        pass

class FREDExtractor(MacroExtractor):
    def extract(self, start_date, end_date):
        # Implementación específica FRED
        pass

class SUAMECAExtractor(MacroExtractor):
    def extract(self, start_date, end_date):
        # USA REST API descubierto (NO Selenium)
        pass

class BCRPExtractor(MacroExtractor):
    def extract(self, start_date, end_date):
        # EMBI desde BCRP Peru
        pass
```

#### B) **Factory Pattern** para crear extractores
```python
class ExtractorFactory:
    _extractors = {
        'fred': FREDExtractor,
        'suameca': SUAMECAExtractor,
        'bcrp': BCRPExtractor,
        'investing': InvestingExtractor,
    }

    @classmethod
    def create(cls, source: str, config: dict) -> MacroExtractor:
        return cls._extractors[source](config)
```

#### C) **Pipeline Pattern** para el flujo
```python
class MacroPipeline:
    """Pipeline unificado con etapas claras."""

    def __init__(self):
        self.stages = [
            ExtractStage(),      # 1. Scraping
            ValidateStage(),     # 2. Validación
            FusionStage(),       # 3. Merge
            CleanStage(),        # 4. Limpieza
            ConsolidateStage(),  # 5. Master files
        ]

    def run(self, config: PipelineConfig) -> PipelineResult:
        context = {}
        for stage in self.stages:
            context = stage.execute(context)
        return PipelineResult(context)
```

#### D) **Repository Pattern** para acceso a datos
```python
class MacroRepository:
    """Abstrae el acceso a datos (CSV, DB, API)."""

    def get_latest_date(self, variable: str) -> date:
        pass

    def save_series(self, variable: str, df: pd.DataFrame):
        pass

    def get_series(self, variable: str, start: date, end: date) -> pd.DataFrame:
        pass
```

### 2.2 Estructura de Archivos Propuesta

```
airflow/
├── dags/
│   ├── l0_macro_unified.py          # DAG principal (refactorizado)
│   └── l0_macro_regeneration.py     # DAG de regeneración completa (NUEVO)
│
├── services/
│   ├── extractors/                   # Strategy Pattern
│   │   ├── __init__.py
│   │   ├── base.py                  # MacroExtractor ABC
│   │   ├── fred_extractor.py
│   │   ├── suameca_extractor.py     # USA REST API
│   │   ├── bcrp_extractor.py
│   │   └── investing_extractor.py
│   │
│   ├── pipeline/                     # Pipeline Pattern
│   │   ├── __init__.py
│   │   ├── stages.py                # Extract, Validate, Fusion, Clean, Consolidate
│   │   └── runner.py                # MacroPipeline
│   │
│   └── repository/                   # Repository Pattern
│       ├── __init__.py
│       └── macro_repository.py
│
└── config/
    └── l0_macro_sources.yaml         # SSOT para todas las fuentes
```

---

## 3. DAGs L0 - Qué Descontinuar

### 3.1 Mantener (Refactorizar)

| DAG | Razón |
|-----|-------|
| `l0_macro_unified.py` | Core del sistema - refactorizar con nuevos extractores |
| `l0_data_initialization.py` | Necesario para cold start |
| `l0_ohlcv_realtime.py` | Independiente de macro |
| `l0_ohlcv_backfill.py` | Independiente de macro |

### 3.2 Descontinuar / Eliminar

| Componente | Razón |
|------------|-------|
| `scraper_suameca_full.py` | ❌ Selenium obsoleto - reemplazado por `scraper_suameca_api.py` |
| `scraper_investing_selenium.py` | ⚠️ Evaluar - si REST API funciona, eliminar Selenium |
| Scripts manuales en `/scripts/` | ⚠️ Migrar lógica a servicios de Airflow |

### 3.3 Nuevo DAG Propuesto: `l0_macro_regeneration.py`

```python
"""
DAG: l0_macro_regeneration
==========================
Regeneración completa de datos macro desde cero.

Propósito:
    - Scraping limpio desde 2020 de TODAS las fuentes
    - Regeneración de los 9 archivos master
    - Validación de integridad

Schedule:
    Manual trigger o @weekly para refresh completo

Reemplaza:
    - scripts/full_macro_regeneration.py
    - scripts/generate_master_consolidated.py
"""

with DAG(
    'l0_macro_regeneration',
    schedule_interval='0 6 * * 0',  # Domingos 6am
    ...
) as dag:

    # 1. Extract from all sources in parallel
    extract_fred = PythonOperator(...)
    extract_suameca = PythonOperator(...)  # USA REST API
    extract_bcrp = PythonOperator(...)
    extract_investing = PythonOperator(...)

    # 2. Fusion
    fusion = PythonOperator(...)

    # 3. Cleaning
    cleaning = PythonOperator(...)

    # 4. Generate 9 master files
    consolidate = PythonOperator(...)

    # 5. Validate
    validate = PythonOperator(...)

    # Dependencies
    [extract_fred, extract_suameca, extract_bcrp, extract_investing] >> fusion
    fusion >> cleaning >> consolidate >> validate
```

---

## 4. Mejoras Específicas por Scraper

### 4.1 SUAMECA (IBR, TPM) - ✅ RESUELTO

**Antes (Selenium):**
```python
# scraper_suameca_full.py - LENTO, FRÁGIL
driver = uc.Chrome(options=options)
driver.get(url)
time.sleep(5)  # Esperar Angular
# ... click Vista tabla, parsear HTML
```

**Después (REST API):**
```python
# scraper_suameca_api.py - RÁPIDO, ESTABLE
url = f"{API_BASE}/consultaInformacionSerie?idSerie={serie_id}"
resp = requests.get(url, headers=HEADERS, timeout=30)
data = resp.json()[0]['data']  # [timestamp_ms, value]
```

**Ventajas:**
- 100x más rápido (segundos vs minutos)
- Sin dependencias de Chrome/Selenium
- Más estable (API vs scraping HTML)
- Datos completos desde 2008

### 4.2 FRED API - ✅ ÓPTIMO

```python
# Ya es óptimo - REST API oficial
url = "https://api.stlouisfed.org/fred/series/observations"
params = {"series_id": series_id, "api_key": API_KEY, ...}
```

**Mejora:** Agregar caché con TTL para evitar llamadas redundantes.

### 4.3 BCRP Peru (EMBI) - ✅ FUNCIONAL

```python
# Scraping HTML pero estable
url = f"https://estadisticas.bcrp.gob.pe/.../resultados/PD04715XD/html/"
# Parsea tabla HTML directamente
```

### 4.4 Investing.com - ⚠️ MEJORAR

**Problema actual:** Rate limiting agresivo (8-15s entre requests)

**Propuesta:**
```python
class InvestingExtractor(MacroExtractor):
    def __init__(self):
        self.session = create_api_session()
        self.rate_limiter = TokenBucket(tokens=10, refill_rate=1/10)

    async def extract_parallel(self, indicators: List[str]):
        """Extrae indicadores en paralelo respetando rate limit."""
        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_one(session, ind) for ind in indicators]
            return await asyncio.gather(*tasks)
```

---

## 5. Configuración YAML Propuesta

```yaml
# config/l0_macro_sources.yaml

sources:
  fred:
    enabled: true
    api_key_env: FRED_API_KEY
    rate_limit: 120  # requests/minute
    series:
      DGS2:
        variable: FINC_BOND_YIELD2Y_USA_D_DGS2
        frequency: D
      DPRIME:
        variable: POLR_PRIME_RATE_USA_D_PRIME
        frequency: D
      # ... más series

  suameca:
    enabled: true
    api_base: https://suameca.banrep.gov.co/estadisticas-economicas-back/rest/estadisticaEconomicaRestService
    method: rest_api  # NO selenium
    series:
      IBR:
        id: 241
        variable: FINC_RATE_IBR_OVERNIGHT_COL_D_IBR
      TPM:
        id: 59
        variable: POLR_POLICY_RATE_COL_M_TPM

  bcrp:
    enabled: true
    series:
      EMBI:
        variable: CRSK_SPREAD_EMBI_COL_D_EMBI
        url: https://estadisticas.bcrp.gob.pe/.../PD04715XD/html/

  investing:
    enabled: true
    rate_limit: 6  # requests/minute (conservative)
    series:
      DXY: {id: 8827, variable: FXRT_INDEX_DXY_USA_D_DXY}
      VIX: {id: 8884, variable: VOLT_VIX_USA_D_VIX}
      # ... más series

pipeline:
  start_date: "2020-01-01"
  output_formats: [csv, xlsx, parquet]
  output_dir: data/pipeline/01_sources/consolidated
```

---

## 6. Resumen de Cambios

### Código a Mantener
- `scraper_suameca_api.py` - ✅ Nuevo, óptimo
- `scraper_embi_bcrp.py` - ✅ Funcional
- FRED functions en `full_macro_regeneration.py` - ✅ Migrar a servicio

### Código a Deprecar
- `scraper_suameca_full.py` - ❌ Selenium obsoleto
- `scraper_suameca_with_selenium()` en `full_macro_regeneration.py` - ❌

### DAGs
- `l0_macro_unified.py` - Refactorizar para usar nuevos extractores
- NUEVO: `l0_macro_regeneration.py` - Regeneración semanal completa

### Scripts
- `generate_master_consolidated.py` - Migrar a stage de pipeline
- `full_macro_regeneration.py` - Migrar a DAG de Airflow

---

## 7. Plan de Implementación

1. **Fase 1:** Crear extractores con Strategy Pattern
2. **Fase 2:** Crear MacroPipeline con stages
3. **Fase 3:** Crear `l0_macro_regeneration.py`
4. **Fase 4:** Refactorizar `l0_macro_unified.py`
5. **Fase 5:** Deprecar scripts manuales
6. **Fase 6:** Testing y validación
