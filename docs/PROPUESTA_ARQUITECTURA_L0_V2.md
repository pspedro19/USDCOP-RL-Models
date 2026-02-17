# Propuesta de Reestructuración: Pipelines L0 v2.0

## Principios de Diseño

| Principio | Aplicación |
|-----------|------------|
| **DRY** | Un solo extractor por fuente, reutilizado en todos los contextos |
| **SOLID** | Single Responsibility por clase, Open/Closed para extensiones |
| **KISS** | Flujos simples y predecibles, sin over-engineering |
| **SSOT** | Una sola fuente de verdad para configuración, extractores, y datos |

---

## 1. Arquitectura Propuesta

### 1.1 Diagrama de Alto Nivel

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         L0 DATA INGESTION v2.0                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    EXTRACTORS LAYER (SSOT)                       │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐            │    │
│  │  │  FRED    │ │ SUAMECA  │ │   BCRP   │ │INVESTING │            │    │
│  │  │ Extractor│ │ Extractor│ │ Extractor│ │ Extractor│            │    │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘            │    │
│  │       └────────────┴────────────┴────────────┘                   │    │
│  │                          │                                        │    │
│  │                    ExtractorRegistry                              │    │
│  └──────────────────────────┼───────────────────────────────────────┘    │
│                             │                                            │
│  ┌──────────────────────────▼───────────────────────────────────────┐    │
│  │                    PIPELINE ORCHESTRATOR                          │    │
│  │                                                                   │    │
│  │   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐            │    │
│  │   │   REALTIME  │   │   BACKFILL  │   │    SEED     │            │    │
│  │   │  (5 min)    │   │  (on-demand)│   │  (startup)  │            │    │
│  │   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘            │    │
│  │          │                 │                 │                    │    │
│  │          └─────────────────┴─────────────────┘                    │    │
│  │                          │                                        │    │
│  │                   UpsertService                                   │    │
│  │              (últimos 5 registros)                                │    │
│  └──────────────────────────┼───────────────────────────────────────┘    │
│                             │                                            │
│  ┌──────────────────────────▼───────────────────────────────────────┐    │
│  │                      DATABASE LAYER                               │    │
│  │   ┌─────────────────┐    ┌─────────────────┐                     │    │
│  │   │ usdcop_m5_ohlcv │    │macro_indicators │                     │    │
│  │   │   (TimescaleDB) │    │     _daily      │                     │    │
│  │   └─────────────────┘    └─────────────────┘                     │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Modos de Operación

| Modo | Trigger | Comportamiento |
|------|---------|----------------|
| **REALTIME** | Cada 5 min (horario mercado) | Extrae + UPSERT últimos 5 registros |
| **BACKFILL** | Manual o gap detectado | Extrae rango completo + UPSERT |
| **SEED** | Startup o restore | Lee backup + Extrae delta hasta hoy |

---

## 2. DAGs Propuestos

### 2.1 DAGs a MANTENER (Refactorizar)

| DAG | Cambios |
|-----|---------|
| `l0_ohlcv_realtime` | Usar ExtractorRegistry, UPSERT últimos 5 |
| `l0_ohlcv_backfill` | Usar ExtractorRegistry, integrar con seed |
| `l0_macro_unified` | Usar ExtractorRegistry, UPSERT últimos 5 días |
| `l0_weekly_backup` | Sin cambios (funciona bien) |

### 2.2 DAGs a ELIMINAR

| DAG | Razón |
|-----|-------|
| `l0_seed_backup` | Consolidar con `l0_weekly_backup` |
| `l0_data_initialization` | Fusionar con `l0_seed_restore` nuevo |

### 2.3 DAGs NUEVOS

| DAG | Propósito |
|-----|-----------|
| `l0_seed_restore` | Unifica restore + backfill + alineación |

---

## 3. Capa de Extractores (SSOT)

### 3.1 Estructura de Archivos

```
airflow/dags/
├── extractors/                    # NUEVA carpeta SSOT
│   ├── __init__.py
│   ├── base.py                    # BaseExtractor ABC
│   ├── registry.py                # ExtractorRegistry (singleton)
│   ├── fred_extractor.py          # FRED API
│   ├── suameca_extractor.py       # BanRep REST API (sin Selenium)
│   ├── bcrp_extractor.py          # BCRP Peru
│   ├── investing_extractor.py     # Investing.com API
│   ├── twelvedata_extractor.py    # TwelveData (OHLCV)
│   └── config.yaml                # Configuración por extractor
│
├── services/
│   ├── upsert_service.py          # NUEVO: UPSERT unificado
│   ├── gap_detector.py            # NUEVO: Detección de gaps
│   └── seed_service.py            # NUEVO: Restore desde backup
│
└── dags/
    ├── l0_ohlcv_realtime.py       # Refactorizado
    ├── l0_macro_realtime.py       # NUEVO: Macro cada 5 min
    ├── l0_backfill.py             # Unificado OHLCV + Macro
    ├── l0_seed_restore.py         # NUEVO: Seed + alineación
    └── l0_weekly_backup.py        # Sin cambios
```

### 3.2 BaseExtractor (ABC)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import pandas as pd

@dataclass
class ExtractionResult:
    """Resultado estandarizado de extracción."""
    source: str
    variable: str
    data: pd.DataFrame
    last_date: datetime
    records_count: int
    success: bool
    error: Optional[str] = None

class BaseExtractor(ABC):
    """Interfaz base para todos los extractores (DRY)."""

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Nombre único de la fuente."""
        pass

    @property
    @abstractmethod
    def variables(self) -> List[str]:
        """Lista de variables que extrae."""
        pass

    @abstractmethod
    def extract(
        self,
        variable: str,
        start_date: datetime,
        end_date: datetime,
        last_n: int = 5
    ) -> ExtractionResult:
        """
        Extrae datos de la fuente.

        Args:
            variable: Nombre de la variable a extraer
            start_date: Fecha inicio (para backfill)
            end_date: Fecha fin
            last_n: Últimos N registros a extraer (para realtime)

        Returns:
            ExtractionResult con datos y metadata
        """
        pass

    @abstractmethod
    def get_latest_date(self, variable: str) -> Optional[datetime]:
        """Obtiene la última fecha disponible en la fuente."""
        pass

    def extract_last_n(self, variable: str, n: int = 5) -> ExtractionResult:
        """Extrae últimos N registros (para UPSERT realtime)."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=n * 2)  # Buffer
        result = self.extract(variable, start_date, end_date, last_n=n)
        if result.success and len(result.data) > n:
            result.data = result.data.tail(n)
        return result
```

### 3.3 ExtractorRegistry (Singleton)

```python
from typing import Dict, Type
from functools import lru_cache

class ExtractorRegistry:
    """Registro central de extractores (SSOT)."""

    _instance = None
    _extractors: Dict[str, BaseExtractor] = {}
    _variable_map: Dict[str, str] = {}  # variable -> source

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """Carga configuración desde YAML."""
        config = load_yaml('extractors/config.yaml')
        for source, cfg in config['sources'].items():
            extractor_class = self._get_extractor_class(source)
            self._extractors[source] = extractor_class(cfg)
            for var in cfg['variables']:
                self._variable_map[var] = source

    def get_extractor(self, source: str) -> BaseExtractor:
        """Obtiene extractor por nombre de fuente."""
        return self._extractors.get(source)

    def get_extractor_for_variable(self, variable: str) -> BaseExtractor:
        """Obtiene extractor para una variable específica."""
        source = self._variable_map.get(variable)
        return self._extractors.get(source)

    def extract_variable(self, variable: str, **kwargs) -> ExtractionResult:
        """Extrae una variable usando el extractor correcto."""
        extractor = self.get_extractor_for_variable(variable)
        if extractor is None:
            raise ValueError(f"No extractor for {variable}")
        return extractor.extract(variable, **kwargs)

    @lru_cache(maxsize=1)
    def get_all_variables(self) -> List[str]:
        """Lista todas las variables disponibles."""
        return list(self._variable_map.keys())
```

---

## 4. UpsertService (DRY)

```python
class UpsertService:
    """
    Servicio unificado de UPSERT (DRY).

    Reescribe los últimos N registros para cada variable,
    permitiendo correcciones de datos retroactivos.
    """

    def __init__(self, conn, table: str, date_col: str = 'fecha'):
        self.conn = conn
        self.table = table
        self.date_col = date_col

    def upsert_last_n(
        self,
        df: pd.DataFrame,
        columns: List[str],
        n: int = 5
    ) -> int:
        """
        UPSERT de los últimos N registros.

        Args:
            df: DataFrame con datos
            columns: Columnas a actualizar
            n: Últimos N registros a reescribir

        Returns:
            Número de registros afectados
        """
        # Tomar solo últimos N
        df_upsert = df.tail(n)

        # Construir query dinámico
        cols = [self.date_col] + columns
        placeholders = ', '.join(['%s'] * len(cols))
        update_clause = ', '.join([
            f"{col} = EXCLUDED.{col}"
            for col in columns
        ])

        query = f"""
            INSERT INTO {self.table} ({', '.join(cols)})
            VALUES ({placeholders})
            ON CONFLICT ({self.date_col}) DO UPDATE SET
                {update_clause},
                updated_at = NOW()
        """

        # Ejecutar batch
        cur = self.conn.cursor()
        data = [tuple(row) for _, row in df_upsert[cols].iterrows()]
        execute_values(cur, query, data)
        self.conn.commit()

        return len(data)

    def upsert_range(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> int:
        """UPSERT de rango completo (para backfill)."""
        return self.upsert_last_n(df, columns, n=len(df))
```

---

## 5. SeedService (Restore + Alineación)

```python
class SeedService:
    """
    Servicio para restaurar desde backup y alinear datos.

    Flujo:
    1. Detecta si existe backup
    2. Restaura backup con UPSERT
    3. Detecta gap entre backup y hoy
    4. Extrae delta desde última fecha hasta hoy
    5. UPSERT delta para alinear
    """

    def __init__(self, registry: ExtractorRegistry):
        self.registry = registry
        self.upsert = UpsertService(get_db_connection(), 'macro_indicators_daily')

    def restore_and_align(self, backup_path: Path) -> Dict:
        """
        Restaura desde backup y alinea hasta hoy.

        Returns:
            Dict con estadísticas de restore y alineación
        """
        stats = {'restored': 0, 'aligned': 0, 'variables': []}

        # 1. Restaurar backup
        if backup_path.exists():
            df_backup = self._load_backup(backup_path)
            stats['restored'] = self.upsert.upsert_range(
                df_backup,
                df_backup.columns.tolist()
            )
            last_backup_date = df_backup['fecha'].max()
        else:
            last_backup_date = datetime(2020, 1, 1)

        # 2. Detectar gap
        today = datetime.now().date()
        if last_backup_date.date() < today:
            gap_start = last_backup_date + timedelta(days=1)

            # 3. Extraer delta por cada variable
            for variable in self.registry.get_all_variables():
                try:
                    result = self.registry.extract_variable(
                        variable,
                        start_date=gap_start,
                        end_date=today
                    )
                    if result.success:
                        self.upsert.upsert_range(result.data, [variable])
                        stats['aligned'] += len(result.data)
                        stats['variables'].append(variable)
                except Exception as e:
                    logging.warning(f"Failed to align {variable}: {e}")

        return stats

    def _load_backup(self, path: Path) -> pd.DataFrame:
        """Carga backup (CSV, CSV.GZ, Parquet)."""
        if path.suffix == '.parquet':
            return pd.read_parquet(path)
        elif path.suffix == '.gz':
            return pd.read_csv(path, compression='gzip')
        else:
            return pd.read_csv(path)
```

---

## 6. DAGs Refactorizados

### 6.1 l0_macro_realtime (NUEVO)

```python
"""
DAG: l0_macro_realtime
======================
Ingesta de indicadores macro cada 5 minutos durante horario de mercado.

Comportamiento:
- Ejecuta cada 5 min (*/5 13-22 * * 1-5 UTC = 8:00-17:00 COT)
- Extrae ÚLTIMOS 5 registros de cada fuente
- UPSERT para reescribir/corregir datos recientes
- Solo variables con actualización intradía (IBR, TPM, yields, FX)

Variables intradía:
- IBR, TPM (BanRep SUAMECA)
- DXY, VIX, yields (Investing.com)
- USDMXN, USDCLP (TwelveData)
"""

from airflow import DAG
from extractors.registry import ExtractorRegistry
from services.upsert_service import UpsertService

INTRADAY_VARIABLES = [
    'FINC_RATE_IBR_OVERNIGHT_COL_D_IBR',
    'POLR_POLICY_RATE_COL_M_TPM',
    'FXRT_INDEX_DXY_USA_D_DXY',
    'VOLT_VIX_USA_D_VIX',
    'FINC_BOND_YIELD10Y_COL_D_COL10Y',
    'FINC_BOND_YIELD5Y_COL_D_COL5Y',
    'EQTY_INDEX_COLCAP_COL_D_COLCAP',
]

def extract_and_upsert_macro(**context):
    """Extrae y UPSERT últimos 5 registros de variables intradía."""
    registry = ExtractorRegistry()
    upsert = UpsertService(get_db_connection(), 'macro_indicators_daily')

    results = {}
    for variable in INTRADAY_VARIABLES:
        try:
            # Extraer últimos 5 registros
            result = registry.extract_variable(
                variable,
                last_n=5
            )

            if result.success:
                # UPSERT últimos 5
                count = upsert.upsert_last_n(result.data, [variable], n=5)
                results[variable] = {'status': 'ok', 'records': count}
            else:
                results[variable] = {'status': 'error', 'error': result.error}

        except Exception as e:
            results[variable] = {'status': 'error', 'error': str(e)}

    return results

with DAG(
    'l0_macro_realtime',
    schedule_interval='*/5 13-22 * * 1-5',  # 8:00-17:00 COT
    catchup=False,
    max_active_runs=1,
) as dag:

    extract_task = PythonOperator(
        task_id='extract_and_upsert',
        python_callable=extract_and_upsert_macro,
    )
```

### 6.2 l0_seed_restore (NUEVO)

```python
"""
DAG: l0_seed_restore
====================
Restaura desde backup y alinea datos hasta hoy.

Flujo:
1. Busca último backup disponible
2. Restaura con UPSERT (no destruye datos más recientes)
3. Detecta gap entre backup y hoy
4. Extrae delta desde todas las fuentes
5. UPSERT delta para alinear
6. Genera reporte de integridad

Trigger:
- Manual (@once)
- O automático al detectar BD vacía
"""

def restore_and_align(**context):
    """Restaura desde backup y alinea hasta hoy."""
    seed_service = SeedService(ExtractorRegistry())

    # Buscar último backup
    backup_dir = Path('/data/backups')
    latest_backup = max(
        backup_dir.glob('macro_*.csv.gz'),
        key=lambda p: p.stat().st_mtime,
        default=None
    )

    if latest_backup:
        stats = seed_service.restore_and_align(latest_backup)
        logging.info(f"Restored {stats['restored']} rows")
        logging.info(f"Aligned {stats['aligned']} rows")
        logging.info(f"Variables: {stats['variables']}")
        return stats
    else:
        # Sin backup, hacer full extraction desde 2020
        return seed_service.full_extraction(start_date='2020-01-01')

with DAG(
    'l0_seed_restore',
    schedule_interval=None,  # Manual
    catchup=False,
) as dag:

    restore_task = PythonOperator(
        task_id='restore_and_align',
        python_callable=restore_and_align,
    )
```

---

## 7. Configuración YAML (SSOT)

```yaml
# extractors/config.yaml

version: "2.0"

# Configuración global
global:
  retry_attempts: 3
  retry_delay_seconds: 5
  timeout_seconds: 30
  upsert_last_n: 5  # Últimos N registros a reescribir

# Market hours (para filtrar datos)
market:
  timezone: "America/Bogota"
  open_hour: 8
  close_hour: 13
  trading_days: [0, 1, 2, 3, 4]  # Mon-Fri

# Fuentes de datos
sources:
  fred:
    enabled: true
    api_key_env: "FRED_API_KEY"
    base_url: "https://api.stlouisfed.org/fred/series/observations"
    rate_limit_per_minute: 120
    variables:
      - name: "FINC_BOND_YIELD2Y_USA_D_DGS2"
        series_id: "DGS2"
        frequency: "D"
      - name: "POLR_PRIME_RATE_USA_D_PRIME"
        series_id: "DPRIME"
        frequency: "D"
      # ... más variables

  suameca:
    enabled: true
    method: "rest_api"  # NO selenium
    base_url: "https://suameca.banrep.gov.co/estadisticas-economicas-back/rest/estadisticaEconomicaRestService"
    variables:
      - name: "FINC_RATE_IBR_OVERNIGHT_COL_D_IBR"
        serie_id: 241
        frequency: "D"
      - name: "POLR_POLICY_RATE_COL_M_TPM"
        serie_id: 59
        frequency: "D"

  bcrp:
    enabled: true
    base_url: "https://estadisticas.bcrp.gob.pe"
    variables:
      - name: "CRSK_SPREAD_EMBI_COL_D_EMBI"
        serie_code: "PD04715XD"
        frequency: "D"

  investing:
    enabled: true
    method: "rest_api"
    rate_limit_per_minute: 6
    variables:
      - name: "FINC_BOND_YIELD5Y_COL_D_COL5Y"
        instrument_id: 29240
        frequency: "D"
      - name: "FINC_BOND_YIELD10Y_COL_D_COL10Y"
        instrument_id: 29236
        frequency: "D"
      - name: "EQTY_INDEX_COLCAP_COL_D_COLCAP"
        instrument_id: 49642
        frequency: "D"
      - name: "FXRT_INDEX_DXY_USA_D_DXY"
        instrument_id: 8827
        frequency: "D"
      - name: "VOLT_VIX_USA_D_VIX"
        instrument_id: 8884
        frequency: "D"
      # ... más variables

  twelvedata:
    enabled: true
    api_keys_env:
      - "TWELVEDATA_API_KEY_1"
      - "TWELVEDATA_API_KEY_2"
    rotate_keys: true
    variables:
      - name: "OHLCV_USDCOP"
        symbol: "USD/COP"
        interval: "5min"
```

---

## 8. Resumen de Cambios

### 8.1 Archivos a CREAR

| Archivo | Propósito |
|---------|-----------|
| `extractors/base.py` | BaseExtractor ABC |
| `extractors/registry.py` | ExtractorRegistry singleton |
| `extractors/fred_extractor.py` | FRED API extractor |
| `extractors/suameca_extractor.py` | SUAMECA REST extractor |
| `extractors/bcrp_extractor.py` | BCRP extractor |
| `extractors/investing_extractor.py` | Investing.com extractor |
| `extractors/config.yaml` | Configuración SSOT |
| `services/upsert_service.py` | UPSERT unificado |
| `services/seed_service.py` | Restore + alineación |
| `dags/l0_macro_realtime.py` | Macro cada 5 min |
| `dags/l0_seed_restore.py` | Seed + align |

### 8.2 Archivos a ELIMINAR

| Archivo | Razón |
|---------|-------|
| `scripts/scraper_suameca_full.py` | Selenium obsoleto |
| `scripts/scraper_suameca_with_selenium()` | Selenium obsoleto |
| `dags/l0_seed_backup.py` | Consolidar en weekly_backup |
| `dags/l0_data_initialization.py` | Reemplazado por seed_restore |
| `services/macro_extraction_strategies.py` | Migrar a extractors/ |

### 8.3 Archivos a REFACTORIZAR

| Archivo | Cambios |
|---------|---------|
| `l0_ohlcv_realtime.py` | Usar ExtractorRegistry, UPSERT últimos 5 |
| `l0_ohlcv_backfill.py` | Usar ExtractorRegistry |
| `l0_macro_unified.py` | Usar ExtractorRegistry, renombrar a l0_macro_daily |
| `l0_weekly_backup.py` | Agregar macro backup si no existe |

---

## 9. Beneficios

| Aspecto | Antes | Después |
|---------|-------|---------|
| **Extractores** | Duplicados en múltiples archivos | SSOT en `extractors/` |
| **UPSERT** | Lógica repetida en 3+ DAGs | `UpsertService` único |
| **Selenium** | Requerido para SUAMECA | REST API (sin Selenium) |
| **Configuración** | Dispersa en múltiples YAML | Un solo `config.yaml` |
| **Restore** | Manual con múltiples pasos | `SeedService` automático |
| **Realtime Macro** | Solo diario | Cada 5 min (intradía) |
| **Correcciones** | Manual | UPSERT últimos 5 automático |

---

## 10. Plan de Implementación

### Fase 1: Extractores (2-3 días)
1. Crear `extractors/base.py` con ABC
2. Crear `extractors/registry.py`
3. Migrar FRED a `extractors/fred_extractor.py`
4. Migrar SUAMECA a `extractors/suameca_extractor.py`
5. Migrar BCRP a `extractors/bcrp_extractor.py`
6. Migrar Investing a `extractors/investing_extractor.py`
7. Crear `extractors/config.yaml`

### Fase 2: Servicios (1-2 días)
1. Crear `services/upsert_service.py`
2. Crear `services/seed_service.py`
3. Crear `services/gap_detector.py`

### Fase 3: DAGs (2-3 días)
1. Crear `l0_macro_realtime.py`
2. Crear `l0_seed_restore.py`
3. Refactorizar `l0_ohlcv_realtime.py`
4. Refactorizar `l0_macro_unified.py` → `l0_macro_daily.py`

### Fase 4: Limpieza (1 día)
1. Eliminar archivos obsoletos
2. Actualizar imports
3. Testing end-to-end
4. Documentación

**Total estimado: 6-9 días**
