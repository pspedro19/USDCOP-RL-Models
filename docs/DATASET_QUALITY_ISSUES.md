# Dataset Quality Issues - L0 Macro Pipeline

**Fecha de Análisis:** 2026-01-20
**Tabla:** `macro_indicators_daily`
**Período Evaluado:** 2020-01-01 a 2026-01-20

---

## Resumen Ejecutivo

| Métrica | Valor |
|---------|-------|
| Total Variables | 38 |
| Variables Configuradas | 34 |
| Variables Legacy (sin extracción) | 4 |
| Calidad General | **7/10** |
| Estado | ⚠️ USABLE CON LIMITACIONES |

---

## Arquitectura Actual (Referencia)

```
Patrón: Strategy + Factory + Decorator Registration

config/l0_macro_sources.yaml          ← SSOT configuración
    ↓
src/core/factories/macro_extractor_factory.py  ← Factory Pattern
    ↓ (decorator @register_strategy)
airflow/dags/services/macro_extraction_strategies.py  ← 7 Strategies
    ↓
airflow/dags/services/macro_extraction_service.py  ← Orchestrator
    ↓
airflow/dags/l0_macro_unified.py  ← DAG definition
```

**Fuentes Actuales:**
| # | Source | Strategy Class | Config Section | Variables |
|---|--------|----------------|----------------|-----------|
| 1 | FRED | `FREDExtractionStrategy` | `sources.fred` | 14 |
| 2 | TwelveData | `TwelveDataExtractionStrategy` | `sources.twelvedata` | 4 |
| 3 | Investing | `InvestingExtractionStrategy` | `sources.investing` | 5 |
| 4 | BanRep | `BanRepExtractionStrategy` | `sources.banrep` | 6 |
| 5 | BCRP | `BCRPExtractionStrategy` | `sources.bcrp` | 1 |
| 6 | Fedesarrollo | `FedesarrolloExtractionStrategy` | `sources.fedesarrollo` | 2 |
| 7 | DANE | `DANEExtractionStrategy` | `sources.dane` | 2 |

---

## Lista de Problemas Identificados

### P0 - CRÍTICOS (Bloquean backtesting)

#### 1. Core CPI (CPILFESL) - Datos Incompletos
```
Severidad: CRÍTICA
Variable: infl_cpi_core_usa_m_cpilfesl
Fuente: FRED
Problema: Solo tiene datos desde 2024-02-01
Completitud: 31.7% (483/1525 filas desde 2020)
Impacto: Variable inutilizable para backtesting histórico
```

**Archivos a Modificar:**
| Archivo | Acción | Cambio |
|---------|--------|--------|
| `scripts/backfill_fred_cpilfesl.py` | CREAR | Script one-shot para backfill histórico |
| (ningún cambio en arquitectura) | - | FRED ya extrae esta serie correctamente |

**Solución:**
```python
# scripts/backfill_fred_cpilfesl.py (CREAR)
from fredapi import Fred
import psycopg2

def backfill_cpilfesl():
    """One-shot backfill of Core CPI from 2020-01-01."""
    fred = Fred(api_key=os.environ['FRED_API_KEY'])
    data = fred.get_series('CPILFESL', observation_start='2020-01-01')
    # INSERT INTO macro_indicators_daily ...
```

---

#### 2. Brent Oil - Valor Anómalo ($19.27)
```
Severidad: CRÍTICA
Variable: comm_oil_brent_glb_d_brent
Fuente: TwelveData
Problema: Último valor $19.27 (debería ser ~$80-85)
Fecha: 2026-01-20
Causa: Símbolo "BZ" puede estar devolviendo datos incorrectos
```

**Archivos a Modificar:**
| Archivo | Acción | Cambio |
|---------|--------|--------|
| `config/l0_macro_sources.yaml` | MODIFICAR | Línea 80: Cambiar símbolo de `BZ` a `BRENT` o verificar |
| `airflow/dags/services/macro_extraction_strategies.py` | MODIFICAR | Agregar validación de rangos en `TwelveDataExtractionStrategy` |

**Solución:**
```yaml
# config/l0_macro_sources.yaml (MODIFICAR línea 80)
# ANTES:
BZ: comm_oil_brent_glb_d_brent
# DESPUÉS (verificar cuál funciona):
BRENT: comm_oil_brent_glb_d_brent
# o
BRN: comm_oil_brent_glb_d_brent
```

**Agregar Validación (DRY - en base class):**
```python
# src/core/interfaces/macro_extractor.py (MODIFICAR)
# Agregar en BaseMacroExtractor:

VALUE_RANGES = {
    'comm_oil_brent_glb_d_brent': (20, 150),
    'comm_oil_wti_glb_d_wti': (20, 150),
    'volt_vix_usa_d_vix': (9, 80),
    'fxrt_index_dxy_usa_d_dxy': (80, 130),
}

def _validate_value(self, column: str, value: float) -> bool:
    """Validate value is within expected range."""
    if column in self.VALUE_RANGES:
        min_val, max_val = self.VALUE_RANGES[column]
        return min_val <= value <= max_val
    return True
```

---

### P1 - IMPORTANTES (Afectan calidad)

#### 3. Inconsistencia de Filas Entre Columnas
```
Severidad: ALTA
Problema: Diferencia de 80-90 filas entre columnas (94% vs 100%)
Detalle:
  - EMBI:   1,525 filas (100%)
  - DXY:    1,444 filas (94.7%)
  - COLCAP: 1,437 filas (94.2%)
Causa: Calendarios de festivos no sincronizados entre fuentes
```

**Archivos a Modificar:**
| Archivo | Acción | Cambio |
|---------|--------|--------|
| `config/l0_macro_sources.yaml` | MODIFICAR | Agregar `calendar` por indicador |
| `airflow/dags/services/macro_cleanup_service.py` | MODIFICAR | Cleanup por calendario, no global |

**Solución - Arquitectura:**
```yaml
# config/l0_macro_sources.yaml (AGREGAR sección)
indicator_calendars:
  # USA calendar (markets closed on US holidays)
  usa:
    - fxrt_index_dxy_usa_d_dxy
    - volt_vix_usa_d_vix
    - finc_bond_yield10y_usa_d_ust10y
    - finc_bond_yield2y_usa_d_dgs2

  # Colombia calendar
  colombia:
    - eqty_index_colcap_col_d_colcap
    - finc_rate_ibr_overnight_col_d_ibr
    - finc_bond_yield10y_col_d_col10y

  # Global (24/7 trading)
  global:
    - comm_metal_gold_glb_d_gold
    - comm_oil_brent_glb_d_brent
    - crsk_spread_embi_col_d_embi
```

---

#### 4. Forward Fill No Óptimo
```
Severidad: ALTA
Problema: FFILL configurado pero completitud es 94-95%
Esperado: Con FFILL debería ser ~99%+
Causa: FFILL se ejecuta ANTES del cleanup, orden incorrecto
```

**Archivos a Modificar:**
| Archivo | Acción | Cambio |
|---------|--------|--------|
| `airflow/dags/l0_macro_unified.py` | VERIFICAR | Task order: merge → cleanup → ffill |
| `airflow/dags/services/macro_merge_service.py` | MODIFICAR | Aumentar `max_days` para mensuales |

**Solución:**
```python
# airflow/dags/services/macro_merge_service.py (MODIFICAR)
# Líneas ~45-55

FFILL_LIMITS = {
    'daily': 5,       # Máximo 5 días para datos diarios
    'monthly': 35,    # Máximo 35 días para datos mensuales
    'quarterly': 95,  # Máximo 95 días para datos trimestrales
}

# Indicadores mensuales que necesitan más días de FFILL
MONTHLY_INDICATORS = [
    'infl_cpi_all_usa_m_cpiaucsl',
    'infl_cpi_core_usa_m_cpilfesl',
    'polr_fed_funds_usa_m_fedfunds',
    # ... etc
]
```

---

#### 5. Cleanup de Festivos No Diferenciado
```
Severidad: ALTA
Problema: Se eliminan festivos USA + Colombia para TODAS las columnas
Impacto: Pérdida innecesaria de datos válidos
```

**Archivos a Modificar:**
| Archivo | Acción | Cambio |
|---------|--------|--------|
| `airflow/dags/services/macro_cleanup_service.py` | REFACTORIZAR | Cleanup por calendar |
| `config/l0_macro_sources.yaml` | MODIFICAR | Agregar mapping columna → calendario |

**Solución - Refactor Cleanup Service:**
```python
# airflow/dags/services/macro_cleanup_service.py (MODIFICAR)

class MacroCleanupService:
    def __init__(self, config_path: str = '/opt/airflow/config/l0_macro_sources.yaml'):
        self.config = self._load_config(config_path)
        self.calendar_mapping = self._build_calendar_mapping()

    def _build_calendar_mapping(self) -> Dict[str, str]:
        """Build column -> calendar mapping from config."""
        mapping = {}
        for calendar, columns in self.config.get('indicator_calendars', {}).items():
            for col in columns:
                mapping[col] = calendar
        return mapping

    def cleanup_by_calendar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove non-trading days per column based on its calendar."""
        for column in df.columns:
            calendar = self.calendar_mapping.get(column, 'global')
            if calendar == 'usa':
                # Solo eliminar festivos USA para esta columna
                non_trading = self.get_usa_non_trading_days(df['fecha'])
                df.loc[df['fecha'].isin(non_trading), column] = np.nan
            elif calendar == 'colombia':
                # Solo eliminar festivos Colombia
                non_trading = self.get_col_non_trading_days(df['fecha'])
                df.loc[df['fecha'].isin(non_trading), column] = np.nan
            # 'global' no se toca
        return df
```

---

### P2 - MODERADOS (Mejoras de arquitectura)

#### 6. Variables Legacy Sin Extracción - PLAN COMPLETO

```
Severidad: MEDIA
Variables Afectadas: 4
Última Actualización: 2025-10-03 (108 días atrás)
```

##### Variables y Metodología:

| Variable | Descripción | Fuente | Método | Archivo Nuevo |
|----------|-------------|--------|--------|---------------|
| `polr_policy_rate_col_m_tpm` | TPM Mensual | BanRep SDMX API | REST API | `banrep_sdmx_strategy.py` |
| `rsbp_current_account_col_q_cacct_q` | Cuenta Corriente Q | BanRep SUAMECA | Selenium | Extender `BanRepExtractionStrategy` |
| `rsbp_fdi_inflow_col_q_fdiin_q` | FDI Entrada Q (IED) | BanRep SUAMECA | Selenium | Extender `BanRepExtractionStrategy` |
| `rsbp_fdi_outflow_col_q_fdiout_q` | FDI Salida Q (IDCE) | BanRep SUAMECA | Selenium | Extender `BanRepExtractionStrategy` |

---

##### OPCIÓN A: Extender BanRepExtractionStrategy (RECOMENDADO - DRY)

**Archivos a Modificar:**

| Archivo | Acción | Cambio |
|---------|--------|--------|
| `config/l0_macro_sources.yaml` | MODIFICAR | Agregar 4 series en `sources.banrep.indicators` |
| `airflow/dags/services/macro_extraction_strategies.py` | NO CAMBIAR | BanRepExtractionStrategy ya soporta múltiples series |

```yaml
# config/l0_macro_sources.yaml (AGREGAR en sources.banrep.indicators)

banrep:
  selenium_timeout_seconds: 30
  page_load_wait_seconds: 5
  chrome_options:
    - "--headless=new"
    - "--no-sandbox"
    - "--disable-dev-shm-usage"
  indicators:
    # === EXISTENTES (6) ===
    "241":
      column: finc_rate_ibr_overnight_col_d_ibr
      url: https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/241/...
      name: IBR Overnight
    # ... otras 5 existentes ...

    # === NUEVAS (4) - BALANZA DE PAGOS ===
    "SDMX_TPM":  # TPM Mensual via SDMX API
      column: polr_policy_rate_col_m_tpm
      url: https://totoro.banrep.gov.co/nsi-jax-ws/rest/data/ESTAT,DF_CBR_MONTHLY_HIST,1.0/all/ALL/
      name: TPM Policy Rate (Monthly)
      method: sdmx  # Flag para usar parser diferente

    "BP_CACCT":  # Cuenta Corriente Trimestral
      column: rsbp_current_account_col_q_cacct_q
      url: https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/BP_CUENTA_CORRIENTE_Q/...
      name: Cuenta Corriente Trimestral
      frequency: quarterly

    "BP_FDIIN":  # FDI Inflows
      column: rsbp_fdi_inflow_col_q_fdiin_q
      url: https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/BP_IED_INFLOWS_Q/...
      name: FDI Inflows (IED)
      frequency: quarterly

    "BP_FDIOUT":  # FDI Outflows
      column: rsbp_fdi_outflow_col_q_fdiout_q
      url: https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/BP_IDCE_OUTFLOWS_Q/...
      name: FDI Outflows (IDCE)
      frequency: quarterly
```

**Nota:** Las URLs exactas de SUAMECA deben verificarse navegando al portal.

---

##### OPCIÓN B: Crear Nueva Strategy para SDMX (Más limpio arquitectónicamente)

**Archivos a Crear/Modificar:**

| Archivo | Acción | Cambio |
|---------|--------|--------|
| `src/core/factories/macro_extractor_factory.py` | MODIFICAR | Agregar `MacroSource.BANREP_SDMX` |
| `airflow/dags/services/macro_extraction_strategies.py` | MODIFICAR | Agregar `BanRepSDMXStrategy` |
| `config/l0_macro_sources.yaml` | MODIFICAR | Agregar sección `banrep_sdmx` |
| `airflow/dags/services/macro_extraction_service.py` | MODIFICAR | Agregar XCOM mapping |
| `airflow/dags/contracts/l0_data_contracts.py` | MODIFICAR | Agregar `L0XComKeys.BANREP_SDMX_DATA` |

```python
# src/core/factories/macro_extractor_factory.py (MODIFICAR)
class MacroSource(str, Enum):
    FRED = "fred"
    TWELVEDATA = "twelvedata"
    INVESTING = "investing"
    BANREP = "banrep"
    BANREP_SDMX = "banrep_sdmx"  # NUEVO
    BCRP = "bcrp"
    FEDESARROLLO = "fedesarrollo"
    DANE = "dane"
```

```python
# airflow/dags/services/macro_extraction_strategies.py (AGREGAR)

@MacroExtractorFactory.register_strategy(MacroSource.BANREP_SDMX)
class BanRepSDMXExtractionStrategy(ConfigurableExtractor):
    """
    BanRep SDMX API extraction strategy.

    Extracts data from BanRep's official SDMX REST API.
    Used for monthly/quarterly series like TPM and Balance of Payments.

    API Docs: https://totoro.banrep.gov.co/nsi-jax-ws/
    """

    @property
    def source_name(self) -> str:
        return "banrep_sdmx"

    def extract(
        self,
        indicators: Dict[str, str],
        lookback_days: int,
        **kwargs
    ) -> ExtractionResult:
        """Extract data from BanRep SDMX API."""
        start_time = datetime.utcnow()
        results: Dict[str, Dict[str, float]] = {}
        errors: List[str] = []

        try:
            import xml.etree.ElementTree as ET
        except ImportError:
            errors.append("xml.etree not available")
            return self._create_result(results, errors, start_time)

        indicators_config = self.config.get('indicators', {})
        timeout = self.get_timeout()

        for flow_id, config in indicators_config.items():
            column = config.get('column')
            url = config.get('url')
            name = config.get('name', flow_id)

            if not column or not url:
                continue

            try:
                logger.info(f"[BanRep SDMX] Fetching {name}...")

                headers = {
                    'Accept': 'application/vnd.sdmx.data+xml;version=2.1',
                    'User-Agent': 'Mozilla/5.0'
                }

                response = requests.get(url, headers=headers, timeout=timeout)

                if response.status_code != 200:
                    errors.append(f"{name}: HTTP {response.status_code}")
                    continue

                # Parse SDMX XML
                root = ET.fromstring(response.content)

                # SDMX namespaces
                ns = {
                    'message': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/message',
                    'generic': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic'
                }

                # Extract observations
                extracted = 0
                for obs in root.findall('.//generic:Obs', ns):
                    obs_dim = obs.find('generic:ObsDimension', ns)
                    obs_val = obs.find('generic:ObsValue', ns)

                    if obs_dim is not None and obs_val is not None:
                        time_period = obs_dim.get('value')  # e.g., "2024-01"
                        value = float(obs_val.get('value'))

                        # Convert YYYY-MM to YYYY-MM-01
                        date_str = f"{time_period}-01" if len(time_period) == 7 else time_period

                        if date_str not in results:
                            results[date_str] = {}
                        results[date_str][column] = value
                        extracted += 1

                logger.info(f"  -> {extracted} records extracted")
                record_macro_ingestion_success('banrep_sdmx', column)

            except Exception as e:
                errors.append(f"{name}: {str(e)}")
                logger.error(f"[BanRep SDMX] Error {name}: {e}")
                record_macro_ingestion_error('banrep_sdmx', column, 'api_error')

        return self._create_result(results, errors, start_time)
```

```yaml
# config/l0_macro_sources.yaml (AGREGAR sección)

  # ---------------------------------------------------------------------------
  # BanRep SDMX API - Monthly/Quarterly Series
  # ---------------------------------------------------------------------------
  banrep_sdmx:
    request_timeout_seconds: 30
    indicators:
      TPM_MONTHLY:
        column: polr_policy_rate_col_m_tpm
        url: https://totoro.banrep.gov.co/nsi-jax-ws/rest/data/ESTAT,DF_CBR_MONTHLY_HIST,1.0/all/ALL/
        name: TPM Policy Rate (Monthly)

      CUENTA_CORRIENTE_Q:
        column: rsbp_current_account_col_q_cacct_q
        url: https://totoro.banrep.gov.co/nsi-jax-ws/rest/data/ESTAT,DF_BALANZA_PAGOS_Q,1.0/...
        name: Cuenta Corriente Trimestral

      FDI_INFLOWS_Q:
        column: rsbp_fdi_inflow_col_q_fdiin_q
        url: https://totoro.banrep.gov.co/nsi-jax-ws/rest/data/ESTAT,DF_IED_Q,1.0/...
        name: FDI Inflows (IED)

      FDI_OUTFLOWS_Q:
        column: rsbp_fdi_outflow_col_q_fdiout_q
        url: https://totoro.banrep.gov.co/nsi-jax-ws/rest/data/ESTAT,DF_IDCE_Q,1.0/...
        name: FDI Outflows (IDCE)

# Agregar a enabled_sources:
enabled_sources:
  - fred
  - twelvedata
  - investing
  - banrep
  - banrep_sdmx  # NUEVO
  - bcrp
  - fedesarrollo
  - dane
```

---

##### OPCIÓN C: Eliminar Columnas Legacy (Rápido pero pierde datos)

**Archivos a Modificar:**

| Archivo | Acción | Cambio |
|---------|--------|--------|
| `init-scripts/drop_legacy_columns.sql` | CREAR | Script SQL para eliminar columnas |
| `config/l0_macro_sources.yaml` | NO CAMBIAR | No agregar las columnas |

```sql
-- init-scripts/drop_legacy_columns.sql (CREAR)
ALTER TABLE macro_indicators_daily
DROP COLUMN IF EXISTS polr_policy_rate_col_m_tpm,
DROP COLUMN IF EXISTS rsbp_current_account_col_q_cacct_q,
DROP COLUMN IF EXISTS rsbp_fdi_inflow_col_q_fdiin_q,
DROP COLUMN IF EXISTS rsbp_fdi_outflow_col_q_fdiout_q;
```

---

##### Decisión Recomendada:

| Opción | Esfuerzo | Beneficio | Recomendación |
|--------|----------|-----------|---------------|
| **A: Extender BanRep** | 2h | Alto | ✅ Si URLs de SUAMECA funcionan |
| **B: Nueva Strategy SDMX** | 4h | Muy Alto | ✅ Arquitectura más limpia |
| **C: Eliminar** | 30min | Bajo | ❌ Pierde datos valiosos |

**Recomendación:** Opción B (Nueva Strategy SDMX) para TPM mensual + verificar si Balanza de Pagos está disponible via SDMX. Si no, extender BanRep Selenium (Opción A).

---

#### 7. Datos Históricos Pre-2020 Inconsistentes
```
Severidad: MEDIA
Problema: Algunas columnas tienen datos desde 1954, otras no
Impacto: No afecta período 2020+, pero confunde análisis
```

**Archivos a Modificar:**
| Archivo | Acción | Cambio |
|---------|--------|--------|
| `scripts/normalize_historical_range.sql` | CREAR | Script para limpiar datos < 2020 |

**Solución:**
```sql
-- scripts/normalize_historical_range.sql (CREAR)
-- Solo mantener datos desde 2020-01-01 para consistencia
DELETE FROM macro_indicators_daily WHERE fecha < '2020-01-01';

-- Verificar distribución
SELECT
    MIN(fecha) as min_fecha,
    MAX(fecha) as max_fecha,
    COUNT(*) as total_rows
FROM macro_indicators_daily;
```

---

#### 8. Sin Validación de Rangos de Valores
```
Severidad: MEDIA
Problema: No hay validación de que los valores estén en rangos esperados
Ejemplo: Brent Oil $19.27 pasó sin alerta
```

**Archivos a Modificar:**
| Archivo | Acción | Cambio |
|---------|--------|--------|
| `src/core/interfaces/macro_extractor.py` | MODIFICAR | Agregar `VALUE_RANGES` y `_validate_value()` |
| `config/l0_macro_sources.yaml` | MODIFICAR | Agregar sección `validation.ranges` |

**Solución (SSOT en config):**
```yaml
# config/l0_macro_sources.yaml (AGREGAR sección)

validation:
  ranges:
    # Commodities
    comm_oil_brent_glb_d_brent: [20, 150]
    comm_oil_wti_glb_d_wti: [20, 150]
    comm_metal_gold_glb_d_gold: [1000, 3000]
    comm_agri_coffee_glb_d_coffee: [50, 400]

    # Volatility
    volt_vix_usa_d_vix: [9, 80]

    # FX indices
    fxrt_index_dxy_usa_d_dxy: [80, 130]
    fxrt_spot_usdmxn_mex_d_usdmxn: [15, 25]
    fxrt_spot_usdclp_chl_d_usdclp: [700, 1100]

    # Yields (percentages)
    finc_bond_yield10y_usa_d_ust10y: [0, 10]
    finc_bond_yield10y_col_d_col10y: [5, 20]

    # Spreads
    crsk_spread_embi_col_d_embi: [100, 1000]
```

```python
# src/core/interfaces/macro_extractor.py (AGREGAR método)

def _validate_value(self, column: str, value: float) -> tuple[bool, str]:
    """
    Validate value is within expected range.

    Returns:
        (is_valid, warning_message)
    """
    ranges = self.config.get('validation', {}).get('ranges', {})

    if column in ranges:
        min_val, max_val = ranges[column]
        if not (min_val <= value <= max_val):
            return False, f"{column}={value} outside range [{min_val}, {max_val}]"

    return True, ""
```

---

### P3 - MENORES (Deuda técnica)

#### 9. Sin Monitoreo de Completitud
```
Severidad: BAJA
Problema: No hay alertas automáticas de completitud
```

**Archivos a Modificar:**
| Archivo | Acción | Cambio |
|---------|--------|--------|
| `airflow/dags/services/macro_merge_service.py` | MODIFICAR | Agregar check de completitud |
| `services/common/prometheus_metrics.py` | MODIFICAR | Agregar métrica `macro_completeness_ratio` |

---

#### 10. Columnas ffilled_from_date y release_date Subutilizadas
```
Severidad: BAJA
Problema: Estas columnas existen pero no se usan para validación
```

**Archivos a Modificar:**
| Archivo | Acción | Cambio |
|---------|--------|--------|
| `airflow/dags/services/macro_merge_service.py` | MODIFICAR | Usar en reportes |

---

#### 11. Sin Tests de Integración para Datos
```
Severidad: BAJA
Problema: No hay tests que validen calidad de datos post-extracción
```

**Archivos a Crear:**
| Archivo | Acción | Cambio |
|---------|--------|--------|
| `tests/integration/test_macro_data_quality.py` | CREAR | Tests de calidad |

---

## Resumen de Archivos por Prioridad

### P0 - CRÍTICOS (Esta semana)

| # | Archivo | Acción | Issue |
|---|---------|--------|-------|
| 1 | `scripts/backfill_fred_cpilfesl.py` | CREAR | P0-1 Core CPI |
| 2 | `config/l0_macro_sources.yaml` | MODIFICAR L80 | P0-2 Brent Symbol |
| 3 | `src/core/interfaces/macro_extractor.py` | MODIFICAR | P0-2 Validación |

### P1 - IMPORTANTES (Próxima semana)

| # | Archivo | Acción | Issue |
|---|---------|--------|-------|
| 4 | `config/l0_macro_sources.yaml` | AGREGAR sección `indicator_calendars` | P1-3, P1-5 |
| 5 | `airflow/dags/services/macro_cleanup_service.py` | REFACTORIZAR | P1-5 |
| 6 | `airflow/dags/services/macro_merge_service.py` | MODIFICAR FFILL limits | P1-4 |

### P2 - MODERADOS (Backlog)

| # | Archivo | Acción | Issue |
|---|---------|--------|-------|
| 7 | `src/core/factories/macro_extractor_factory.py` | AGREGAR `MacroSource.BANREP_SDMX` | P2-6 |
| 8 | `airflow/dags/services/macro_extraction_strategies.py` | AGREGAR `BanRepSDMXStrategy` | P2-6 |
| 9 | `config/l0_macro_sources.yaml` | AGREGAR sección `banrep_sdmx` | P2-6 |
| 10 | `airflow/dags/contracts/l0_data_contracts.py` | AGREGAR XCom key | P2-6 |
| 11 | `airflow/dags/services/macro_extraction_service.py` | AGREGAR XCOM mapping | P2-6 |

---

## Matriz de Impacto vs Esfuerzo (Actualizada)

```
                    BAJO ESFUERZO          ALTO ESFUERZO
                    ─────────────          ─────────────
ALTO IMPACTO    │  [2] Brent Symbol     │  [5] Cleanup diferenciado
                │  [8] Validación YAML  │  [3] Calendar mapping
                │                       │
────────────────┼───────────────────────┼────────────────────────────
                │                       │
BAJO IMPACTO    │  [7] Normalizar hist  │  [1] Backfill Core CPI
                │  [9] Monitoreo        │  [6] BanRep SDMX Strategy
                │  [10-11] Tests        │  [4] FFILL por frecuencia
```

---

## Orden de Ejecución Recomendado

### Sprint 1 (Esta semana) - Fixes Críticos

```bash
# 1. Fix Brent Oil (30 min)
# Modificar config/l0_macro_sources.yaml línea 80

# 2. Agregar validación de rangos (2h)
# Modificar src/core/interfaces/macro_extractor.py
# Agregar sección validation.ranges en YAML

# 3. Backfill Core CPI (2h)
# Crear scripts/backfill_fred_cpilfesl.py
# Ejecutar one-shot
```

### Sprint 2 (Próxima semana) - Cleanup y FFILL

```bash
# 4. Agregar calendar mapping (2h)
# Modificar config/l0_macro_sources.yaml

# 5. Refactorizar cleanup service (4h)
# Modificar airflow/dags/services/macro_cleanup_service.py

# 6. Ajustar FFILL limits (1h)
# Modificar airflow/dags/services/macro_merge_service.py
```

### Sprint 3 (Backlog) - Variables Legacy

```bash
# 7. Crear BanRep SDMX Strategy (4h)
# Agregar MacroSource.BANREP_SDMX
# Crear BanRepSDMXExtractionStrategy
# Configurar en YAML

# 8. Tests de calidad (2h)
# Crear tests/integration/test_macro_data_quality.py
```

---

## Principios de Diseño Aplicados

| Principio | Aplicación en Este Plan |
|-----------|------------------------|
| **SSOT** | Config en YAML, no código hardcodeado |
| **DRY** | Una Strategy base, validación en base class |
| **KISS** | Extender existente antes de crear nuevo |
| **Strategy Pattern** | Cada fuente = 1 Strategy class |
| **Factory Pattern** | Creación centralizada con decorator |
| **Open/Closed** | Agregar fuentes sin modificar código existente |

---

## Notas

- Documento creado: 2026-01-20
- Actualizado: 2026-01-20 (Cruce con arquitectura actual)
- Autor: Claude (Análisis automatizado)
- Próxima revisión: Post Sprint 1
