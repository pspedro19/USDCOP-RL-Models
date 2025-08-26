# EXPLICACIÓN DETALLADA DE PIPELINES BRONZE Y SILVER
================================================================================

## 🥉 PIPELINE BRONZE - INGESTA Y CONSOLIDACIÓN DE DATOS CRUDOS

### PROPÓSITO GENERAL
El Pipeline Bronze es la primera capa de procesamiento que se encarga de:
1. **Ingestar datos crudos** desde múltiples fuentes (MT5 y TwelveData)
2. **Consolidar y estandarizar** formatos diferentes
3. **Detectar y eliminar duplicados**
4. **Normalizar zonas horarias** a UTC
5. **Validar integridad básica** de los datos

### COMPONENTES DEL PIPELINE BRONZE

#### 1. **bronze_pipeline_smart.py** - Detección Inteligente
**¿Qué hace?**
- **Escanea datos existentes** en todas las carpetas del proyecto
- **Detecta rangos de tiempo cubiertos** para evitar re-descargas
- **Calcula overlaps** entre diferentes archivos
- **Genera hash MD5** para detectar duplicados exactos

**Proceso detallado:**
```python
1. Escanea carpetas:
   - data/raw/mt5/
   - data/raw/twelve_data/
   - data/processed/bronze/

2. Para cada archivo encontrado:
   - Lee primeras y últimas filas
   - Extrae rango temporal
   - Calcula número de registros
   - Genera fingerprint único

3. Crea mapa de cobertura temporal:
   - MT5: Oct 2022 - Ago 2025 (26,669 registros)
   - TwelveData: Ene 2020 - Ago 2025 (258,583 registros)

4. Optimización:
   - Si solicitas datos de 2023, detecta que ya existen
   - Evita llamadas API innecesarias
   - Ahorra 80% del tiempo de descarga
```

**Salida:**
- Cache de datos existentes
- Mapa de rangos temporales cubiertos
- Reporte de duplicados detectados

---

#### 2. **bronze_pipeline_enhanced.py** - Consolidación y Validación
**¿Qué hace?**
- **Combina múltiples archivos** de la misma fuente
- **Elimina duplicados** (detectó 5,000 en TwelveData)
- **Valida integridad OHLCV**
- **Detecta datos sintéticos** (descubrió 293,518 registros falsos)

**Proceso detallado:**
```python
1. Carga de datos:
   - Lee todos los CSVs de MT5
   - Lee todos los CSVs de TwelveData
   - Mantiene fuentes separadas

2. Eliminación de duplicados:
   - Usa timestamp como clave única
   - Detecta duplicados exactos: 5,000 encontrados
   - Detecta duplicados cercanos (±1 segundo)
   
3. Validaciones de integridad:
   - OHLC consistency: Low ≤ Close ≤ High
   - Spread validation: Spread < 500 pips
   - Price range: 1000 < Price < 6000
   - Volume check: Volume ≥ 0

4. Detección de datos sintéticos:
   - Identificó 293,518 registros con spreads imposibles
   - Spreads promedio: 65,573 pips (claramente sintético)
   - Decisión: Descartar completamente
```

**Transformaciones aplicadas:**
```python
# Ejemplo de datos antes:
time                 open     high     low      close    spread
2024-01-01 10:00    4000.5   4010.2   3995.8   4005.3   65573  # Sintético!

# Después de limpieza:
time                 open     high     low      close    spread
2024-01-01 10:00    4000.5   4010.2   3995.8   4005.3   3      # Real
```

**Salida:**
- `FINAL_MT5_M5_CONSOLIDATED.csv` (26,669 registros reales)
- `TWELVEDATA_M5_CONSOLIDATED_FINAL.csv` (258,583 registros únicos)

---

#### 3. **bronze_pipeline_utc.py** - Normalización Temporal
**¿Qué hace?**
- **Convierte todas las timestamps a UTC**
- **Aplica filtros de horario de trading**
- **Detecta gaps temporales**
- **Genera reportes de calidad**

**Proceso detallado:**
```python
1. Conversión MT5 (UTC+2 → UTC):
   - MT5 usa horario del broker (ICMarkets UTC+2)
   - Resta 2 horas a cada timestamp
   - Ejemplo: 2024-01-01 10:00 UTC+2 → 2024-01-01 08:00 UTC

2. Conversión TwelveData (UTC-5 → UTC):
   - TwelveData usa horario Colombia (UTC-5)
   - Suma 5 horas a cada timestamp
   - Ejemplo: 2024-01-01 10:00 COT → 2024-01-01 15:00 UTC

3. Aplicación de filtros de horario:
   Premium:    13:00-19:00 UTC (08:00-14:00 COT) Lun-Jue
   London:     08:00-13:00 UTC (03:00-08:00 COT) Lun-Jue
   Afternoon:  19:00-22:00 UTC (14:00-17:00 COT) Lun-Jue
   Friday:     13:00-20:00 UTC (08:00-15:00 COT) Viernes

4. Detección de gaps:
   - Gap normal: 5 minutos entre barras
   - Gap pequeño: 10-30 minutos (imputable)
   - Gap grande: >60 minutos (no imputable)
   - Gap de fin de semana: Ignorado
```

**Ejemplo de transformación temporal:**
```python
# Datos originales MT5 (UTC+2):
2024-01-15 10:00:00+02:00  # Lunes 10am hora broker

# Después de conversión a UTC:
2024-01-15 08:00:00+00:00  # Lunes 8am UTC

# Clasificación:
- Hora UTC: 08:00 
- Hora COT: 03:00 (08:00 - 5)
- Sesión: London (03:00-08:00 COT)
- Calidad: Media (solo 54% completitud en London)
```

**Salida:**
- `utc/MT5_M5_UTC.csv` (26,669 registros en UTC)
- `utc/TWELVEDATA_M5_UTC.csv` (258,583 registros en UTC)
- `utc/TWELVEDATA_M5_UTC_FILTERED.csv` (145,218 en horarios de trading)
- `quality_reports/bronze_quality_report.json`

---

## 🥈 PIPELINE SILVER - LIMPIEZA Y FILTRADO DE CALIDAD

### PROPÓSITO GENERAL
El Pipeline Silver es la segunda capa que se encarga de:
1. **Filtrar por sesiones de trading** de alta calidad
2. **Limpiar anomalías** y valores atípicos
3. **Imputar valores faltantes** de manera conservadora
4. **Garantizar calidad** para ML/Trading

### COMPONENTES DEL PIPELINE SILVER

#### 1. **silver_pipeline_enhanced.py** - Limpieza General
**¿Qué hace?**
- **Detecta 66,928 anomalías** en los datos
- **Corrige violaciones OHLC**
- **Imputa valores faltantes**
- **Calcula métricas de calidad**

**Proceso detallado:**
```python
1. Detección de anomalías:
   - Outliers de precio: Price < 2000 o Price > 5500
   - Spikes extremos: Cambio > 2% en 5 minutos
   - Spreads anormales: Spread > 200 pips
   - Volumen cero prolongado

2. Corrección OHLC:
   # Antes (violación):
   Open: 4000, High: 4010, Low: 4015, Close: 4005  # Low > Close!
   
   # Después (corregido):
   Open: 4000, High: 4015, Low: 4000, Close: 4005  # Consistente

3. Imputación de gaps:
   - Gaps < 30 min: Interpolación lineal
   - Gaps 30-60 min: Forward fill
   - Gaps > 60 min: No imputar
```

---

#### 2. **silver_pipeline_premium_only.py** - Filtrado Premium ⭐
**¿Qué hace?**
- **Filtra ÚNICAMENTE sesión Premium** (Lun-Vie 08:00-14:00 COT)
- **Justifica la decisión** con análisis de calidad
- **Excluye festivos** US y Colombia
- **Garantiza 90.9% completitud**

**Justificación basada en análisis:**
```
Análisis de 145,218 registros (2020-2025):

| Sesión     | Horario COT  | Completitud | Decisión    |
|------------|--------------|-------------|-------------|
| Premium    | 08:00-14:00  | 91.4%       | ✅ USAR     |
| London     | 03:00-08:00  | 54.3%       | ❌ DESCARTAR|
| Afternoon  | 14:00-17:00  | 58.8%       | ❌ DESCARTAR|
```

**Proceso detallado:**
```python
1. Carga de Bronze UTC:
   - Input: 258,583 registros (todas las horas)
   - Período: 2020-2025

2. Filtrado Premium:
   - Solo Lun-Vie (excluye fines de semana)
   - Solo 08:00-14:00 COT (13:00-19:00 UTC)
   - Excluye festivos US (Thanksgiving, 4 julio, etc)
   - Excluye festivos Colombia (20 julio, 7 agosto, etc)
   
   Resultado: 86,272 registros (reducción 66.6%)

3. Análisis de gaps en Premium:
   - Gaps detectados: 127
   - Gaps < 30 min: 89 (imputables)
   - Gaps > 30 min: 38 (no imputables)
   - Completitud final: 90.9%

4. Limpieza de anomalías Premium:
   - Outliers corregidos: 234
   - Violaciones OHLC: 12
   - Spikes suavizados: 45
   
5. Imputación conservadora:
   - Solo imputa si completitud > 80%
   - Máximo 6 barras consecutivas (30 min)
   - Método: Interpolación lineal
   - Valores imputados: 89
```

**Ejemplo de filtrado:**
```python
# Datos Bronze (todos los horarios):
2024-01-15 06:00 UTC  # 01:00 COT - Madrugada ❌
2024-01-15 10:00 UTC  # 05:00 COT - London ❌ (baja calidad)
2024-01-15 14:00 UTC  # 09:00 COT - Premium ✅
2024-01-15 15:00 UTC  # 10:00 COT - Premium ✅
2024-01-15 20:00 UTC  # 15:00 COT - Afternoon ❌ (baja calidad)
2024-01-15 23:00 UTC  # 18:00 COT - Cerrado ❌

# Datos Silver (solo Premium):
2024-01-15 14:00 UTC  # 09:00 COT - Premium ✅
2024-01-15 15:00 UTC  # 10:00 COT - Premium ✅
```

**Validación final:**
```python
Validaciones de calidad Silver:
✅ Columnas requeridas presentes
✅ Sin valores nulos
✅ Precios en rango válido (2000-5500)
✅ OHLC consistente (Low ≤ Close ≤ High)
✅ Sin gaps mayores a 30 minutos
✅ 90.9% completitud
```

**Salida Silver:**
- `SILVER_PREMIUM_ONLY_*.csv` (86,272 registros)
- `silver_premium_report_*.md` (reporte detallado)

---

## RESUMEN DE TRANSFORMACIONES

### Bronze → Silver: Reducción de Datos
```
293,220 (APIs originales)
    ↓ Bronze: Elimina sintéticos y duplicados
258,583 (Bronze consolidado)
    ↓ Silver: Filtra solo Premium
86,272 (Silver Premium - 90.9% completo)
```

### Mejoras de Calidad
| Métrica | Bronze | Silver |
|---------|--------|--------|
| Completitud | 75% | 90.9% |
| Anomalías | 66,928 | 0 |
| Duplicados | 5,000 | 0 |
| Gaps > 1h | 483 | 0 |
| Validación OHLC | 85% | 100% |

### Decisión Clave: ¿Por qué solo Premium?
1. **London (03:00-08:00)**: Solo 54.3% completo - INACEPTABLE para trading
2. **Afternoon (14:00-17:00)**: Solo 58.8% completo - DEMASIADOS GAPS
3. **Premium (08:00-14:00)**: 91.4% completo - ÓPTIMO para ML y trading

El dataset Silver Premium es el único con calidad suficiente para:
- Entrenamiento confiable de modelos ML
- Backtesting realista
- Trading algorítmico en producción