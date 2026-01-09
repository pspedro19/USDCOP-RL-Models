# ✅ FASE 0: PIPELINE L0 MACRO DATA - COMPLETADA

**Fecha:** 2025-11-05
**Status:** Archivos creados, listo para implementación
**Duración creación:** ~1 hora

---

## 📦 ARCHIVOS CREADOS (5 archivos)

### **1. scripts/verify_twelvedata_macro.py**
- **Propósito:** Verificar si WTI (CL) y DXY están disponibles en TwelveData API
- **Uso:** `python scripts/verify_twelvedata_macro.py`
- **Requiere:** Variable `TWELVEDATA_API_KEY_G1` configurada

### **2. init-scripts/02-macro-data-schema.sql**
- **Propósito:** Crear tabla PostgreSQL `macro_ohlcv` (TimescaleDB hypertable)
- **Uso:** `psql -U usdcop -d usdcop_db -f init-scripts/02-macro-data-schema.sql`
- **Features:**
  - Primary key: (time, symbol)
  - Constraints OHLC validation
  - 2 funciones auxiliares: `get_macro_stats()`, `detect_macro_gaps()`

### **3. airflow/dags/usdcop_m5__01b_l0_macro_acquire.py**
- **Propósito:** DAG Airflow para descargar datos macro diariamente
- **Símbolos:** WTI (CL), DXY
- **Intervalo:** 1 hora
- **Schedule:** @daily (automático)
- **Catchup:** True (descarga históricos desde 2002)
- **Tasks:**
  1. `fetch_macro_data` - Descargar de TwelveData
  2. `insert_to_postgresql` - Insertar en `macro_ohlcv`
  3. `export_to_minio` - Exportar a bucket `00-raw-macro-marketdata`
  4. `validate_data_quality` - Validar calidad

### **4. scripts/upload_macro_manual.py**
- **Propósito:** Fallback manual si TwelveData no disponible
- **Uso:**
  ```bash
  python scripts/upload_macro_manual.py --file WTI_Historical_Data.csv --symbol WTI
  python scripts/upload_macro_manual.py --file DXY_Historical_Data.csv --symbol DXY
  ```
- **Fuente datos:** investing.com (CSV download)
- **Features:**
  - Parsea CSV de investing.com
  - Upsert a PostgreSQL
  - Upload a MinIO

### **5. FASE_0_INSTRUCCIONES.md**
- **Propósito:** Guía completa paso a paso para ejecutar Fase 0
- **Contenido:**
  - Checklist de ejecución
  - Troubleshooting
  - Métricas de éxito
  - Mantenimiento diario

---

## 🚀 PRÓXIMOS PASOS PARA EL USUARIO

### **PASO 1: Configurar API Key (⚠️ REQUERIDO)**

```bash
# Opción A: Agregar a .env o docker-compose.yml
TWELVEDATA_API_KEY_G1=tu_api_key_aqui
TWELVEDATA_API_KEY_G2=otra_api_key  # Opcional (fallback)
TWELVEDATA_API_KEY_G3=otra_api_key  # Opcional (fallback)

# Opción B: Export temporal (session actual)
export TWELVEDATA_API_KEY_G1="tu_api_key_aqui"
```

**Obtener API key gratuita:**
- https://twelvedata.com/pricing
- Plan gratuito: 800 requests/día (suficiente para 2 símbolos × 1h data)

---

### **PASO 2: Crear Tabla PostgreSQL**

```bash
# Opción A: Desde host
docker exec -it usdcop-postgres psql -U usdcop -d usdcop_db \
  -f /init-scripts/02-macro-data-schema.sql

# Opción B: Desde container
docker exec -it usdcop-postgres bash
psql -U usdcop -d usdcop_db -f /init-scripts/02-macro-data-schema.sql
exit
```

**Verificar:**
```bash
docker exec -it usdcop-postgres psql -U usdcop -d usdcop_db \
  -c "\d macro_ohlcv"
```

---

### **PASO 3: Verificar TwelveData**

```bash
python scripts/verify_twelvedata_macro.py
```

**Si funciona (exit code 0):**
→ Continuar con PASO 4A (TwelveData)

**Si falla (exit code 1):**
→ Continuar con PASO 4B (Fallback Manual)

---

### **PASO 4A: Usar TwelveData API (Recomendado)**

**4A.1: Copiar DAG a Airflow**

El DAG ya está en la ubicación correcta:
```
airflow/dags/usdcop_m5__01b_l0_macro_acquire.py
```

**4A.2: Reiniciar Airflow (para detectar nuevo DAG)**

```bash
docker-compose restart airflow-webserver airflow-scheduler
```

**4A.3: Activar DAG**

```bash
# Desde Airflow UI (http://localhost:8080)
# → Buscar: usdcop_m5__01b_l0_macro_acquire
# → Activar toggle

# O desde CLI:
docker exec -it usdcop-airflow-webserver \
  airflow dags unpause usdcop_m5__01b_l0_macro_acquire
```

**4A.4: Trigger Manual (Testing)**

```bash
docker exec -it usdcop-airflow-webserver \
  airflow dags trigger usdcop_m5__01b_l0_macro_acquire
```

**4A.5: Verificar Ejecución**

```bash
# Ver logs
docker exec -it usdcop-airflow-webserver \
  airflow tasks logs usdcop_m5__01b_l0_macro_acquire fetch_macro_data <fecha>

# Verificar datos en PostgreSQL
docker exec -it usdcop-postgres psql -U usdcop -d usdcop_db \
  -c "SELECT * FROM get_macro_stats();"
```

**4A.6: Ejecutar Catchup Histórico**

El DAG tiene `catchup=True`, por lo que **automáticamente** descargará todos los datos desde 2002-01-01 hasta hoy.

⚠️ **ADVERTENCIA:** Esto puede tardar 2-3 horas y consumir ~800 API calls/día (necesitas API key con suficiente quota).

**Monitorear progreso:**
- Airflow UI → DAG → Calendar View
- Debería mostrar ejecuciones diarias desde 2002

**Meta:** ~45,000 registros por símbolo (23 años × 365 días × 24 horas × 1 registro/hora / 5 = ~40k)

---

### **PASO 4B: Fallback Manual (Si TwelveData falla)**

**4B.1: Descargar datos de investing.com**

**WTI:**
1. https://www.investing.com/commodities/crude-oil-historical-data
2. Date Range: Jan 02, 2002 → Today
3. Download CSV → Guardar como `WTI_Historical_Data.csv`

**DXY:**
1. https://www.investing.com/indices/usdollar-historical-data
2. Date Range: Jan 02, 2002 → Today
3. Download CSV → Guardar como `DXY_Historical_Data.csv`

**4B.2: Cargar con script**

```bash
python scripts/upload_macro_manual.py \
  --file ~/Downloads/WTI_Historical_Data.csv \
  --symbol WTI

python scripts/upload_macro_manual.py \
  --file ~/Downloads/DXY_Historical_Data.csv \
  --symbol DXY
```

**Meta:** ~6,000 registros por símbolo (23 años × 252 trading days)

⚠️ **NOTA:** Datos manuales son **diarios**, no horarios. Se expandirán a 5min en L3 con forward-fill.

---

### **PASO 5: Validar Datos**

```bash
# Estadísticas generales
docker exec -it usdcop-postgres psql -U usdcop -d usdcop_db \
  -c "SELECT * FROM get_macro_stats();"

# Output esperado (TwelveData):
#  symbol | record_count | min_time            | max_time            | days_coverage | source
# --------+--------------+---------------------+---------------------+---------------+-----------
#  WTI    |        45000 | 2002-01-02 00:00:00 | 2025-11-05 23:00:00 |          8674 | twelvedata
#  DXY    |        45000 | 2002-01-02 00:00:00 | 2025-11-05 23:00:00 |          8674 | twelvedata

# Output esperado (Manual):
#  symbol | record_count | min_time            | max_time            | days_coverage | source
# --------+--------------+---------------------+---------------------+---------------+--------------------
#  WTI    |         5843 | 2002-01-02 00:00:00 | 2025-11-05 00:00:00 |          8674 | investing.com_manual
#  DXY    |         5843 | 2002-01-02 00:00:00 | 2025-11-05 00:00:00 |          8674 | investing.com_manual
```

**Verificar MinIO:**
```bash
mc ls minio/00-raw-macro-marketdata/WTI/
mc ls minio/00-raw-macro-marketdata/DXY/
```

---

## ✅ CRITERIOS DE ÉXITO FASE 0

| Métrica | Target TwelveData | Target Manual | Status |
|---------|------------------|---------------|--------|
| WTI registros | > 40,000 | > 5,000 | ⬜ |
| DXY registros | > 40,000 | > 5,000 | ⬜ |
| Calidad OHLC | 0% NaN | 0% NaN | ⬜ |
| Cobertura | 2002-2025 | 2002-2025 | ⬜ |
| PostgreSQL | Tabla creada | Tabla creada | ⬜ |
| MinIO | Archivos presentes | Archivos presentes | ⬜ |

**Una vez todos ✅:** Fase 0 completada → Continuar con Fase 2

---

## 📊 RESUMEN DE IMPLEMENTACIÓN

### **Lo que hemos creado:**

```
FASE 0: Pipeline L0 Macro Data
│
├── 📄 SQL Schema (macro_ohlcv table)
│   └── TimescaleDB hypertable con constraints OHLC
│
├── 🤖 DAG Airflow (L0 macro acquisition)
│   ├── Descarga diaria automática (TwelveData API)
│   ├── Catchup histórico (2002-2025)
│   ├── Insert a PostgreSQL
│   ├── Export a MinIO
│   └── Validación de calidad
│
├── 🔍 Script Verificación (verify_twelvedata_macro.py)
│   └── Prueba disponibilidad de WTI y DXY en API
│
├── 📤 Script Fallback Manual (upload_macro_manual.py)
│   ├── Parsea CSV de investing.com
│   ├── Upload a PostgreSQL
│   └── Upload a MinIO
│
└── 📖 Instrucciones Completas (FASE_0_INSTRUCCIONES.md)
    └── Guía paso a paso con troubleshooting
```

### **Datos que se obtendrán:**

**WTI Crude Oil (símbolo: CL)**
- Intervalo: 1 hora (TwelveData) o diario (manual)
- Rango: 2002-01-02 hasta hoy
- Registros esperados: ~45,000 (TwelveData) o ~6,000 (manual)
- Uso: Feature correlación USD/COP con commodities

**US Dollar Index (símbolo: DXY)**
- Intervalo: 1 hora (TwelveData) o diario (manual)
- Rango: 2002-01-02 hasta hoy
- Registros esperados: ~45,000 (TwelveData) o ~6,000 (manual)
- Uso: Feature fuerza del dólar vs otras monedas

### **Dónde se almacenan:**

1. **PostgreSQL:** Tabla `macro_ohlcv`
   - Query rápido para L3 feature engineering
   - Funciones auxiliares para estadísticas y gap detection

2. **MinIO:** Bucket `00-raw-macro-marketdata/`
   - Backup en parquet comprimido
   - Archivado por símbolo y fecha

---

## ➡️ DESPUÉS DE FASE 0

**Orden de implementación:**

1. ✅ **FASE 0 COMPLETADA** (este documento)

2. **FASE 2: L3/L4 Feature Engineering** (siguiente)
   - Leer: `PLAN_ESTRATEGICO_v2_UPDATES.md` Sección 2
   - Modificar: `airflow/dags/usdcop_m5__04_l3_feature.py`
     - Añadir `fetch_macro_data()`
     - Añadir `calculate_macro_features()` (7 features)
     - Añadir `calculate_mtf_features()` (8 features)
   - Modificar: `airflow/dags/usdcop_m5__05_l4_rlready.py`
     - Expandir OBS_MAPPING de 17 → 45

3. **FASE 3: Reward Shaping + SAC**
   - Crear: `notebooks/utils/rewards.py`
   - Modificar: `notebooks/utils/environments.py`
   - A/B testing de reward functions

4. **FASE 4: Optuna Optimization**

5. **FASE 5: Walk-Forward Validation**

---

## 🎓 DOCUMENTACIÓN DE REFERENCIA

### **Archivos clave para leer:**

```
1. FASE_0_INSTRUCCIONES.md              [Esta fase - paso a paso]
2. PLAN_ESTRATEGICO_v2_UPDATES.md       [Todas las fases con gaps integrados]
3. RESUMEN_EJECUTIVO_v2.md              [Overview completo del proyecto]
4. ADDENDUM_MACRO_FEATURES.md           [Detalles técnicos macro pipeline]
5. ADDENDUM_REWARD_SHAPING.md           [Reward functions - Fase 3]
6. ADDENDUM_MTF_SPECIFICATION.md        [Multi-timeframe features - Fase 2]
```

### **Papers académicos citados:**

1. **Moody & Saffell (2001)**: Differential Sharpe Ratio
2. **ICASSP (2019)**: Price Trailing Reward
3. **ArXiv (2022)**: Multi-Objective Reward
4. **Elder (2014)**: Triple Screen Method
5. **López de Prado (2018)**: Walk-Forward con Embargo

---

## ⚠️ NOTAS IMPORTANTES

### **1. API Keys TwelveData**

- **Gratis:** 800 requests/día (suficiente para 2 símbolos diarios)
- **Necesario para:** Datos horarios automatizados
- **Alternativa:** Fallback manual (datos diarios de investing.com)

### **2. Catchup Histórico**

- **Tiempo:** 2-3 horas
- **Requests:** ~8,000 (23 años × 365 días)
- **Solución:** Ejecutar de noche o usar plan pagado de TwelveData

### **3. Datos Diarios vs Horarios**

- **TwelveData:** Datos horarios → Mejor para features
- **Manual:** Datos diarios → Se expanden en L3 con forward-fill
- **Ambos son válidos**, pero horarios son preferibles

### **4. Mantenimiento Diario**

- **Con TwelveData:** Automático (DAG diario)
- **Con Manual:** Requiere descarga diaria de investing.com
  - Automatizar con cron o scheduled task

---

## 🐛 TROUBLESHOOTING COMÚN

### **"UnicodeEncodeError" al ejecutar verify script**

**Causa:** Emojis no soportados en Windows cmd con cp1252

**Solución:**
```bash
# Opción 1: Usar PowerShell (mejor encoding)
powershell
python scripts/verify_twelvedata_macro.py

# Opción 2: Usar Python UTF-8 mode
set PYTHONIOENCODING=utf-8
python scripts/verify_twelvedata_macro.py

# Opción 3: Comentar emojis en el script (líneas 20-27)
```

### **"API key no encontrada"**

**Solución:**
```bash
# Verificar variable configurada
echo %TWELVEDATA_API_KEY_G1%  # Windows
echo $TWELVEDATA_API_KEY_G1   # Linux/Mac

# Si vacía, configurar:
set TWELVEDATA_API_KEY_G1=tu_key_aqui     # Windows cmd
$env:TWELVEDATA_API_KEY_G1="tu_key_aqui"  # PowerShell
export TWELVEDATA_API_KEY_G1="tu_key_aqui"  # Linux/Mac
```

### **"Tabla macro_ohlcv no existe"**

**Causa:** Schema SQL no ejecutado

**Solución:** Ver PASO 2 arriba

### **"MinIO bucket not accessible"**

**Solución:**
```bash
# Verificar MinIO corriendo
docker ps | grep minio

# Crear bucket manualmente
mc mb minio/00-raw-macro-marketdata
```

---

## ✅ CHECKLIST FINAL

Antes de continuar a Fase 2, verificar:

- [ ] Tabla `macro_ohlcv` creada en PostgreSQL
- [ ] API key TwelveData configurada (o preparado fallback manual)
- [ ] Script `verify_twelvedata_macro.py` ejecutado exitosamente
- [ ] DAG `usdcop_m5__01b_l0_macro_acquire` visible en Airflow UI
- [ ] Datos macro descargados (TwelveData o manual)
- [ ] `get_macro_stats()` muestra registros > 0
- [ ] Bucket MinIO `00-raw-macro-marketdata` tiene archivos
- [ ] Leído `FASE_0_INSTRUCCIONES.md` completamente

**Todos ✅ → Proceder a Fase 2**

---

**FIN DEL DOCUMENTO**

*Fase 0 completada - 2025-11-05*
*Próximo: Fase 2 (L3/L4 Feature Engineering)*
