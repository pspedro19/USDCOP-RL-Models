# 📋 FASE 0: PIPELINE L0 MACRO DATA - INSTRUCCIONES

**Versión:** 1.0
**Fecha:** 2025-11-05
**Duración estimada:** 2-3 días
**Objetivo:** Adquirir datos macro (WTI, DXY) para features correlacionadas con USD/COP

---

## 🎯 RESUMEN

Esta fase crea el pipeline L0 para datos macro económicos:
- **WTI Crude Oil** (símbolo: CL)
- **US Dollar Index** (símbolo: DXY)

Los datos se almacenan en:
1. PostgreSQL tabla `macro_ohlcv` (TimescaleDB hypertable)
2. MinIO bucket `00-raw-macro-marketdata`

---

## 📁 ARCHIVOS CREADOS

```
scripts/
  ├── verify_twelvedata_macro.py       [Verificar API TwelveData]
  └── upload_macro_manual.py            [Fallback manual investing.com]

init-scripts/
  └── 02-macro-data-schema.sql          [Schema PostgreSQL]

airflow/dags/
  └── usdcop_m5__01b_l0_macro_acquire.py  [DAG L0 macro]
```

---

## ✅ CHECKLIST DE EJECUCIÓN

### **Paso 1: Crear Tabla PostgreSQL** (5 min)

```bash
# Conectar a PostgreSQL
docker exec -it usdcop-postgres bash

# Dentro del container
psql -U usdcop -d usdcop_db -f /init-scripts/02-macro-data-schema.sql

# Salir del container
exit
```

**Verificar tabla creada:**
```bash
docker exec -it usdcop-postgres psql -U usdcop -d usdcop_db \
  -c "SELECT * FROM get_macro_stats();"
```

**Output esperado:**
```
 symbol | record_count | min_time | max_time | days_coverage | source
--------+--------------+----------+----------+---------------+--------
(0 rows)
```

✅ **Criterio de éxito:** Tabla `macro_ohlcv` existe y está vacía

---

### **Paso 2: Verificar TwelveData API** (2 min)

```bash
# Configurar API key (si no está ya configurada)
export TWELVEDATA_API_KEY_G1="tu_api_key_aqui"

# Ejecutar verificación
python scripts/verify_twelvedata_macro.py
```

**Escenario A: TwelveData DISPONIBLE ✅**

Output esperado:
```
============================================================
        VERIFICACIÓN TWELVEDATA API - MACRO DATA
============================================================

✅ API key encontrada: abcd1234...

------------------------------------------------------------
ℹ️  Verificando WTI Crude Oil (CL)...
✅ CL encontrado:
  - Nombre: Crude Oil WTI Futures
  - Tipo: Commodity
  - Exchange: NYMEX
  - Currency: USD
ℹ️  Probando descarga de datos históricos...
✅ Datos obtenidos correctamente:
  - Registros: 48
  - Primer timestamp: 2025-10-29 09:00:00
  - Último timestamp: 2025-11-05 12:00:00
  - Último precio: 75.23
✅ Todos los campos OHLCV presentes

------------------------------------------------------------
ℹ️  Verificando US Dollar Index (DXY)...
✅ DXY encontrado:
  - Nombre: US Dollar Index
  - Tipo: Index
  - Exchange: ICE
  - Currency: USD
ℹ️  Probando descarga de datos históricos...
✅ Datos obtenidos correctamente:
  - Registros: 48
  - Primer timestamp: 2025-10-29 09:00:00
  - Último timestamp: 2025-11-05 12:00:00
  - Último precio: 103.45
✅ Todos los campos OHLCV presentes

============================================================
                RESUMEN DE VERIFICACIÓN
============================================================

✅ CL (WTI Crude Oil): DISPONIBLE
    Último precio: 75.23
    Timestamp: 2025-11-05 12:00:00
✅ DXY (US Dollar Index): DISPONIBLE
    Último precio: 103.45
    Timestamp: 2025-11-05 12:00:00

============================================================
✅ DECISIÓN: Usar TwelveData API para macro data
ℹ️  Próximo paso: Crear DAG usdcop_m5__01b_l0_macro_acquire.py
```

→ **Continuar con Paso 3A (TwelveData)**

---

**Escenario B: TwelveData NO DISPONIBLE ❌**

Output esperado:
```
❌ CL (WTI Crude Oil): NO DISPONIBLE
    Error: Symbol not found
❌ DXY (US Dollar Index): NO DISPONIBLE
    Error: API rate limit exceeded

============================================================
❌ DECISIÓN: TwelveData NO disponible para todos los símbolos
⚠️  Usar fallback manual desde investing.com
ℹ️  Próximo paso: Ejecutar scripts/upload_macro_manual.py
```

→ **Continuar con Paso 3B (Fallback Manual)**

---

### **Paso 3A: Configurar DAG TwelveData** (SI DISPONIBLE)

El DAG ya está creado en `airflow/dags/usdcop_m5__01b_l0_macro_acquire.py`

**Verificar DAG en Airflow UI:**
```bash
# Abrir Airflow UI
http://localhost:8080

# Buscar DAG: usdcop_m5__01b_l0_macro_acquire
# Estado: Debería aparecer pausado inicialmente
```

**Activar DAG:**
```bash
# Opción 1: Desde Airflow UI
# → Clic en toggle para activar

# Opción 2: Desde CLI
docker exec -it usdcop-airflow-webserver bash
airflow dags unpause usdcop_m5__01b_l0_macro_acquire
exit
```

**Trigger manual (testing):**
```bash
# Ejecutar para hoy
docker exec -it usdcop-airflow-webserver bash
airflow dags trigger usdcop_m5__01b_l0_macro_acquire --exec-date 2025-11-05
exit
```

**Monitorear ejecución:**
```bash
# Ver logs del DAG
docker exec -it usdcop-airflow-webserver bash
airflow tasks logs usdcop_m5__01b_l0_macro_acquire fetch_macro_data 2025-11-05
exit
```

**Verificar datos insertados:**
```bash
docker exec -it usdcop-postgres psql -U usdcop -d usdcop_db \
  -c "SELECT symbol, COUNT(*), MIN(time), MAX(time) FROM macro_ohlcv GROUP BY symbol;"
```

**Output esperado:**
```
 symbol | count | min                 | max
--------+-------+---------------------+---------------------
 WTI    |    24 | 2025-11-05 00:00:00 | 2025-11-05 23:00:00
 DXY    |    24 | 2025-11-05 00:00:00 | 2025-11-05 23:00:00
```

✅ **Criterio de éxito:** ~24 registros por símbolo (24 horas × 1 registro/hora)

**Ejecutar catchup histórico (2002-2025):**

⚠️ **IMPORTANTE:** Esto descargará ~23 años de datos. Tiempo estimado: 2-3 horas.

```bash
# Activar catchup en el DAG (ya está en True por defecto)
# Airflow ejecutará automáticamente todos los días desde start_date (2002-01-01) hasta hoy

# Monitorear progreso
# Ir a Airflow UI → DAG usdcop_m5__01b_l0_macro_acquire → Calendar View
# Debería mostrar ejecuciones para cada día desde 2002
```

**Verificar catchup completo:**
```bash
docker exec -it usdcop-postgres psql -U usdcop -d usdcop_db \
  -c "SELECT * FROM get_macro_stats();"
```

**Output esperado:**
```
 symbol | record_count | min_time            | max_time            | days_coverage | source
--------+--------------+---------------------+---------------------+---------------+-----------
 WTI    |        45000 | 2002-01-02 00:00:00 | 2025-11-05 23:00:00 |          8674 | twelvedata
 DXY    |        45000 | 2002-01-02 00:00:00 | 2025-11-05 23:00:00 |          8674 | twelvedata
```

✅ **Criterio de éxito:** ~45,000 registros por símbolo, ~8,600 días de cobertura

---

### **Paso 3B: Fallback Manual (SI TWELVEDATA NO DISPONIBLE)**

#### **3B.1: Descargar datos de investing.com**

**WTI Crude Oil:**
1. Ir a: https://www.investing.com/commodities/crude-oil-historical-data
2. Date Range: Seleccionar desde **Jan 02, 2002** hasta **hoy**
3. Clic en "Download" (descarga CSV)
4. Guardar como: `WTI_Historical_Data.csv`

**US Dollar Index:**
1. Ir a: https://www.investing.com/indices/usdollar-historical-data
2. Date Range: Seleccionar desde **Jan 02, 2002** hasta **hoy**
3. Clic en "Download" (descarga CSV)
4. Guardar como: `DXY_Historical_Data.csv`

#### **3B.2: Cargar datos con script**

```bash
# WTI
python scripts/upload_macro_manual.py \
  --file ~/Downloads/WTI_Historical_Data.csv \
  --symbol WTI

# DXY
python scripts/upload_macro_manual.py \
  --file ~/Downloads/DXY_Historical_Data.csv \
  --symbol DXY
```

**Output esperado (WTI):**
```
============================================================
                UPLOAD MACRO DATA - WTI
============================================================

ℹ️  Leyendo CSV: /home/user/Downloads/WTI_Historical_Data.csv
ℹ️  Registros encontrados: 5843
✅ CSV parseado correctamente:
  - Registros: 5843
  - Rango: 2002-01-02 → 2025-11-05
  - Último precio: 75.23

============================================================
                SUBIENDO A POSTGRESQL
============================================================

ℹ️  Conectando a PostgreSQL...
✅ Conectado a PostgreSQL
ℹ️  Insertando 5843 registros...
  Progreso: 5843/5843 registros...
✅ Insertados/actualizados 5843 registros

============================================================
                SUBIENDO A MINIO
============================================================

ℹ️  Conectando a MinIO...
✅ Conectado a MinIO
ℹ️  Subiendo a MinIO: WTI/manual/macro_WTI_manual_20020102_20251105.parquet
✅ Archivo subido: WTI/manual/macro_WTI_manual_20020102_20251105.parquet (45.23 KB)

============================================================
                    VERIFICACIÓN
============================================================

ℹ️  Verificando datos en PostgreSQL...
✅ Datos verificados:
  - Registros totales: 5843
  - Rango temporal: 2002-01-02 00:00:00 → 2025-11-05 00:00:00
  - Precio promedio: 68.45
  - Precio mín/máx: 19.20 / 145.31

============================================================
                ✅ PROCESO COMPLETADO
============================================================

✅ WTI: 5843 registros cargados
```

⚠️ **NOTA:** El fallback manual descarga datos **diarios** (no horarios). En L3 se replicarán a 5min con forward-fill.

✅ **Criterio de éxito:** ~6,000 registros por símbolo (datos diarios 2002-2025)

---

### **Paso 4: Validar Datos en PostgreSQL** (2 min)

```bash
# Estadísticas generales
docker exec -it usdcop-postgres psql -U usdcop -d usdcop_db \
  -c "SELECT * FROM get_macro_stats();"

# Detectar gaps en WTI
docker exec -it usdcop-postgres psql -U usdcop -d usdcop_db \
  -c "SELECT * FROM detect_macro_gaps('WTI', '1 hour') LIMIT 10;"

# Detectar gaps en DXY
docker exec -it usdcop-postgres psql -U usdcop -d usdcop_db \
  -c "SELECT * FROM detect_macro_gaps('DXY', '1 hour') LIMIT 10;"

# Últimos 10 registros por símbolo
docker exec -it usdcop-postgres psql -U usdcop -d usdcop_db \
  -c "SELECT * FROM macro_ohlcv WHERE symbol = 'WTI' ORDER BY time DESC LIMIT 10;"
```

**Verificar MinIO:**
```bash
# Listar archivos en bucket
mc ls minio/00-raw-macro-marketdata/WTI/
mc ls minio/00-raw-macro-marketdata/DXY/
```

✅ **Criterio de éxito:**
- PostgreSQL: ~45,000 registros (TwelveData) o ~6,000 (manual) por símbolo
- MinIO: Archivos parquet presentes para cada símbolo
- 0% NaN en columnas OHLC
- Gaps detectados: Razonables (fines de semana, holidays)

---

## 🔄 MANTENIMIENTO DIARIO

### **Con TwelveData (Automático)**

El DAG `usdcop_m5__01b_l0_macro_acquire` se ejecuta **diariamente** a las 00:00 UTC.

**No requiere acción manual.**

**Monitoreo:**
```bash
# Ver última ejecución
docker exec -it usdcop-postgres psql -U usdcop -d usdcop_db \
  -c "SELECT symbol, MAX(time) as last_update FROM macro_ohlcv GROUP BY symbol;"
```

### **Con Fallback Manual (Diario)**

**Pasos diarios:**
1. Descargar últimos datos de investing.com (WTI y DXY)
2. Ejecutar script:
   ```bash
   python scripts/upload_macro_manual.py --file wti_today.csv --symbol WTI
   python scripts/upload_macro_manual.py --file dxy_today.csv --symbol DXY
   ```

**Automatizar con cron (opcional):**
```bash
# Editar crontab
crontab -e

# Añadir (ejecutar a las 8 AM diario)
0 8 * * * cd /path/to/project && python scripts/upload_macro_manual.py --file /path/to/wti.csv --symbol WTI
0 8 * * * cd /path/to/project && python scripts/upload_macro_manual.py --file /path/to/dxy.csv --symbol DXY
```

---

## ❌ TROUBLESHOOTING

### **Error: "Tabla macro_ohlcv no existe"**

**Solución:**
```bash
docker exec -it usdcop-postgres psql -U usdcop -d usdcop_db \
  -f /init-scripts/02-macro-data-schema.sql
```

---

### **Error: "TwelveData rate limit exceeded"**

**Solución:**
1. Verificar cuotas de API en TwelveData dashboard
2. Rotar a otra API key:
   ```bash
   export TWELVEDATA_API_KEY_G1="otra_api_key"
   ```
3. Si persiste, usar fallback manual

---

### **Error: "MinIO bucket not accessible"**

**Solución:**
```bash
# Verificar MinIO está corriendo
docker ps | grep minio

# Crear bucket manualmente
mc mb minio/00-raw-macro-marketdata
```

---

### **Gap detection muestra muchos gaps**

**Esperado:** Fines de semana y holidays tendrán gaps naturales

**Verificar:**
```bash
# Contar gaps
docker exec -it usdcop-postgres psql -U usdcop -d usdcop_db \
  -c "SELECT COUNT(*) FROM detect_macro_gaps('WTI', '1 hour');"
```

Si gaps > 500: Revisar descarga de datos históricos

---

### **CSV de investing.com no se parsea**

**Causa común:** Formato de fecha diferente

**Solución:**
1. Abrir CSV en Excel/LibreOffice
2. Verificar formato de columna "Date"
3. Debe ser: "Nov 05, 2025" o "2025-11-05"
4. Guardar y reintentar

---

## 📊 MÉTRICAS DE ÉXITO FASE 0

| Métrica | Target | Status |
|---------|--------|--------|
| WTI registros | > 40,000 (TwelveData) o > 5,000 (manual) | ✅/❌ |
| DXY registros | > 40,000 (TwelveData) o > 5,000 (manual) | ✅/❌ |
| Calidad OHLC | 0% NaN | ✅/❌ |
| Cobertura temporal | 2002-2025 | ✅/❌ |
| Latencia DAG | < 5 min/día | ✅/❌ |
| MinIO archivos | Presentes | ✅/❌ |

---

## ➡️ PRÓXIMOS PASOS

Una vez completada Fase 0 (todos ✅):

1. **Fase 2:** Actualizar L3 para calcular macro features
   - Leer archivo: `PLAN_ESTRATEGICO_v2_UPDATES.md` Sección 2.1
   - Modificar: `airflow/dags/usdcop_m5__04_l3_feature.py`

2. **Fase 2:** Actualizar L4 para expandir obs_XX
   - Leer archivo: `PLAN_ESTRATEGICO_v2_UPDATES.md` Sección 2.3
   - Modificar: `airflow/dags/usdcop_m5__05_l4_rlready.py`

---

## 📞 SOPORTE

**Logs del DAG:**
```bash
docker logs usdcop-airflow-webserver | grep macro
```

**PostgreSQL debug:**
```bash
docker exec -it usdcop-postgres psql -U usdcop -d usdcop_db
```

**MinIO debug:**
```bash
mc ls --recursive minio/00-raw-macro-marketdata/
```

---

**FIN DE INSTRUCCIONES FASE 0**

*Versión 1.0 - 2025-11-05*
