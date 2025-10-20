# Arquitectura de Fuentes de Datos - USD/COP Trading System

## 📊 RESUMEN: De Dónde Vienen TODOS los Datos

Este documento explica **EXACTAMENTE** cómo fluyen los datos desde las fuentes originales hasta el frontend del dashboard.

---

## 🎯 LAS 3 FUENTES DE DATOS PRINCIPALES

### 1. **TwelveData API** (Fuente Original)
- **¿Qué es?**: API externa de datos financieros forex
- **Símbolo**: USD/COP (Dólar vs Peso Colombiano)
- **Frecuencia**: Barras de 5 minutos (OHLC)
- **Datos**: Open, High, Low, Close, Volume
- **Horario**: 8:00 AM - 12:55 PM COT (Horario colombiano)
- **Endpoint**: https://api.twelvedata.com/
- **Uso**: **FUENTE PRIMARIA** - Todos los datos históricos y en tiempo real provienen de aquí

### 2. **PostgreSQL/TimescaleDB** (Base de Datos)
- **¿Qué es?**: Base de datos relacional optimizada para series de tiempo
- **Tabla**: `market_data`
- **Registros actuales**: **92,936 barras** (2020-01-02 a 2025-10-10)
- **Estructura**:
  ```sql
  CREATE TABLE market_data (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL DEFAULT 'USDCOP',
    price NUMERIC(12,4) NOT NULL,
    bid NUMERIC(12,4),
    ask NUMERIC(12,4),
    volume BIGINT,
    source TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (timestamp, symbol)
  );
  ```
- **Uso**: **ALMACENAMIENTO PERSISTENTE** de datos descargados de TwelveData

### 3. **MinIO** (Object Storage)
- **¿Qué es?**: Sistema de almacenamiento de objetos (como AWS S3)
- **Buckets**: 7 buckets para cada capa del pipeline (L0-L6)
- **Formato**: Archivos JSON y Parquet
- **Uso**: **ALMACENAMIENTO DE RESULTADOS** de procesamiento del pipeline

---

## 🔄 FLUJO COMPLETO DE DATOS (Paso a Paso)

### **PASO 1: Adquisición Inicial (L0)**

#### DAG de Airflow: `usdcop_m5__01_l0_intelligent_acquire.py`

```python
# Este DAG corre cada 5 minutos durante horario de mercado
# Ubicación: /home/GlobalForex/USDCOP-RL-Models/airflow/dags/

@dag(
    schedule_interval="*/5 8-12 * * 1-5",  # Cada 5 min, 8am-12pm, Lunes-Viernes
    start_date=datetime(2020, 1, 1),
)
def usdcop_m5_l0_intelligent_acquire():
    # 1. Descarga datos de TwelveData API
    fetch_task = fetch_twelve_data_api()

    # 2. Guarda en PostgreSQL
    save_to_postgres = save_market_data()

    # 3. Guarda copia en MinIO
    save_to_minio = save_raw_data_minio()
```

**¿Qué hace exactamente?**

1. **Llama a TwelveData API**:
   ```python
   # Archivo: airflow/dags/tasks/twelve_data_client.py
   import requests

   def fetch_usdcop_data(interval='5min', outputsize=5000):
       url = "https://api.twelvedata.com/time_series"
       params = {
           'symbol': 'USD/COP',
           'interval': '5min',
           'outputsize': 5000,
           'apikey': os.getenv('TWELVEDATA_API_KEY_1')
       }
       response = requests.get(url, params=params)
       return response.json()['values']  # Lista de barras OHLC
   ```

2. **Inserta en PostgreSQL**:
   ```sql
   -- Se ejecuta desde Python
   INSERT INTO market_data (timestamp, symbol, price, bid, ask, volume, source)
   VALUES
     ('2024-01-15 13:00:00+00', 'USDCOP', 4012.50, 4011.00, 4014.00, 12500, 'twelvedata'),
     ('2024-01-15 13:05:00+00', 'USDCOP', 4013.25, 4012.00, 4014.50, 13200, 'twelvedata'),
     -- ... más barras
   ON CONFLICT (timestamp, symbol) DO NOTHING;  -- Evita duplicados
   ```

3. **Guarda en MinIO** (bucket: `00-raw-usdcop-marketdata`):
   ```python
   # Estructura en MinIO:
   # 00-raw-usdcop-marketdata/
   #   data/
   #     20240115/  # Fecha en formato YYYYMMDD
   #       usdcop_5m_20240115_1300.json
   #       usdcop_5m_20240115_1305.json
   ```

**Resultado L0**: Ahora tenemos 92,936 barras en PostgreSQL y copias en MinIO

---

### **PASO 2: Estandarización (L1)**

#### DAG: `usdcop_m5__02_l1_standardize.py`

**¿De dónde lee?**: Lee de PostgreSQL (tabla `market_data`)

**¿Qué hace?**:
1. **Crea episodios de 60 barras** (5 horas de trading)
2. **Aplica quality gates**:
   - Sin valores faltantes
   - Variación de precio suficiente
   - Dentro de horario de mercado
   - Volumen válido

**Código simplificado**:
```python
def create_episodes_from_postgres():
    # Lee de PostgreSQL
    query = """
        SELECT timestamp, price, volume
        FROM market_data
        WHERE symbol = 'USDCOP'
        ORDER BY timestamp
    """
    df = pd.read_sql(query, connection)

    # Crea episodios de 60 barras
    episodes = []
    for i in range(0, len(df), 60):
        episode = df.iloc[i:i+60]

        # Quality gates
        if len(episode) == 60 and no_missing_values(episode):
            episodes.append(episode)

    # Guarda en MinIO como Parquet
    save_to_minio(episodes, bucket='01-l1-ds-usdcop-standardize')

    return f"Accepted {len(episodes)} episodes out of {len(df)//60}"
```

**Resultado L1**:
- **929 episodios** aceptados (de ~1,549 episodios totales)
- Guardados en MinIO: `01-l1-ds-usdcop-standardize/data/`
- Formato: Parquet (más eficiente que JSON)

---

### **PASO 3-6: Procesamiento L2, L3, L4, L5**

Cada capa **lee de la capa anterior en MinIO**:

```
L2 lee L1 → Deseasonaliza → Guarda en 02-l2-ds-usdcop-prep
L3 lee L2 → Crea 17 features → Guarda en 03-l3-ds-usdcop-features
L4 lee L3 → Crea splits train/val/test → Guarda en 04-l4-ds-usdcop-rlready
L5 lee L4 → Entrena modelo RL → Guarda ONNX en 05-l5-ds-usdcop-serving
L6 lee L5 → Backtesting → Guarda métricas en usdcop-l6-backtest
```

**NO SE VUELVE A LLAMAR A TWELVEDATA** después de L0

---

## 🖥️ CÓMO LLEGAN LOS DATOS AL FRONTEND

### **Ruta 1: Dashboard Home (Vista Principal)**

```typescript
// Archivo: usdcop-trading-dashboard/components/views/UnifiedTradingTerminal.tsx

useEffect(() => {
  // 1. Llama al endpoint de la API Next.js
  fetch('/api/pipeline/l0/raw-data?limit=100&source=postgres')
    .then(res => res.json())
    .then(data => {
      // 2. Los datos vienen de PostgreSQL
      setMarketData(data.data);  // Array de 100 barras OHLC
    });
}, []);
```

**Flujo exacto**:
```
Frontend (React)
  ↓ HTTP GET
Next.js API Route (/api/pipeline/l0/raw-data/route.ts)
  ↓ SQL Query
PostgreSQL (market_data table)
  ↓ Retorna JSON
Frontend renderiza gráfica
```

---

### **Ruta 2: Backtest Results**

```typescript
// Archivo: components/views/L6BacktestResults.tsx

useEffect(() => {
  // 1. Llama al endpoint L6
  fetch('/api/pipeline/l6/backtest-results?split=test')
    .then(res => res.json())
    .then(data => {
      // 2. Los datos vienen de MinIO bucket
      setBacktestMetrics(data.results.test.kpis);
    });
}, []);
```

**Flujo exacto**:
```
Frontend (React)
  ↓ HTTP GET
Next.js API Route (/api/pipeline/l6/backtest-results/route.ts)
  ↓ MinIO Client
MinIO (bucket: usdcop-l6-backtest)
  ↓ Lee JSON/Parquet
Frontend muestra Sharpe Ratio, Max Drawdown, etc.
```

---

## 📍 UBICACIONES FÍSICAS DE LOS DATOS

### En el Servidor

```bash
# PostgreSQL Data
/var/lib/postgresql/data/
  └── market_data table (92,936 rows)

# MinIO Data
/var/lib/minio/data/
  ├── 00-raw-usdcop-marketdata/       # ~5 GB
  ├── 01-l1-ds-usdcop-standardize/    # ~2 GB (929 episodios)
  ├── 02-l2-ds-usdcop-prep/           # ~2 GB
  ├── 03-l3-ds-usdcop-features/       # ~3 GB (17 features)
  ├── 04-l4-ds-usdcop-rlready/        # ~4 GB (splits)
  ├── 05-l5-ds-usdcop-serving/        # ~500 MB (modelos ONNX)
  └── usdcop-l6-backtest/             # ~100 MB (métricas)
```

---

## 🔍 VERIFICACIÓN: ¿De Dónde Vienen los Datos AHORA?

### Comprobación 1: PostgreSQL

```bash
docker compose exec -T postgres psql -U admin -d usdcop_trading -c "
  SELECT
    COUNT(*) as total_bars,
    MIN(timestamp) as earliest,
    MAX(timestamp) as latest,
    COUNT(DISTINCT DATE(timestamp)) as trading_days
  FROM market_data;
"
```

**Resultado actual**:
```
total_bars | earliest            | latest              | trading_days
-----------+---------------------+---------------------+--------------
92936      | 2020-01-02 07:30:00 | 2025-10-10 18:55:00 | 1450
```

### Comprobación 2: MinIO Buckets

```bash
docker compose exec -T minio mc ls minio/
```

**Resultado esperado**:
```
[2024-10-15 10:30:00] 5.2GiB 00-raw-usdcop-marketdata
[2024-10-15 11:45:00] 2.1GiB 01-l1-ds-usdcop-standardize
[2024-10-15 12:30:00] 2.3GiB 02-l2-ds-usdcop-prep
...
```

---

## 🎨 MAPEO: Vista del Frontend → Fuente de Datos

| Vista del Dashboard | Endpoint API | Fuente Real | Tabla/Bucket |
|---------------------|--------------|-------------|--------------|
| **Home Chart** | `/api/pipeline/l0/raw-data` | PostgreSQL | `market_data` |
| **L0 Statistics** | `/api/pipeline/l0/statistics` | PostgreSQL | `market_data` (agregaciones) |
| **L1 Episodes** | `/api/pipeline/l1/episodes` | MinIO | `01-l1-ds-usdcop-standardize/` |
| **L1 Quality Report** | `/api/pipeline/l1/quality-report` | MinIO | `01-l1.../reports/quality_*.json` |
| **L2 Prepared Data** | `/api/pipeline/l2/prepared-data` | MinIO | `02-l2-ds-usdcop-prep/data/` |
| **L3 Features** | `/api/pipeline/l3/features` | MinIO | `03-l3-ds-usdcop-features/data/` |
| **L4 Dataset** | `/api/pipeline/l4/dataset` | MinIO | `04-l4-ds-usdcop-rlready/` |
| **L5 Models** | `/api/pipeline/l5/models` | MinIO | `05-l5-ds-usdcop-serving/*.onnx` |
| **L6 Backtest** | `/api/pipeline/l6/backtest-results` | MinIO | `usdcop-l6-backtest/metrics/` |
| **Real-time Updates** | `/api/market/realtime` | TwelveData | API call directo |

---

## 🔄 ACTUALIZACIÓN DE DATOS: ¿Cuándo se Ejecutan los DAGs?

### Scheduler de Airflow

```python
# L0: Cada 5 minutos durante mercado
"*/5 8-12 * * 1-5"  # 8:00-12:55, Lunes-Viernes

# L1-L5: Diariamente después del cierre
"0 14 * * 1-5"      # 2:00 PM COT (después de 12:55 PM cierre)

# L6: Semanalmente
"0 3 * * 6"         # Sábados a las 3 AM
```

---

## ⚠️ IMPORTANTE: NO Hay Datos Hardcodeados

**ANTES** (problema que tenías):
```typescript
// ❌ MAL - Datos hardcodeados
const marketData = [
  { time: '2024-01-01', price: 4000 },
  { time: '2024-01-02', price: 4010 },
  // ... hardcoded
];
```

**AHORA** (solución implementada):
```typescript
// ✅ BIEN - Datos reales de PostgreSQL/MinIO
const [marketData, setMarketData] = useState([]);

useEffect(() => {
  fetch('/api/pipeline/l0/raw-data')
    .then(res => res.json())
    .then(data => setMarketData(data.data));  // Datos REALES
}, []);
```

---

## 🧪 PRUEBA TÚ MISMO

### Test 1: Verificar datos en PostgreSQL
```bash
curl -s http://localhost:5000/api/pipeline/l0/statistics | jq '.statistics.overview'
```

**Output esperado**:
```json
{
  "totalRecords": 92936,
  "dateRange": {
    "earliest": "2020-01-02T07:30:00Z",
    "latest": "2025-10-10T18:55:00Z",
    "tradingDays": 1450
  },
  "priceMetrics": {
    "min": 3800.5,
    "max": 4250.75,
    "avg": 4012.35
  }
}
```

### Test 2: Verificar MinIO L6
```bash
curl -s http://localhost:5000/api/pipeline/l6/backtest-results?split=test | jq '.results.test.kpis'
```

---

## 📊 DIAGRAMA DE FLUJO COMPLETO

```
                    ┌─────────────────┐
                    │  TwelveData API │ ← FUENTE ORIGINAL
                    │   (USD/COP)     │
                    └────────┬────────┘
                             │
                             ↓ fetch_twelve_data()
                    ┌─────────────────┐
                    │  L0 DAG (Airflow)│
                    └────────┬────────┘
                             │
                ┌────────────┴────────────┐
                ↓                         ↓
        ┌──────────────┐         ┌──────────────┐
        │  PostgreSQL  │         │    MinIO     │
        │ (market_data)│         │  (L0 bucket) │
        │  92,936 rows │         └──────────────┘
        └──────┬───────┘
               │ read_sql()
               ↓
        ┌──────────────┐
        │  L1 DAG      │
        │ (929 episodes)│
        └──────┬───────┘
               │
               ↓ save_parquet()
        ┌──────────────┐
        │    MinIO     │
        │  (L1 bucket) │
        └──────┬───────┘
               │
               ↓ L2, L3, L4, L5, L6...
        ┌──────────────┐
        │    MinIO     │
        │ (L2-L6 buckets)│
        └──────┬───────┘
               │
               ↓ API calls
        ┌──────────────┐
        │  Next.js API │
        │  (/api/...)  │
        └──────┬───────┘
               │
               ↓ fetch()
        ┌──────────────┐
        │   Frontend   │
        │   (React)    │
        └──────────────┘
```

---

## ✅ RESUMEN FINAL

### Datos del Frontend vienen de:

1. **Gráficas de precio (L0)**: PostgreSQL `market_data` (92,936 barras de TwelveData)
2. **Estadísticas L0**: Agregaciones SQL sobre PostgreSQL
3. **Episodios L1-L4**: Archivos Parquet en MinIO (procesados por Airflow)
4. **Modelos L5**: Archivos ONNX en MinIO (entrenados con PPO)
5. **Backtests L6**: JSON/Parquet en MinIO (métricas calculadas)
6. **Datos en tiempo real**: TwelveData API (llamadas directas cada 5 min)

### NO hay datos inventados o hardcodeados. TODO viene de:
- **TwelveData** (fuente original)
- **PostgreSQL** (almacenamiento persistente)
- **MinIO** (resultados de procesamiento)

¿Te queda claro ahora el flujo completo? 🚀
