# 📊 USDCOP Historical Data Extraction

## 🎯 Descripción

Sistema completo para extraer datos históricos de USD/COP desde octubre 2022 hasta octubre 2023 con intervalos de 5 minutos usando la API de Twelve Data.

## 🚀 Quick Start

### 1. Instalar Dependencias
```bash
python scripts/install_dependencies.py
```

### 2. Probar Conexión API
```bash
python scripts/test_twelve_data_api.py
```

### 3. Extraer Datos Históricos
```bash
python scripts/extract_usdcop_historical.py
```

## 📁 Archivos del Sistema

### 🔧 Scripts Principales
- **`extract_usdcop_historical.py`**: Extractor principal de datos históricos
- **`test_twelve_data_api.py`**: Pruebas de conectividad y disponibilidad de datos
- **`install_dependencies.py`**: Instalador automático de dependencias

### ⚙️ Configuración
- **`config/twelve_data_config.yaml`**: Configuración completa del sistema

### 📊 Datos de Salida
- **`data/raw/twelve_data/`**: Datos extraídos en formato CSV
- **`logs/usdcop_extraction.log`**: Logs detallados del proceso

## 🌐 API Twelve Data

### 📋 Características
- **Plan Gratuito**: 8 requests/min, 800 requests/día
- **Símbolo**: USD/COP (par listado explícitamente)
- **Intervalos**: 1min, 5min, 15min, 30min, 1hour, 1day
- **Cobertura**: Datos intradía desde 1 minuto
- **Histórico**: Hasta ~1 año según instrumento/plan

### 🔑 API Key
```
085ba06282774cbc8e796f46a5af8ece
```

### 📡 Endpoints Utilizados
- **Quote**: `/quote` - Precio actual
- **Time Series**: `/time_series` - Datos históricos

## 📅 Período de Extracción

### 🗓️ Rango de Fechas
- **Inicio**: 1 de octubre de 2022, 00:00:00
- **Fin**: 31 de octubre de 2023, 23:59:59
- **Duración**: 13 meses (396 días)

### ⏱️ Intervalo de Datos
- **Timeframe**: 5 minutos
- **Registros por día**: 288 (24h × 12 intervalos de 5min)
- **Total estimado**: ~103,680 registros

## 🔄 Proceso de Extracción

### 📊 Estrategia de Paginación
```
Período Total (396 días)
    ↓
Dividido en chunks de 30 días
    ↓
13 períodos de extracción
    ↓
Respetando límite de 8 requests/min
    ↓
Delay de 7.5 segundos entre períodos
```

### ⚡ Optimizaciones
- **Chunking inteligente**: 30 días por request
- **Rate limiting**: Respeta límites de la API
- **Manejo de errores**: Continúa en caso de fallos parciales
- **Recuperación**: Guarda datos parciales

### 🧹 Procesamiento de Datos
1. **Validación**: Verificación de formato y rangos
2. **Limpieza**: Eliminación de duplicados y valores nulos
3. **Normalización**: Formato consistente de columnas
4. **Ordenamiento**: Por timestamp ascendente

## 📊 Estructura de Datos

### 📋 Columnas de Salida
```csv
time,open,high,low,close,volume
2022-10-01 00:00:00,4200.50,4205.75,4198.25,4202.00,1250000
2022-10-01 00:05:00,4202.00,4208.50,4200.00,4206.25,1180000
...
```

### 🔍 Validaciones
- **Precios**: Rango 3,000 - 5,000 COP
- **Timestamps**: Intervalos de 5 minutos
- **Volumen**: Valores no negativos
- **Consistencia OHLC**: High ≥ Low, Open/Close dentro del rango

## 🚨 Límites y Consideraciones

### ⚠️ Limitaciones de la API
- **Requests por minuto**: 8 (plan gratuito)
- **Requests por día**: 800 (plan gratuito)
- **Tiempo de extracción estimado**: ~13 minutos
- **Datos históricos**: Limitados según plan

### 🔧 Estrategias de Fallback
- **Intervalos alternativos**: 15min, 1hour, 1day
- **Reintentos automáticos**: 3 intentos por período
- **Guardado parcial**: Datos extraídos se guardan inmediatamente

## 📈 Monitoreo y Logs

### 📝 Sistema de Logging
- **Nivel**: INFO (configurable)
- **Archivo**: `logs/usdcop_extraction.log`
- **Consola**: Salida en tiempo real
- **Rotación**: Archivos de hasta 10MB

### 📊 Métricas de Progreso
- **Períodos procesados**: X/13
- **Registros extraídos**: X registros
- **Tiempo transcurrido**: X minutos
- **Tiempo restante estimado**: X minutos

## 🔍 Troubleshooting

### ❌ Problemas Comunes

#### 1. Error de Conexión
```bash
# Verificar conectividad a internet
ping api.twelvedata.com

# Verificar API key
python scripts/test_twelve_data_api.py
```

#### 2. Límites de API Excedidos
```bash
# Esperar hasta el siguiente día
# O actualizar a plan premium
```

#### 3. Datos Incompletos
```bash
# Verificar logs
tail -f logs/usdcop_extraction.log

# Reintentar extracción
python scripts/extract_usdcop_historical.py
```

### 🔧 Soluciones

#### Verificar Estado de la API
```bash
curl "https://api.twelvedata.com/quote?symbol=USD/COP&apikey=TU_API_KEY"
```

#### Limpiar Datos Temporales
```bash
rm -rf data/raw/twelve_data/*.tmp
rm -rf logs/usdcop_extraction.log
```

## 📚 Ejemplos de Uso

### 🔍 Verificar Datos Extraídos
```python
import pandas as pd

# Cargar datos
df = pd.read_csv('data/raw/twelve_data/USDCOP_5min_2022_10_2023_10.csv')

# Información básica
print(f"Total registros: {len(df):,}")
print(f"Rango de fechas: {df['time'].min()} - {df['time'].max()}")
print(f"Columnas: {list(df.columns)}")

# Estadísticas de precios
print(f"Precio mínimo: {df['low'].min():.2f} COP")
print(f"Precio máximo: {df['high'].max():.2f} COP")
print(f"Precio promedio: {df['close'].mean():.2f} COP")
```

### 📊 Análisis de Calidad
```python
# Verificar completitud
expected_records = 396 * 24 * 12  # días × horas × intervalos
actual_records = len(df)
completeness = (actual_records / expected_records) * 100

print(f"Completitud de datos: {completeness:.1f}%")

# Detectar gaps
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time')

# Calcular diferencias entre timestamps
time_diffs = df['time'].diff()
gaps = time_diffs[time_diffs > pd.Timedelta(minutes=10)]

print(f"Gaps detectados: {len(gaps)}")
```

## 🚀 Próximos Pasos

### 📊 Después de la Extracción
1. **Verificar calidad**: Revisar completitud y consistencia
2. **Integrar con pipeline**: Usar datos en `master_pipeline.py`
3. **Análisis exploratorio**: Explorar patrones y características
4. **Feature engineering**: Crear indicadores técnicos

### 🔄 Actualizaciones Futuras
- **Datos en tiempo real**: Streaming de datos actuales
- **Múltiples timeframes**: 1min, 15min, 1hour
- **Otros pares**: EUR/COP, GBP/COP
- **Datos fundamentales**: Noticias económicas, indicadores

## 📞 Soporte

### 🔗 Recursos
- **Documentación API**: [support.twelvedata.com](https://support.twelvedata.com)
- **Límites de plan**: [Twelve Data Plans](https://twelvedata.com/pricing)
- **Soporte técnico**: [support@twelvedata.com](mailto:support@twelvedata.com)

### 📝 Logs y Debugging
- **Logs del sistema**: `logs/usdcop_extraction.log`
- **Estado de extracción**: Monitorear en tiempo real
- **Errores específicos**: Revisar códigos de error HTTP

---

## ⚠️ Disclaimer

Este sistema es para propósitos educativos y de investigación. La API de Twelve Data tiene límites de uso que deben respetarse. Los datos extraídos son responsabilidad del usuario.

---

**Última Actualización**: Agosto 2025  
**Versión**: 1.0.0  
**Estado**: ✅ PRODUCCIÓN  
**API**: Twelve Data  
**Símbolo**: USD/COP  
**Período**: Oct 2022 - Oct 2023
