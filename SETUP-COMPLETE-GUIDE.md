# USDCOP Trading System - Guía Completa de Configuración y Replicación

## 🎯 Resumen Ejecutivo

Este documento proporciona una guía **completa, determinística y replicable** para configurar el Sistema de Trading USDCOP con Reinforcement Learning. El sistema incluye pipelines de datos automatizados (L0-L6), dashboard en tiempo real, monitoreo completo y configuración MLOps empresarial.

## 🏗️ Arquitectura del Sistema

### Servicios Principales
- **📊 Dashboard Trading**: Next.js 15.5.2 + TypeScript (Puerto 3000)
- **🔄 Apache Airflow**: Orquestación de pipelines L0-L6 (Puerto 8080)
- **💾 MinIO**: Almacenamiento S3-compatible para datos (Puerto 9000/9001)
- **🗃️ PostgreSQL**: Base de datos principal (Puerto 5432)
- **⚡ Redis**: Cache y mensaje broker (Puerto 6379)
- **📈 Prometheus**: Métricas y monitoreo (Puerto 9090)
- **📊 Grafana**: Dashboards de observabilidad (Puerto 3001)
- **🔄 Nginx**: Reverse proxy y balanceador (Puerto 80/443)

### Pipeline de Datos (L0-L6)
```
L0: Adquisición → L1: Estandarización → L2: Preparación → 
L3: Features → L4: ML-Ready → L5: Serving → L6: Backtesting
```

## 🚀 Instalación Rápida (3 Comandos)

```bash
# 1. Clonar y navegar al directorio
git clone [repository-url] && cd USDCOP-RL-Models

# 2. Configurar ambiente
cp .env.example .env
# Editar .env con tus credenciales

# 3. Deploy completo
chmod +x deploy.sh && ./deploy.sh start
```

## 📋 Prerrequisitos del Sistema

### Software Requerido
```bash
# Docker & Docker Compose
docker --version    # >= 20.10.0
docker-compose --version  # >= 1.29.0

# Sistema Operativo
- Linux (Ubuntu 20.04+ recomendado)
- macOS 10.15+
- Windows 10/11 con WSL2
```

### Recursos de Hardware
```
CPU: 4+ cores
RAM: 8GB mínimo, 16GB recomendado
Disco: 50GB espacio libre
Red: Conexión estable a internet
```

## ⚙️ Configuración Detallada

### 1. Archivo de Configuración (.env)

```bash
# Copiar archivo de ejemplo
cp .env.example .env

# Variables críticas a configurar:
POSTGRES_PASSWORD=tu_password_seguro
REDIS_PASSWORD=tu_redis_password
MINIO_SECRET_KEY=tu_minio_password
AIRFLOW_PASSWORD=tu_airflow_password
GRAFANA_PASSWORD=tu_grafana_password

# Credenciales de Trading (OBLIGATORIO)
TWELVEDATA_API_KEY=tu_api_key_aqui
MT5_LOGIN=tu_mt5_login
MT5_PASSWORD=tu_mt5_password
MT5_SERVER=tu_mt5_server
```

### 2. Estructura de Buckets MinIO (Automatizada)

Los siguientes buckets se crean **automáticamente** al iniciar el sistema:

```
📁 Pipeline Principal (L0-L5)
├── 00-raw-usdcop-marketdata          # L0: Datos raw MT5/TwelveData
├── 01-l1-ds-usdcop-standardize       # L1: Normalización UTC y sesiones
├── 02-l2-ds-usdcop-prepare           # L2: Limpieza y filtros premium
├── 03-l3-ds-usdcop-feature           # L3: Indicadores técnicos (50+ features)
├── 04-l4-ds-usdcop-rlready           # L4: Datasets ML train/val/test
└── 05-l5-ds-usdcop-serving           # L5: Predicciones y serving

📁 Pipeline L6 (Backtesting)
├── usdcop-l4-rlready                 # Input para backtesting
├── usdcop-l5-serving                 # Modelos para backtest
└── usdcop-l6-backtest                # Resultados backtesting

📁 Buckets Comunes
├── 99-common-trading-models          # Modelos entrenados
├── 99-common-trading-reports         # Reportes y analytics
└── 99-common-trading-backups         # Respaldos del sistema
```

### 3. Scripts de Deployment

#### Script Principal (deploy.sh)
```bash
# Permisos de ejecución
chmod +x deploy.sh

# Comandos disponibles
./deploy.sh start      # Deploy completo
./deploy.sh stop       # Detener servicios
./deploy.sh restart    # Reiniciar sistema
./deploy.sh status     # Estado de servicios
./deploy.sh logs       # Ver logs
./deploy.sh backup     # Crear respaldo
./deploy.sh health     # Verificar salud
```

## 🔧 Configuración Avanzada MLOps

### Automatización de Buckets
El sistema incluye **configuración YAML** para automatización MLOps:

```yaml
# airflow/configs/pipeline_dataflow.yml
buckets:
  l0_acquire: 00-raw-usdcop-marketdata
  l1_standardize: 01-l1-ds-usdcop-standardize
  # ... configuración completa automática
```

### Orquestación de Pipelines
```yaml
# Triggers automáticos entre capas
L0_to_L1:
  trigger: file_created
  pattern: "*/READY_*"
  timeout: 3600

L1_to_L2:
  trigger: control_signal
  pattern: "*/_control/*/READY"
  timeout: 1800
```

### Quality Gates
```yaml
# Validación automática de calidad
L1_STANDARDIZE:
  completeness_min: 0.98
  max_gap_bars: 1
  required_bars_per_episode: 60

L2_PREPARE:
  outliers_max_pct: 0.01
  missing_features_max: 0
```

## 🌐 Acceso a Servicios

Una vez desplegado, accede a los servicios:

| Servicio | URL | Credenciales |
|----------|-----|-------------|
| 🎯 **Dashboard Trading** | http://localhost:3000 | N/A |
| 📊 **Grafana** | http://localhost:3001 | admin/admin123 |
| 🔄 **Airflow** | http://localhost:8080 | admin/admin123 |
| 💾 **MinIO Console** | http://localhost:9001 | minioadmin/minioadmin123 |
| 📈 **Prometheus** | http://localhost:9090 | N/A |

## 🔍 Verificación del Sistema

### 1. Verificación de Servicios
```bash
# Estado general
docker-compose ps

# Logs específicos
docker-compose logs -f airflow-webserver
docker-compose logs -f dashboard
docker-compose logs -f minio
```

### 2. Verificación de Buckets
```bash
# Listar buckets creados
docker exec usdcop-minio-init mc ls minio

# Verificar conectividad MinIO
curl http://localhost:9000/minio/health/live
```

### 3. Verificación de DAGs
1. Navegar a http://localhost:8080
2. Login: admin/admin123
3. Verificar 6 DAGs aparecen sin errores:
   - `usdcop_m5__01_l0_acquire`
   - `usdcop_m5__02_l1_standardize`
   - `usdcop_m5__03_l2_prepare`
   - `usdcop_m5__04_l3_features`
   - `usdcop_m5__05_l4_rlready`
   - `usdcop_m5__06_l5_serving`

### 4. Verificación de Dashboard
1. Navegar a http://localhost:3000
2. Verificar carga sin errores
3. Confirmar conexión a base de datos

## 🔐 Configuración de Seguridad

### Passwords Seguros
```bash
# Generar passwords seguros
openssl rand -base64 32  # Para passwords generales
openssl rand -hex 16     # Para keys/tokens

# Fernet Key para Airflow
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

### Configuración de Producción
```bash
# Variables adicionales para producción
ENVIRONMENT=production
ENABLE_SSL=true
API_RATE_LIMIT=1000
ENABLE_MONITORING=true
```

## 🔄 Operaciones de Mantenimiento

### Respaldos Automáticos
```bash
# Crear respaldo completo
./deploy.sh backup

# Respaldos se guardan en:
./backups/backup_YYYYMMDD_HHMMSS/
├── postgres_data.tar.gz
├── minio_data.tar.gz
├── airflow/
└── .env
```

### Actualización del Sistema
```bash
# Parar servicios
./deploy.sh stop

# Actualizar código
git pull origin main

# Rebuild y restart
./deploy.sh restart
```

### Limpieza de Sistema
```bash
# Limpiar containers e imágenes
./deploy.sh clean

# Limpiar volúmenes (¡CUIDADO! Borra datos)
docker-compose down -v
```

## 🚨 Troubleshooting

### Problemas Comunes

#### 1. Servicios no inician
```bash
# Verificar puertos en uso
netstat -tlnp | grep :8080

# Liberar puertos si necesario
sudo kill -9 $(lsof -ti:8080)
```

#### 2. Airflow DAGs con errores
```bash
# Verificar logs de Airflow
docker-compose logs airflow-webserver
docker-compose logs airflow-scheduler

# Verificar importación de DAGs
docker exec usdcop-airflow-webserver airflow dags list
```

#### 3. Dashboard no carga
```bash
# Verificar build del dashboard
docker-compose logs dashboard

# Verificar conectividad a DB
docker exec usdcop-postgres pg_isready -U admin
```

#### 4. MinIO buckets no se crean
```bash
# Verificar init container
docker-compose logs minio-init

# Crear buckets manualmente
docker exec usdcop-minio-init mc mb minio/00-raw-usdcop-marketdata
```

### Logs Importantes
```bash
# Logs principales por servicio
docker-compose logs -f postgres      # Base de datos
docker-compose logs -f redis         # Cache/broker
docker-compose logs -f minio         # Almacenamiento
docker-compose logs -f airflow-scheduler  # Orquestación
docker-compose logs -f dashboard     # Frontend
```

## 📊 Monitoreo y Observabilidad

### Métricas Clave
- **Airflow**: Task success rate, DAG run duration
- **MinIO**: Storage usage, request latency  
- **PostgreSQL**: Connection pool, query performance
- **Dashboard**: Response time, user sessions
- **Sistema**: CPU, memoria, disco

### Alertas Configuradas
- Pipeline failures → Slack/email
- Data quality degradation → Dashboard alerts
- Resource exhaustion → System notifications
- Service health checks → Automated restart

## 🎯 Casos de Uso

### Uso en Desarrollo
```bash
# Deploy local con datos de prueba
ENVIRONMENT=development ./deploy.sh start

# Activar modo debug
DEBUG_MODE=true docker-compose up -d dashboard
```

### Uso en Producción
```bash
# Deploy producción con SSL
ENVIRONMENT=production ENABLE_SSL=true ./deploy.sh start

# Configurar backups automáticos
crontab -e
# Agregar: 0 2 * * * /path/to/deploy.sh backup
```

### Uso para QA/Testing
```bash
# Deploy temporal para testing
./deploy.sh start
# ... ejecutar pruebas ...
./deploy.sh stop && docker-compose down -v
```

## 📚 Documentación Adicional

### Archivos de Referencia
- `PROJECT-SUMMARY.md` - Resumen técnico del proyecto
- `AIRFLOW-FIX-GUIDE.md` - Guía específica de Airflow
- `backend-api-specification.md` - Especificación de APIs
- `frontend-backend-integration.md` - Integración frontend

### Configuraciones YAML
- `airflow/configs/pipeline_dataflow.yml` - Configuración pipelines
- `docker-compose.yml` - Orquestación de servicios
- `.env.example` - Variables de ambiente

## ✅ Checklist de Verificación Final

### Pre-Deploy
- [ ] Docker y Docker Compose instalados
- [ ] Archivo .env configurado con credenciales reales
- [ ] Puertos 3000, 8080, 9000, 9001, 5432, 6379 disponibles
- [ ] Conexión a internet estable

### Post-Deploy
- [ ] Todos los servicios running (`docker-compose ps`)
- [ ] MinIO buckets creados (12 buckets total)
- [ ] Airflow DAGs cargados sin errores (6 DAGs)
- [ ] Dashboard accesible en http://localhost:3000
- [ ] Grafana dashboard configurado
- [ ] Health checks pasando

### Operacional
- [ ] Backups programados
- [ ] Monitoreo activo
- [ ] Alertas configuradas
- [ ] Documentación del equipo actualizada

## 🎉 ¡Sistema Listo!

Con esta configuración tienes un **sistema de trading completo, determinístico y replicable** que incluye:

✅ **Pipelines automatizados** L0-L6 con MLOps
✅ **Dashboard en tiempo real** con métricas de trading
✅ **Monitoreo completo** con alertas y observabilidad
✅ **Configuración empresarial** con seguridad y backups
✅ **Documentación completa** para replicación
✅ **Scripts automatizados** para operaciones

El sistema está **listo para producción** y puede ser replicado en cualquier servidor siguiendo exactamente estos pasos.

---

**Soporte**: Para problemas o preguntas, revisar la sección de Troubleshooting o consultar logs específicos con `./deploy.sh logs [servicio]`.