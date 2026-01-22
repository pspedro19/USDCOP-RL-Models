# USDCOP Trading System - Startup Checklist

## Lecciones Aprendidas y Verificaciones para Arranque Limpio

Este documento resume los problemas encontrados al levantar los servicios desde cero
y las soluciones aplicadas para que no se repitan.

---

## Estado de Correcciones

| Problema | Causa Raíz | Corrección | Estado |
|----------|------------|------------|--------|
| Tablas config.models incompletas | Columnas faltantes | ✅ Actualizado en `10-multi-model-schema.sql` | **RESUELTO** |
| DemoConfig sin monthly_bias | Atributo faltante | ✅ Agregado en `services/demo_mode/config.py` | **RESUELTO** |
| entrypoint.sh "no such file" | Line endings CRLF (Windows) | ✅ Creado `.gitattributes` para forzar LF | **RESUELTO** |
| Datos OHLCV y macro vacíos | Volúmenes eliminados | ⚠️ Requiere restauración manual de backups | **PARCIAL** |

---

## 1. Problemas de Base de Datos

### 1.1 Esquema config.models
**Problema**: El frontend y backend esperaban columnas que no existían.

**Solución aplicada**: El archivo `init-scripts/10-multi-model-schema.sql` ahora incluye:
```sql
-- Columnas completas para config.models
CREATE TABLE IF NOT EXISTS config.models (
    model_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,           -- ← Requerido por UI
    algorithm VARCHAR(20) NOT NULL,        -- ← Requerido para detectar demo mode
    version VARCHAR(20) NOT NULL,
    status VARCHAR(20) DEFAULT 'inactive',
    color VARCHAR(7) DEFAULT '#3B82F6',    -- ← Para visualización
    hyperparameters JSONB NOT NULL DEFAULT '{}'::jsonb,
    policy_config JSONB NOT NULL DEFAULT '{}'::jsonb,
    environment_config JSONB NOT NULL DEFAULT '{}'::jsonb,
    backtest_metrics JSONB DEFAULT '{}'::jsonb,  -- ← Métricas de backtest
    model_path VARCHAR(500),
    framework VARCHAR(50) DEFAULT 'stable-baselines3',
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

**Verificación**: ✅ Ya está aplicado en el init-script con `ON CONFLICT DO UPDATE`

### 1.2 Seed Data para Modelos
**Problema**: Los modelos `investor_demo` y `ppo_primary` no se creaban automáticamente.

**Solución aplicada**: El mismo `10-multi-model-schema.sql` incluye:
- `investor_demo` con `algorithm='SYNTHETIC'`
- `ppo_primary` con `algorithm='PPO'`

**Verificación**: ✅ Ya está aplicado con inserts idempotentes

---

## 2. Problemas de Código Python

### 2.1 DemoConfig.monthly_bias
**Problema**: `AttributeError: 'DemoConfig' object has no attribute 'monthly_bias'`

**Causa**: El `trade_generator.py` llamaba a `self.config.monthly_bias.get(month, 0)`
pero `DemoConfig` no tenía ese atributo.

**Solución aplicada** en `services/demo_mode/config.py`:
```python
@dataclass
class DemoConfig:
    # ... otros campos ...

    # Monthly bias for trade direction
    monthly_bias: Dict[int, float] = None

    def __post_init__(self):
        if self.monthly_bias is None:
            self.monthly_bias = {
                1: 0.3, 2: -0.1, 3: 0.1, 4: 0.4,
                5: -0.2, 6: -0.3, 7: -0.4, 8: -0.3,
                9: -0.5, 10: -0.4, 11: -0.3, 12: 0.0
            }
```

**Verificación**: ✅ Ya está aplicado en el código fuente

---

## 3. Problemas de Docker

### 3.1 entrypoint.sh Line Endings
**Problema**: `exec /entrypoint.sh: no such file or directory`

**Causa**: En Windows, los archivos .sh pueden guardarse con line endings CRLF,
lo que causa que el shebang `#!/bin/bash\r` no sea reconocido por Linux.

**Solución aplicada**:
1. Creado `.gitattributes` para forzar line endings LF:
```
# Shell scripts - ALWAYS use LF (critical for Docker)
*.sh text eol=lf
entrypoint.sh text eol=lf
```

2. Convertido manualmente con: `sed -i 's/\r$//' services/inference_api/entrypoint.sh`

**Verificación**: ✅ `.gitattributes` creado, pero necesita:
```bash
# Después de crear .gitattributes, normalizar archivos existentes:
git add --renormalize .
git commit -m "Normalize line endings"
```

---

## 4. Restauración de Datos

### 4.1 Datos OHLCV y Macro
**Problema**: Al hacer `docker-compose down -v`, se eliminan todos los datos.

**Ubicación de backups**:
```
backups/
├── trading_data/
│   ├── usdcop_m5_ohlcv_YYYYMMDD_HHMMSS.csv.gz
│   └── macro_indicators_daily_YYYYMMDD_HHMMSS.csv.gz
└── trades_history/
    └── trades_history_YYYYMMDD_HHMMSS.csv.gz
```

**Proceso de restauración** (manual):
```bash
# 1. Descomprimir backup más reciente
gunzip -c backups/trading_data/usdcop_m5_ohlcv_LATEST.csv.gz > /tmp/ohlcv.csv

# 2. Importar a PostgreSQL
docker exec -i usdcop-postgres-timescale psql -U admin -d usdcop_trading -c "\copy public.usdcop_m5_ohlcv FROM '/tmp/ohlcv.csv' CSV HEADER"
```

**Verificación**: ⚠️ Requiere automatización futura

---

## 5. Checklist para Próximo Arranque Limpio

### Antes de `docker-compose down -v`:
- [ ] Verificar que existen backups recientes
- [ ] Exportar datos si es necesario: `make backup-data` (si existe)

### Después de `docker-compose up -d`:
```bash
# 1. Esperar a que PostgreSQL esté listo
docker compose logs postgres -f  # Ctrl+C cuando muestre "ready to accept connections"

# 2. Verificar que init-scripts se ejecutaron
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c "SELECT COUNT(*) FROM config.models;"
# Debe retornar 2 (investor_demo y ppo_primary)

# 3. Verificar tablas principales
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c "\dt public.*"

# 4. Si faltan datos, restaurar desde backups
# (ver sección 4.1)

# 5. Verificar API de backtest
curl -s http://localhost:8003/api/v1/models
# Debe retornar ambos modelos

# 6. Test de backtest con investor_demo
curl -s -X POST http://localhost:8003/api/v1/backtest \
  -H "Content-Type: application/json" \
  -d '{"model_id":"investor_demo","start_date":"2025-12-01","end_date":"2025-12-05","initial_capital":10000}'
# Debe retornar trades generados
```

---

## 6. Comandos Útiles

### Verificar estado de servicios:
```bash
docker compose ps
```

### Ver logs de un servicio:
```bash
docker compose logs backtest-api --tail 50
```

### Reconstruir un servicio con cambios:
```bash
docker compose build --no-cache backtest-api
docker compose up -d backtest-api
```

### Verificar line endings de un archivo:
```bash
file services/inference_api/entrypoint.sh
# Debe decir "ASCII text" NO "ASCII text, with CRLF line terminators"
```

### Convertir line endings a LF:
```bash
sed -i 's/\r$//' services/inference_api/entrypoint.sh
```

---

## 7. Archivos Críticos

| Archivo | Propósito | Verificación |
|---------|-----------|--------------|
| `init-scripts/10-multi-model-schema.sql` | Crear config.models completo | Columnas y seed data |
| `services/demo_mode/config.py` | DemoConfig con monthly_bias | Atributo y __post_init__ |
| `services/inference_api/entrypoint.sh` | Arranque del contenedor | Line endings LF |
| `.gitattributes` | Forzar line endings | *.sh text eol=lf |

---

## Historial de Cambios

| Fecha | Cambio | Autor |
|-------|--------|-------|
| 2026-01-18 | Creación inicial del documento | Claude |
| 2026-01-18 | Agregado DemoConfig.monthly_bias fix | Claude |
| 2026-01-18 | Creado .gitattributes para line endings | Claude |
