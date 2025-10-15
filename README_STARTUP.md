# 🚀 Guía de Inicio Rápido - USDCOP Trading System

## Datos Disponibles
✅ **Backup incluido en el repositorio:**
- **Ubicación**: `data/backups/20251015_162604/`
- **Archivo**: `market_data.csv.gz` (1.1MB comprimido)
- **Registros**: 92,936 datos históricos de USDCOP
- **Período**: 2020-2025

## Inicio Rápido

### 1. Clonar el repositorio
```bash
git clone https://github.com/pspedro19/USDCOP-RL-Models.git
cd USDCOP-RL-Models
```

### 2. Iniciar el sistema
```bash
# Dar permisos de ejecución
chmod +x start-system.sh

# Iniciar todos los servicios
./start-system.sh
```

### 3. Acceder al sistema
- **Dashboard**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **MinIO Console**: http://localhost:9001 (admin/admin123)

## Estructura del Backup

El backup está versionado en Git y contiene:
```
data/backups/20251015_162604/
├── market_data.csv.gz    # Datos históricos comprimidos
└── README.md             # Documentación del backup
```

### Formato de datos
```csv
timestamp,symbol,price,bid,ask,volume,source,created_at
2020-01-02 07:30:00+00:00,USDCOP,3287.23,3287.23,3287.23,0,twelvedata,...
```

## Restaurar Datos Manualmente

Si necesitas restaurar los datos en PostgreSQL:

```python
import gzip
import pandas as pd
import psycopg2

# Leer backup
with gzip.open('data/backups/20251015_162604/market_data.csv.gz', 'rt') as f:
    df = pd.read_csv(f)

# Conectar a PostgreSQL
conn = psycopg2.connect(
    host='localhost',
    port=5432,
    database='usdcop_trading',
    user='admin',
    password='admin123'
)

# Cargar datos
df.to_sql('market_data', conn, if_exists='append', index=False)
```

## API Endpoints Principales

### Obtener datos históricos
```bash
curl http://localhost:8000/api/market/historical
```

### Obtener último precio
```bash
curl http://localhost:8000/api/market/latest
```

### WebSocket para tiempo real
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Precio:', data.price);
};
```

## Verificación del Sistema

### Comprobar servicios
```bash
sudo docker compose ps
```

### Ver logs
```bash
sudo docker compose logs -f dashboard
sudo docker compose logs -f trading-api
```

### Verificar datos en PostgreSQL
```bash
sudo docker exec -it usdcop-postgres-timescale psql -U admin -d usdcop_trading -c "SELECT COUNT(*) FROM market_data;"
```

## Troubleshooting

### Si los servicios no inician:
```bash
# Reiniciar Docker
sudo systemctl restart docker

# Limpiar y reiniciar
sudo docker compose down
sudo docker compose up -d
```

### Si no hay datos:
```bash
# Ejecutar script de backup
python3 backup_database.py
```

## Notas Importantes

✅ **Los backups están incluidos en el repositorio** gracias a la configuración de `.gitignore`:
```gitignore
!data/backups/     # Esta línea asegura que los backups se incluyan
```

✅ **Datos listos para usar**: No necesitas descargar datos adicionales, todo está incluido.

✅ **Sistema completo**: Frontend, Backend, APIs y datos históricos configurados.

---

Para más información, revisa la documentación completa en `/docs`