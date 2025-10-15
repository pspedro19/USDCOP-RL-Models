# USDCOP Market Data Backup

## Información del Backup
- **Fecha**: 2025-10-15 16:26:04
- **Archivo**: market_data.csv.gz (comprimido)
- **Total de registros**: 92,936
- **Período**: 2020-01-02 hasta 2025-10-10
- **Par**: USDCOP
- **Fuente**: TwelveData

## Estructura de datos
```csv
timestamp,symbol,price,bid,ask,volume,source,created_at
```

## Cómo usar
Para descomprimir y ver los datos:
```bash
gunzip -k market_data.csv.gz
head market_data.csv
```

## Notas
- Backup único consolidado sin duplicados
- Datos históricos completos del par USDCOP
- Archivo comprimido para ahorrar espacio (1.1MB vs 9.5MB)