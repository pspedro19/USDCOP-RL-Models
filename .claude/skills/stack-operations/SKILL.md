---
name: stack-operations
description: Bring the Docker stack up or down in the right mode, verify services are actually healthy, and diagnose a service that will not start. Use when starting work for the day, when a page or API is unreachable, or when the user asks to run the system locally.
---

# Stack operations

## Modes

| Mode | Command | Servicios reales | Añade |
|---|---|---|---|
| Compact | `make compact` | **18** | uso diario: DB, Airflow, APIs, MLflow, SignalBridge, dashboard |
| Compact + monitoring | `make compact-monitoring` | **21** | Prometheus, Grafana, AlertManager |
| Enterprise | `make docker-up` | **22** (24 con perfil `full`) | Vault, Jaeger, Loki, Promtail, pgAdmin |

> Los conteos salen de parsear los compose (`docker-compose.compact.yml`, `docker-compose.yml`),
> no del texto de ayuda del Makefile — ese decía 12/15/25+ y **los tres estaban mal**.
> Verifica siempre con `docker compose ... config --services | wc -l`, nunca con la doc.

Bajar: `make compact-down` (incluye el perfil monitoring) · `make docker-down`.

## Arrancar

```powershell
docker compose -f docker-compose.compact.yml config --services   # validar ANTES de levantar
make compact
docker compose -f docker-compose.compact.yml ps
```

`config` falla rápido y barato ante un YAML roto o una variable sin definir; `up` falla lento y a
medias. Valida primero.

## Verificar que está sano de verdad

"Levantado" ≠ "funcionando". Comprueba las dependencias en orden:

```powershell
# 1. Postgres responde y tiene datos (un contenedor sano con DB vacía es el fallo mas comun)
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c "SELECT COUNT(*) FROM usdcop_m5_ohlcv;"

# 2. Airflow cargó los DAGs sin errores de import
docker exec usdcop-airflow-scheduler airflow dags list-import-errors

# 3. Puertos que deben responder
#    Airflow 8080 · SignalBridge 8085 · MLflow 5001 · dashboard 3000 · Grafana 3002
```

Si la DB está vacía tras un arranque en frío, **no es un problema de stack** → skill `data-recovery`.

## Diagnosticar un servicio que no arranca

```powershell
docker compose -f docker-compose.compact.yml ps        # ¿qué estado tiene?
make docker-logs SERVICE=<nombre>                      # sus logs
docker inspect <contenedor> --format '{{.State.Health.Status}}'
```

Orden de sospecha: variable de entorno faltante → dependencia aún no sana → puerto ocupado →
volumen con permisos → imagen desactualizada.

## Constraints

- **Docker Desktop debe estar corriendo** antes de cualquier `make compact`. En Windows es la
  causa nº 1 de fallo, y el error que devuelve no lo dice claramente.
- **Nunca `docker-compose down -v`** en local sin preguntar: borra volúmenes y con ellos la DB.
  El restore existe, pero cuesta una recuperación completa.
- `public/data/production/deploy_status.json` lo escribe el contenedor y **debe quedar fuera del
  build context** — su modo NTFS rompe el tar de `docker build`.
- No arregles la frescura de datos reiniciando contenedores: son problemas distintos.
- Si cambias conteos de servicios, **no actualices la prosa** — el texto de ayuda del Makefile ya
  demostró que se pudre. Deriva el número del compose.
