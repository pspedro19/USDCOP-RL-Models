# Remediación de fuga de `.env` — 2026-07-09

## Qué pasó
Dos commits de enero-2026 (`ee91273`, `1d41812`) incluyeron un `.env` **real** con credenciales.
El repositorio `pspedro19/USDCOP-RL-Models` era **público**, por lo que esas claves estuvieron
accesibles vía `raw.githubusercontent.com/.../ee91273/.env`.

## Acciones ejecutadas (2026-07-09)
- [x] **Repo pasado a PRIVADO** (API GitHub, verificado `visibility=private`; la URL de la fuga ahora da 404 anónimo).
- [x] **Historia purgada** con `git filter-repo` — eliminados `/.env` y `data/backups/full_backup_20260114_153824/env/.env` de TODOS los commits. Verificado: 0 `.env` reales en la historia; los `.env.example` se conservan.
- [x] **`main` reescrito y force-push** con la historia limpia + todo el trabajo de la sesión.
- [x] **Backup de seguridad** pre-purga en `C:\tmp\pre-purge-backup.bundle` (por si hiciera falta revertir).

## PENDIENTE — SOLO EL OPERADOR (requiere acceso a cada proveedor)
Aunque el repo ya es privado, las claves **estuvieron públicas** → asumir comprometidas y **ROTAR TODAS**:

| Clave | Dónde rotar |
|---|---|
| `ANTHROPIC_API_KEY` | console.anthropic.com → API Keys → revocar+crear |
| `DEEPSEEK_API_KEY` | platform.deepseek.com |
| `FRED_API_KEY` | fred.stlouisfed.org/docs/api |
| `OPENAI/AZURE` (si aplica) | portal.azure.com |
| `JWT_SECRET` | regenerar `openssl rand -base64 48` + actualizar `.env` |
| `AIRFLOW_FERNET_KEY` | `python -c "from cryptography.fernet import Fernet;print(Fernet.generate_key().decode())"` (⚠ rota conexiones existentes) |
| `AIRFLOW_SECRET_KEY` | `openssl rand -base64 32` |
| `POSTGRES/REDIS/MINIO/GRAFANA/PGADMIN/AIRFLOW` passwords | `.env` + recrear contenedores |
| `EMAIL_SMTP_*` | proveedor SMTP |
| **MEXC / Binance API keys** | (ya marcadas por exposición en chat) — exchange → borrar+crear, **SPOT-only, sin retiro** |

Tras rotar: `docker compose down && docker compose up -d` para recargar `.env`.

## Higiene adicional recomendada
- Los forks/clones previos y las cachés de GitHub pueden retener la historia vieja → contactar **GitHub Support** para purga de cachés si el repo tuvo forks públicos.
- Considerar `gitleaks` como hook pre-commit (ya hay workflow de security en CI).
- Regla ya vigente: `.gitignore` bloquea `.env*` con excepción solo para `*.example` (verificado).
