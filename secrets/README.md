# Secrets Directory

This directory contains sensitive credentials and API keys for the USDCOP Trading System.

## Security Warning

- **NEVER** commit actual secret files to git
- All secret files are ignored via `.gitignore`
- Only `secrets.template.txt`, `.gitignore`, and this `README.md` are tracked

## Quick Start

1. Read `secrets.template.txt` for the list of required secrets
2. Create each secret file with the appropriate value
3. Set proper file permissions

### Unix/Linux/macOS

```bash
# Generate and create a secret
openssl rand -base64 32 > secrets/db_password.txt
chmod 600 secrets/db_password.txt
```

### Windows PowerShell

```powershell
# Generate a secret
[Convert]::ToBase64String((1..32 | ForEach-Object { Get-Random -Maximum 256 })) | Out-File -FilePath secrets\db_password.txt -NoNewline
```

## Usage with Docker

Docker Compose will mount these secrets at `/run/secrets/<secret_name>`:

```yaml
services:
  postgres:
    secrets:
      - db_password
    environment:
      POSTGRES_PASSWORD_FILE: /run/secrets/db_password

secrets:
  db_password:
    file: ./secrets/db_password.txt
```

## Usage in Python

```python
from src.core.secrets.secret_manager import SecretManager

# Initialize once
secrets = SecretManager()

# Get a secret (tries Docker secrets, then local files, then env vars)
db_password = secrets.get_secret("db_password")
api_key = secrets.get_secret("twelvedata_api_key_1")
```

## Required Secrets

| Secret Name | Description | Generation Method |
|-------------|-------------|-------------------|
| db_password | PostgreSQL password | `openssl rand -base64 32` |
| redis_password | Redis password | `openssl rand -base64 32` |
| minio_secret_key | MinIO secret key | `openssl rand -base64 32` |
| airflow_password | Airflow admin password | `openssl rand -base64 32` |
| airflow_fernet_key | Airflow encryption key | Python cryptography |
| grafana_password | Grafana admin password | `openssl rand -base64 32` |
| jwt_secret | API JWT secret | `openssl rand -base64 64` |

See `secrets.template.txt` for the complete list.
