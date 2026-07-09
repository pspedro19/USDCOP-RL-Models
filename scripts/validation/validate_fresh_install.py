#!/usr/bin/env python3
"""Validate that the USDCOP trading system is correctly set up.

Runs health checks against all core services after docker-compose up.
Use this after a fresh install or after recovering from an outage.

Usage:
    python scripts/validate_fresh_install.py          # Full validation
    python scripts/validate_fresh_install.py --quick   # Skip slow checks
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Color output helpers
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def ok(msg: str):
    print(f"  {GREEN}[OK]{RESET} {msg}")


def fail(msg: str):
    print(f"  {RED}[FAIL]{RESET} {msg}")


def warn(msg: str):
    print(f"  {YELLOW}[WARN]{RESET} {msg}")


def check_docker_running() -> bool:
    """Check if Docker is running."""
    try:
        result = subprocess.run(
            ["docker", "info"], capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_container(name: str) -> bool:
    """Check if a container is running."""
    try:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", name],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() == "true"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_container_health(name: str) -> str:
    """Check container health status. Returns 'healthy', 'unhealthy', 'starting', or 'none'."""
    try:
        result = subprocess.run(
            [
                "docker",
                "inspect",
                "-f",
                "{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}",
                name,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "error"


def check_http_endpoint(url: str, timeout: int = 5) -> bool:
    """Check if an HTTP endpoint responds."""
    try:
        import urllib.request

        req = urllib.request.Request(url)
        resp = urllib.request.urlopen(req, timeout=timeout)
        return resp.status == 200
    except Exception:
        return False


def check_postgres_query(query: str) -> str:
    """Run a query against PostgreSQL via docker exec."""
    try:
        result = subprocess.run(
            [
                "docker",
                "exec",
                "usdcop-postgres-timescale",
                "psql",
                "-U",
                "admin",
                "-d",
                "usdcop_trading",
                "-t",
                "-A",
                "-c",
                query,
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def check_secrets() -> tuple[int, int]:
    """Check if secret files exist and have content. Returns (present, total)."""
    secrets_dir = PROJECT_ROOT / "secrets"
    expected = [
        "db_password.txt",
        "redis_password.txt",
        "minio_secret_key.txt",
        "airflow_password.txt",
        "airflow_fernet_key.txt",
        "airflow_secret_key.txt",
        "grafana_password.txt",
        "pgadmin_password.txt",
    ]
    present = 0
    for name in expected:
        path = secrets_dir / name
        if path.exists() and not path.is_dir() and path.stat().st_size > 0:
            present += 1
    return present, len(expected)


def check_seed_files() -> list[str]:
    """Check if critical seed files exist."""
    missing = []
    seeds = [
        "seeds/latest/usdcop_daily_ohlcv.parquet",
        "data/pipeline/04_cleaning/output/MACRO_DAILY_CLEAN.parquet",
    ]
    for seed in seeds:
        path = PROJECT_ROOT / seed
        if not path.exists() or path.stat().st_size < 1000:
            missing.append(seed)
    return missing


def main():
    parser = argparse.ArgumentParser(description="Validate USDCOP fresh install")
    parser.add_argument(
        "--quick", action="store_true", help="Skip slow checks (DB queries, HTTP)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("USDCOP Trading System — Fresh Install Validation")
    print("=" * 60)

    failures = 0
    warnings = 0

    # 1. Docker
    print("\n1. Docker")
    if check_docker_running():
        ok("Docker is running")
    else:
        fail("Docker is not running")
        print("\n   Cannot continue. Start Docker Desktop first.")
        return 1

    # 2. Secrets
    print("\n2. Secrets")
    present, total = check_secrets()
    if present == total:
        ok(f"All {total} secret files present")
    elif present > 0:
        warn(f"{present}/{total} secrets present. Run: python scripts/generate_secrets.py --random --force")
        warnings += 1
    else:
        fail(f"No secrets generated. Run: python scripts/generate_secrets.py --random --force")
        failures += 1

    # 3. Seed files
    print("\n3. Seed Files")
    missing_seeds = check_seed_files()
    if not missing_seeds:
        ok("Critical seed files present")
    else:
        for s in missing_seeds:
            fail(f"Missing: {s}")
            failures += 1

    # 4. Core containers
    print("\n4. Core Containers")
    core_containers = {
        "usdcop-postgres-timescale": "PostgreSQL + TimescaleDB",
        "usdcop-redis": "Redis",
        "usdcop-airflow-scheduler": "Airflow Scheduler",
        "usdcop-airflow-webserver": "Airflow Webserver",
        "usdcop-dashboard": "Next.js Dashboard",
    }
    for container, label in core_containers.items():
        if check_container(container):
            health = check_container_health(container)
            if health == "healthy":
                ok(f"{label} (healthy)")
            elif health == "none":
                ok(f"{label} (running, no healthcheck)")
            elif health == "starting":
                warn(f"{label} (starting...)")
                warnings += 1
            else:
                warn(f"{label} (running, health={health})")
                warnings += 1
        else:
            fail(f"{label} not running")
            failures += 1

    # 5. Monitoring stack
    print("\n5. Monitoring Stack")
    monitoring = {
        "usdcop-prometheus": "Prometheus",
        "usdcop-grafana": "Grafana",
        "usdcop-loki": "Loki",
        "usdcop-promtail": "Promtail",
        "usdcop-alertmanager": "AlertManager",
    }
    for container, label in monitoring.items():
        if check_container(container):
            health = check_container_health(container)
            if health in ("healthy", "none"):
                ok(f"{label}")
            else:
                warn(f"{label} ({health})")
                warnings += 1
        else:
            warn(f"{label} not running (optional)")
            warnings += 1

    if args.quick:
        print("\n(Skipping slow checks — use without --quick for full validation)")
    else:
        # 6. Database content
        print("\n6. Database Content")
        ohlcv_count = check_postgres_query(
            "SELECT COUNT(*) FROM usdcop_m5_ohlcv WHERE symbol='USD/COP'"
        )
        if ohlcv_count and int(ohlcv_count) > 0:
            ok(f"OHLCV: {int(ohlcv_count):,} rows")
        else:
            fail("OHLCV table empty or inaccessible")
            failures += 1

        macro_count = check_postgres_query(
            "SELECT COUNT(*) FROM macro_indicators_daily"
        )
        if macro_count and int(macro_count) > 0:
            ok(f"Macro: {int(macro_count):,} rows")
        else:
            warn("Macro table empty (run macro backfill)")
            warnings += 1

        # Check migrations
        h5_tables = check_postgres_query(
            "SELECT COUNT(*) FROM pg_tables WHERE tablename IN ('forecast_h5_predictions','forecast_h5_signals','news_articles','weekly_analysis')"
        )
        if h5_tables and int(h5_tables) >= 4:
            ok(f"Migrations 043-046 applied ({h5_tables}/4 tables)")
        else:
            warn(f"Missing migration tables ({h5_tables or 0}/4). Run: python scripts/db_migrate.py")
            warnings += 1

        # 7. HTTP endpoints
        print("\n7. HTTP Endpoints")
        endpoints = {
            "http://localhost:5000/api/health": "Dashboard",
            "http://localhost:9090/-/healthy": "Prometheus",
            "http://localhost:3002/api/health": "Grafana",
            "http://localhost:9094/-/healthy": "AlertManager",
        }
        for url, label in endpoints.items():
            if check_http_endpoint(url):
                ok(f"{label} ({url})")
            else:
                warn(f"{label} not responding ({url})")
                warnings += 1

    # Summary
    print("\n" + "=" * 60)
    if failures == 0 and warnings == 0:
        print(f"{GREEN}All checks passed!{RESET}")
    elif failures == 0:
        print(f"{YELLOW}{warnings} warnings, 0 failures{RESET}")
        print("System is functional but some components need attention.")
    else:
        print(f"{RED}{failures} failures, {warnings} warnings{RESET}")
        print("Fix failures before proceeding.")
    print("=" * 60)

    return 1 if failures > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
