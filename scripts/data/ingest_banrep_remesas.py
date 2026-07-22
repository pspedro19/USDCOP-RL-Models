#!/usr/bin/env python3
"""
Ingest BanRep workers' remittances (remesas) monthly series — TAREA C2
======================================================================

Point-in-time monthly ingestion of "Ingresos de Remesas de trabajadores,
mensual" (USD millions) from Banco de la República, 2000-01 -> today.
0 trials: data only, no study is opened here.

Source (documented 2026-07-21)
------------------------------
- Human catalog page:
  https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/4150/remesas_trabajadores/
- Machine endpoint (JSON REST used by the SUAMECA Angular app):
  POST https://suameca.banrep.gov.co/buscador-de-series/rest/buscadorSeriesRestService/consultaDatosSeries
  body: {"series":[{"idSerie":15363,"idPeriodicidades":[9]}],
         "fechaInicio":20000101,"fechaFin":<yyyymmdd today>}
  -> [{"data": [[epoch_ms_bogota_midnight_eom, value_usd_mn], ...], ...}]
- Series: id=15363, idCargue=REMESAS_MENSUAL, unit "Millones de USD",
  tipoDato 9 = "Dato fin de mes".

Publication lag (defines the PIT date)
--------------------------------------
BanRep loads month M near the end of M+1. Observed from series metadata on
2026-07-21: May-2026 value loaded 2026-06-26 (fechaUltimoCargue); June-2026
scheduled 2026-07-24 (fechaProximoCargue). We store the conservative rule
    published_at = last calendar day of (month + 1)
so a PIT join on published_at <= as_of never sees a value before the market
could have.

Usage
-----
    set -a; . ./.env; set +a          # load POSTGRES_* credentials
    python scripts/data/ingest_banrep_remesas.py                 # full ingest
    python scripts/data/ingest_banrep_remesas.py --dry-run       # no DB writes
    python scripts/data/ingest_banrep_remesas.py --from-json f.json  # offline parse

Applies migration database/migrations/066_banrep_remesas_monthly.sql
automatically if the table does not exist (psql-style: executes the SQL file).
"""

from __future__ import annotations

import argparse
import calendar
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # scripts/data/<this> -> repo root
MIGRATION_FILE = PROJECT_ROOT / "database" / "migrations" / "066_banrep_remesas_monthly.sql"

SERIES_ID = 15363
TIPO_DATO_EOM = 9  # "Dato fin de mes"
API_URL = (
    "https://suameca.banrep.gov.co/buscador-de-series/rest/"
    "buscadorSeriesRestService/consultaDatosSeries"
)
CATALOG_URL = (
    "https://suameca.banrep.gov.co/estadisticas-economicas/"
    "informacionSerie/4150/remesas_trabajadores/"
)
SOURCE_TAG = f"banrep_suameca_{SERIES_ID}"

# Gaps documented as legitimate (none known as of 2026-07: series is complete
# Jan-2000 onward). Add (year, month) tuples here if BanRep ever skips a month.
DOCUMENTED_MISSING_MONTHS: set[tuple[int, int]] = set()


# ---------------------------------------------------------------------------
# Env / DB helpers
# ---------------------------------------------------------------------------

def _load_dotenv_fallback() -> None:
    """Populate POSTGRES_* from repo .env if not already in the environment."""
    if os.environ.get("POSTGRES_PASSWORD"):
        return
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key.startswith("POSTGRES") and key not in os.environ:
            os.environ[key] = value


def get_connection():
    import psycopg2

    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        dbname=os.environ.get("POSTGRES_DB", "usdcop_trading"),
        user=os.environ.get("POSTGRES_USER", "admin"),
        password=os.environ.get("POSTGRES_PASSWORD", ""),
    )


def ensure_table(conn) -> None:
    """Apply migration 066 if macro_remesas_monthly does not exist."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = 'macro_remesas_monthly'
            )
            """
        )
        exists = cur.fetchone()[0]
    if exists:
        return
    print(f"[MIGRATE] applying {MIGRATION_FILE.name}")
    sql = MIGRATION_FILE.read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
    print("[MIGRATE] macro_remesas_monthly created")


# ---------------------------------------------------------------------------
# Download + parse
# ---------------------------------------------------------------------------

def fetch_series(start: date, end: date, retries: int = 3) -> list:
    """POST to the SUAMECA REST endpoint; return raw JSON list."""
    import requests

    payload = {
        "series": [{"idSerie": SERIES_ID, "idPeriodicidades": [TIPO_DATO_EOM]}],
        "fechaInicio": int(start.strftime("%Y%m%d")),
        "fechaFin": int(end.strftime("%Y%m%d")),
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    }
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(API_URL, json=payload, headers=headers, timeout=90)
            resp.raise_for_status()
            body = resp.json()
            if not isinstance(body, list) or not body or "data" not in body[0]:
                raise ValueError(f"Unexpected API response shape: {str(body)[:200]}")
            return body
        except Exception as exc:  # noqa: BLE001 — retry then surface
            last_err = exc
            print(f"[WARN] fetch attempt {attempt}/{retries} failed: {exc}")
    raise RuntimeError(
        f"Could not download series {SERIES_ID} from {API_URL}: {last_err}\n"
        f"BLOCKED fallback: download manually from {CATALOG_URL} "
        f"(Exportar), save the JSON of consultaDatosSeries and re-run with "
        f"--from-json <file>."
    )


def epoch_ms_to_month(ms: int) -> date:
    """SUAMECA stamps = Bogota-midnight of the month's last day (UTC-5).

    Converting the instant to UTC yields e.g. 2000-01-31T05:00Z, whose UTC
    calendar date is still inside the reference month, so year/month of the
    UTC date identify the month unambiguously. Instant-based, no COT localize.
    """
    ts = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    return date(ts.year, ts.month, 1)


def published_at_for(month: date) -> date:
    """Conservative PIT date: last calendar day of month+1."""
    y, m = (month.year, month.month + 1) if month.month < 12 else (month.year + 1, 1)
    return date(y, m, calendar.monthrange(y, m)[1])


def parse_rows(raw: list) -> list[dict]:
    serie = raw[0]
    rows = []
    for ms, value in serie["data"]:
        month = epoch_ms_to_month(int(ms))
        rows.append(
            {
                "month": month,
                "remesas_usd_mn": float(value),
                "published_at": published_at_for(month),
                "source": SOURCE_TAG,
            }
        )
    rows.sort(key=lambda r: r["month"])
    return rows


# ---------------------------------------------------------------------------
# Validation (descriptive — hard-fails only on data integrity)
# ---------------------------------------------------------------------------

def _add_months(d: date, n: int) -> date:
    m = d.year * 12 + (d.month - 1) + n
    return date(m // 12, m % 12 + 1, 1)


def _month_iter(start: date, end: date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield date(y, m, 1)
        y, m = (y, m + 1) if m < 12 else (y + 1, 1)


def validate(rows: list[dict]) -> None:
    if not rows:
        raise SystemExit("[FAIL] no rows parsed")

    # 1) no negatives
    negatives = [r for r in rows if r["remesas_usd_mn"] < 0]
    if negatives:
        raise SystemExit(f"[FAIL] {len(negatives)} negative values, e.g. {negatives[:3]}")

    # 2) no duplicated months
    months = [r["month"] for r in rows]
    if len(months) != len(set(months)):
        raise SystemExit("[FAIL] duplicated months in parsed data")

    # 3) gaps: ANY undocumented missing month is a hard FAIL. The expected span
    #    runs from the REQUESTED start (2000-01) to the last month the publication
    #    lag says should already be loaded — so holes at either edge of what the
    #    source returned are caught too, not just holes between first/last row.
    have = set(months)
    today = date.today()
    # dato de M disponible ~día 26 de M+1 (regla PIT documentada arriba)
    last_expected = _add_months(date(today.year, today.month, 1),
                                -2 if today.day < 27 else -1)
    expected_start = date(2000, 1, 1)
    missing = [
        mo for mo in _month_iter(expected_start, max(last_expected, months[-1]))
        if mo not in have and (mo.year, mo.month) not in DOCUMENTED_MISSING_MONTHS
    ]
    if missing:
        raise SystemExit(
            f"[FAIL] {len(missing)} undocumented missing months (first 12: "
            f"{missing[:12]}), expected span {expected_start} -> "
            f"{max(last_expected, months[-1])}. If a hole is legitimate, add it "
            f"to DOCUMENTED_MISSING_MONTHS."
        )
    print("[OK] no missing months in expected span")

    # 4) seasonality sanity check (descriptive only, never blocks):
    #    December remittances are expected to run above the overall mean.
    full_years = {r["month"].year for r in rows if r["month"].month == 12}
    dec_vals = [r["remesas_usd_mn"] for r in rows if r["month"].month == 12]
    all_vals = [r["remesas_usd_mn"] for r in rows if r["month"].year in full_years]
    if dec_vals and all_vals:
        dec_mean = sum(dec_vals) / len(dec_vals)
        overall = sum(all_vals) / len(all_vals)
        verdict = "as expected (dic alto)" if dec_mean > overall else "UNEXPECTED"
        print(
            f"[SEASONALITY] mean Dec = {dec_mean:,.1f} vs overall mean = "
            f"{overall:,.1f} USD mn -> {verdict}"
        )

    print(f"[OK] validation passed: {len(rows)} rows, "
          f"{months[0]} -> {months[-1]}")


def report_by_decade(rows: list[dict]) -> None:
    decades: dict[int, int] = {}
    for r in rows:
        decades[r["month"].year // 10 * 10] = decades.get(r["month"].year // 10 * 10, 0) + 1
    print("[REPORT] rows per decade:")
    for dec in sorted(decades):
        print(f"    {dec}s : {decades[dec]} months")


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

UPSERT_SQL = """
INSERT INTO macro_remesas_monthly (month, remesas_usd_mn, published_at, source)
VALUES (%(month)s, %(remesas_usd_mn)s, %(published_at)s, %(source)s)
ON CONFLICT (month) DO UPDATE SET
    remesas_usd_mn = EXCLUDED.remesas_usd_mn,
    published_at   = EXCLUDED.published_at,
    source         = EXCLUDED.source,
    ingested_at    = now()
"""


def upsert(conn, rows: list[dict]) -> None:
    from psycopg2.extras import execute_batch

    with conn.cursor() as cur:
        execute_batch(cur, UPSERT_SQL, rows, page_size=500)
    conn.commit()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*), min(month), max(month) FROM macro_remesas_monthly"
        )
        n, lo, hi = cur.fetchone()
    print(f"[DB] macro_remesas_monthly now holds {n} rows, {lo} -> {hi}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest BanRep monthly workers' remittances (PIT, USD mn)."
    )
    parser.add_argument("--start", default="2000-01-01", help="range start (default 2000-01-01)")
    parser.add_argument("--dry-run", action="store_true", help="parse+validate only, no DB writes")
    parser.add_argument(
        "--from-json", default=None,
        help="offline mode: parse a saved consultaDatosSeries JSON instead of downloading",
    )
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.today() + timedelta(days=31)  # inclusive of any freshly loaded month

    print("=" * 70)
    print("BanRep remesas mensuales -> macro_remesas_monthly (TAREA C2)")
    print(f"  serie   : {SERIES_ID} (Ingresos de Remesas de trabajadores, mensual)")
    print(f"  range   : {start} -> {end}")
    print(f"  source  : {API_URL}")
    print("=" * 70)

    if args.from_json:
        raw = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
        print(f"[OFFLINE] parsed {args.from_json}")
    else:
        raw = fetch_series(start, end)

    rows = parse_rows(raw)
    rows = [r for r in rows if r["month"] >= date(start.year, start.month, 1)]

    validate(rows)
    report_by_decade(rows)

    if args.dry_run:
        print("[DRY-RUN] skipping DB writes")
        return 0

    _load_dotenv_fallback()
    conn = get_connection()
    try:
        ensure_table(conn)
        upsert(conn, rows)
    finally:
        conn.close()

    print("[DONE] ingest complete (0 trials: data only, no study opened)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
