"""BL-40 + BL-17 — puebla la espina de identidad `reference.*` desde los SSOT.

La migración 072 creó siete tablas de identidad canónica. Seis quedaron **vacías**:
sólo `bar_interval` tenía filas, sembradas por la propia migración. Y como
`provider_symbol` exige `instrument`, que exige `asset`, que exige `calendar`, ningún
consumidor podía apoyarse en la espina: existía el esquema, no la identidad.

Este script es el productor que faltaba. Su regla única:

    **todo campo sale de un SSOT existente o el activo NO entra.**

No hay defaults de relleno. Concretamente:

* `calendar` / `asset` salen del `AssetProfile` (`config/assets/*.yaml`). Ojo con la
  anualización: `AnnualizationRegistry` sólo rechaza valores no-enteros o no-positivos,
  y **nunca llega a ver el hueco**, porque el contrato aplica antes un default silencioso
  de 250 (`asset_profile.py`: `d.get("trading_days_per_year", 250)`). Medido: USD/COP
  vale 261, así que ese default anualizaría un 4% mal sin avisar. Por eso este seed
  comprueba el **YAML crudo** y aborta si nadie la declaró.
* `provider` / `provider_symbol` NO se declaran en ningún YAML: se **miden**. Un
  proveedor existe si escribió filas OHLCV reales, y su `authoritative_for` es el
  conjunto de activos para los que efectivamente escribió. Inventar aquí un proveedor
  plausible sería exactamente el fallo que la espina existe para impedir.

Consecuencia visible y deliberada: `USD/BRL`, `USD/MXN` y `SPY` tienen filas OHLCV
reales pero **no tienen `AssetProfile`**, así que se quedan fuera y el script lo
reporta. Es la regla funcionando, no un olvido. `SPY` además no es `SPX/500`: la
propia 072 declara que "SPX, SPY and ES are distinct instruments"; colapsarlos para
que "cuadre" destruiría la identidad que la tabla protege.

Idempotente: se puede correr N veces. Sólo DML — la espina se puebla sobre tablas ya
creadas por 072; este script nunca hace DDL.

Uso:
    python -m scripts.data.seed_reference_spine --dry-run   # plan, sin escribir
    python -m scripts.data.seed_reference_spine --apply

Contract: CTR-QLAB-FABRIC-004 (BL-40, BL-17) · Date: 2026-08-04
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys

import yaml
from pathlib import Path
from typing import Any, Iterable, Mapping, NamedTuple

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.contracts.asset_profile import AssetProfile, load_asset_profile  # noqa: E402
from src.metrics.annualization import AnnualizationRegistry  # noqa: E402

ASSETS_DIR = REPO / "config" / "assets"

#: Tablas OHLCV que registran `(source, symbol)` — la evidencia de qué proveedor es real.
OHLCV_SOURCES = ("usdcop_m5_ohlcv", "asset_daily_ohlcv")

#: `instrument_type` derivado de la `asset_class` DECLARADA. No es un default: es una
#: traducción total y explícita. Una `asset_class` nueva **rompe** el seed en vez de
#: recibir un tipo genérico, que es lo que haría un `.get(..., 'spot')`.
INSTRUMENT_TYPE_BY_CLASS = {
    "fx": "spot",
    "crypto": "spot",
    "commodity": "spot",
    "equity_index": "index",
}


class SpineError(RuntimeError):
    """La espina no puede poblarse sin inventar un dato. Falla cerrado."""


class ProviderFact(NamedTuple):
    """Un `(proveedor, símbolo)` que **existe porque escribió filas**."""

    provider_id: str
    symbol: str
    rows: int
    table: str


# --------------------------------------------------------------------------- SSOT


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def calendar_id_for(profile: AssetProfile) -> str:
    """Identidad del calendario derivada de lo declarado: modo + zona horaria.

    Dos activos con el mismo modo y la misma zona comparten calendario de verdad; no
    se les fabrica uno propio para que "cada activo tenga el suyo".
    """
    session = profile.session
    return f"{_slug(session.mode)}__{_slug(session.timezone)}"


def _annualization_is_declared(assets_dir: Path, asset_id: str) -> bool:
    """¿El YAML declara `session.trading_days_per_year` EXPLÍCITAMENTE?

    Hay que mirar el YAML crudo, no el `AssetProfile`: el contrato aplica un default
    silencioso de 250 (`asset_profile.py`, `d.get("trading_days_per_year", 250)`), de
    modo que un activo que no la declare llega al registry ya "válido" y
    `AnnualizationRegistry` —que sólo rechaza no-enteros o no-positivos— nunca ve el
    hueco. Medido: para USD/COP el valor real es 261, así que un default de 250
    anualizaría un 4% mal **en silencio**, que es precisamente el relleno que esta
    espina no admite.
    """
    raw = yaml.safe_load((assets_dir / f"{asset_id}.yaml").read_text(encoding="utf-8")) or {}
    session = raw.get("session")
    if not isinstance(session, Mapping):
        return False
    valor = session.get("trading_days_per_year")
    return isinstance(valor, int) and not isinstance(valor, bool) and valor > 0


def declared_assets(assets_dir: Path = ASSETS_DIR) -> dict[str, AssetProfile]:
    """Perfiles declarados, con su anualización resuelta y **realmente** declarada.

    Un activo sin anualización explícita **aborta el seed** en vez de ser excluido en
    silencio: excluirlo lo haría desaparecer de la identidad canónica sin que nadie lo
    note, que es el mismo modo de fallo que el default que este chequeo impide.
    """
    registry = AnnualizationRegistry.load(assets_dir)
    perfiles: dict[str, AssetProfile] = {}
    for asset_id in sorted(registry.profiles):
        if not _annualization_is_declared(assets_dir, asset_id):
            raise SpineError(
                f"{asset_id}: no declara 'session.trading_days_per_year'. El contrato "
                "le asignaría 250 en silencio; la espina no acepta una anualización "
                "que nadie declaró"
            )
        # Falla cerrado también si el valor no es un entero positivo.
        registry.periods_per_year(asset_id, "P1D")
        perfiles[asset_id] = load_asset_profile(asset_id, assets_dir=assets_dir)
    return perfiles


def calendar_rows(perfiles: Mapping[str, AssetProfile]) -> list[dict[str, Any]]:
    filas: dict[str, dict[str, Any]] = {}
    for profile in perfiles.values():
        cid = calendar_id_for(profile)
        session = dataclasses.asdict(profile.session)
        session["days"] = list(session["days"])  # JSONB no admite tuplas
        filas.setdefault(
            cid,
            {
                "calendar_id": cid,
                "timezone": profile.session.timezone,
                "calendar_kind": profile.session.mode,
                "session_definition": json.dumps(session, sort_keys=True),
                "version": "asset-profile-v1",
            },
        )
    return [filas[k] for k in sorted(filas)]


def asset_rows(perfiles: Mapping[str, AssetProfile]) -> list[dict[str, Any]]:
    filas = []
    for asset_id, profile in perfiles.items():
        clase = profile.asset_class
        if clase not in INSTRUMENT_TYPE_BY_CLASS:
            raise SpineError(
                f"{asset_id}: asset_class {clase!r} no tiene traducción declarada a "
                f"instrument_type (conocidas: {sorted(INSTRUMENT_TYPE_BY_CLASS)}). "
                "Añádela explícitamente en vez de dejar que reciba un tipo genérico"
            )
        filas.append(
            {
                "asset_id": asset_id,
                "display_name": profile.display_name,
                "asset_class": clase,
                "quote_currency": profile.quote_ccy,
                # Ya validada como declarada y positiva en `declared_assets`.
                "annualization": profile.session.trading_days_per_year,
                "calendar_id": calendar_id_for(profile),
            }
        )
    return filas


def instrument_rows(perfiles: Mapping[str, AssetProfile]) -> list[dict[str, Any]]:
    return [
        {
            "canonical_symbol": profile.symbol,
            "asset_id": asset_id,
            "instrument_type": INSTRUMENT_TYPE_BY_CLASS[profile.asset_class],
            "base_currency": profile.base_ccy,
            "quote_currency": profile.quote_ccy,
        }
        for asset_id, profile in perfiles.items()
    ]


# ------------------------------------------------------------------- evidencia real


def measure_provider_facts(conn) -> list[ProviderFact]:
    """Lee de la base qué `(source, symbol)` escribieron filas de verdad."""
    hechos: list[ProviderFact] = []
    for tabla in OHLCV_SOURCES:
        # Una tabla por transacción: `usdcop_m5_ohlcv` es un hypertable y sus locks de
        # chunk siguen retenidos mientras la transacción viva. Agrupar ambas lecturas en
        # una sola agota `max_locks_per_transaction` en la SEGUNDA tabla, no en la
        # primera — medido. El `SET` evita además el paralelismo, que multiplica locks.
        with conn.cursor() as cur:
            cur.execute("SET max_parallel_workers_per_gather = 0")
            cur.execute(
                f"SELECT source, symbol, COUNT(*) FROM {tabla} "  # noqa: S608 - lista fija
                "WHERE source IS NOT NULL AND symbol IS NOT NULL GROUP BY 1, 2"
            )
            hechos.extend(
                ProviderFact(src, sym, int(n), tabla) for src, sym, n in cur.fetchall()
            )
        conn.rollback()  # lectura pura: libera los locks antes de la siguiente tabla
    return sorted(hechos)


def split_facts(
    hechos: Iterable[ProviderFact], perfiles: Mapping[str, AssetProfile]
) -> tuple[list[ProviderFact], list[ProviderFact]]:
    """Separa los hechos mapeables de los que se quedan fuera por no estar declarados."""
    por_simbolo = {p.symbol: aid for aid, p in perfiles.items()}
    dentro = [h for h in hechos if h.symbol in por_simbolo]
    fuera = [h for h in hechos if h.symbol not in por_simbolo]
    return dentro, fuera


# ------------------------------------------------------------------------ escritura


def _upsert_spine(conn, perfiles, dentro: list[ProviderFact]) -> dict[str, int]:
    por_simbolo = {p.symbol: aid for aid, p in perfiles.items()}
    escrito: dict[str, int] = {}

    with conn.cursor() as cur:
        for fila in calendar_rows(perfiles):
            cur.execute(
                "INSERT INTO reference.calendar "
                "(calendar_id, timezone, calendar_kind, session_definition, version) "
                "VALUES (%(calendar_id)s, %(timezone)s, %(calendar_kind)s, "
                "%(session_definition)s::jsonb, %(version)s) "
                "ON CONFLICT (calendar_id) DO UPDATE SET "
                "timezone = EXCLUDED.timezone, calendar_kind = EXCLUDED.calendar_kind, "
                "session_definition = EXCLUDED.session_definition, version = EXCLUDED.version",
                fila,
            )
        escrito["calendar"] = len(calendar_rows(perfiles))

        for fila in asset_rows(perfiles):
            cur.execute(
                "INSERT INTO reference.asset "
                "(asset_id, display_name, asset_class, quote_currency, annualization, calendar_id) "
                "VALUES (%(asset_id)s, %(display_name)s, %(asset_class)s, "
                "%(quote_currency)s, %(annualization)s, %(calendar_id)s) "
                "ON CONFLICT (asset_id) DO UPDATE SET "
                "display_name = EXCLUDED.display_name, asset_class = EXCLUDED.asset_class, "
                "quote_currency = EXCLUDED.quote_currency, "
                "annualization = EXCLUDED.annualization, calendar_id = EXCLUDED.calendar_id",
                fila,
            )
        escrito["asset"] = len(asset_rows(perfiles))

        for fila in instrument_rows(perfiles):
            cur.execute(
                "INSERT INTO reference.instrument "
                "(canonical_symbol, asset_id, instrument_type, base_currency, quote_currency) "
                "VALUES (%(canonical_symbol)s, %(asset_id)s, %(instrument_type)s, "
                "%(base_currency)s, %(quote_currency)s) "
                "ON CONFLICT (canonical_symbol) DO UPDATE SET "
                "asset_id = EXCLUDED.asset_id, instrument_type = EXCLUDED.instrument_type, "
                "base_currency = EXCLUDED.base_currency, quote_currency = EXCLUDED.quote_currency",
                fila,
            )
        escrito["instrument"] = len(instrument_rows(perfiles))

        # `authoritative_for` se MIDE: los activos para los que el proveedor escribió.
        autoridad: dict[str, set[str]] = {}
        for hecho in dentro:
            autoridad.setdefault(hecho.provider_id, set()).add(por_simbolo[hecho.symbol])
        for provider_id, activos in sorted(autoridad.items()):
            cur.execute(
                "INSERT INTO reference.provider (provider_id, display_name, authoritative_for) "
                "VALUES (%s, %s, %s) ON CONFLICT (provider_id) DO UPDATE SET "
                "authoritative_for = EXCLUDED.authoritative_for",
                (provider_id, provider_id, sorted(activos)),
            )
        escrito["provider"] = len(autoridad)

        cur.execute("SELECT canonical_symbol, instrument_id FROM reference.instrument")
        instrumento_por_simbolo = dict(cur.fetchall())

        vistos: set[tuple[str, str]] = set()
        for hecho in dentro:
            clave = (hecho.provider_id, hecho.symbol)
            if clave in vistos:
                continue  # el mismo par puede aparecer en m5 y daily
            vistos.add(clave)
            cur.execute(
                "INSERT INTO reference.provider_symbol "
                "(provider_id, provider_symbol, instrument_id, metadata) "
                "VALUES (%s, %s, %s, %s::jsonb) "
                "ON CONFLICT (provider_id, provider_symbol) DO UPDATE SET "
                "instrument_id = EXCLUDED.instrument_id, metadata = EXCLUDED.metadata",
                (
                    hecho.provider_id,
                    hecho.symbol,
                    instrumento_por_simbolo[hecho.symbol],
                    json.dumps({"evidence_table": hecho.table, "evidence_rows": hecho.rows}),
                ),
            )
        escrito["provider_symbol"] = len(vistos)

    return escrito


# ----------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--dry-run", action="store_true", help="imprime el plan, no escribe")
    grupo.add_argument("--apply", action="store_true", help="puebla la espina (idempotente)")
    args = parser.parse_args(argv)

    perfiles = declared_assets()
    print(f"Activos declarados con anualización válida: {sorted(perfiles)}")
    for fila in asset_rows(perfiles):
        print(
            f"  {fila['asset_id']:9s} ann={fila['annualization']:<5} "
            f"calendar={fila['calendar_id']}"
        )

    from scripts.data.ingest_asset_ohlcv import _db_conn

    conn = _db_conn()
    try:
        hechos = measure_provider_facts(conn)
        dentro, fuera = split_facts(hechos, perfiles)

        print(f"\nPares (proveedor, símbolo) medidos: {len(hechos)}")
        print(f"  mapeables a un activo declarado : {len(dentro)}")
        print(f"  FUERA por no estar declarados   : {len(fuera)}")
        for hecho in fuera:
            print(
                f"    - {hecho.provider_id}/{hecho.symbol} ({hecho.rows} filas en "
                f"{hecho.table}): sin AssetProfile, no entra"
            )

        if args.dry_run:
            print("\n--dry-run: nada escrito.")
            return 0

        escrito = _upsert_spine(conn, perfiles, dentro)
        conn.commit()
        print("\nEspina poblada:")
        for tabla, n in escrito.items():
            print(f"  reference.{tabla:17s} {n}")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
