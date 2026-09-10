#!/usr/bin/env python
"""Inventario de fuentes: la compuerta del brazo LLM, antes de gastar un token.

Contract: CTR-RESEARCH-FORWARD-001 · Date: 2026-08-25

## Qué decide

El brazo LLM lee prensa publicada **antes de las 08:00 COT**. Si las fuentes colombianas no
publican material accionable a esa hora, el brazo forward tiene el mismo problema que el
retrospectivo —nada que leer— solo que descubierto tres meses más tarde y habiendo pagado
tokens por el camino.

Este script mide exactamente eso: cuántos documentos hay dentro de la ventana, por fuente y por
día. Se corre **cinco días hábiles seguidos** antes de sellar la primera decisión real.

**Regla de cierre, escrita antes de mirar el resultado**: si la mediana de documentos
pre-apertura queda por debajo de `min_docs`, el brazo se cierra y se escribe. Descubrirlo el día
cinco cuesta cinco días; descubrirlo en el mes tres cuesta la ventana entera.

## Lo que ya se sabe

Sondeo del 2026-08-25 contra las tres URL que traía el arnés:

    banrep.gov.co/rss/noticias.xml   -> HTTP 404
    portafolio.co/rss/economia       -> HTTP 404
    larepublica.co/rss/finanzas      -> HTTP 200, 60 items, 6 antes de las 08:00 COT

Dos de tres eran placeholders muertos. Por eso este inventario existe y por eso `curl_cffi`
sustituye al `urllib` de `feedparser`: varios sitios colombianos devuelven 403 a un
`User-Agent` de librería, y un feed vivo que parece muerto es peor que uno muerto.

Uso:
    python scripts/diagnostics/forward_source_inventory.py
    python scripts/diagnostics/forward_source_inventory.py --append   # acumula el dia
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import yaml  # noqa: E402

COT = timezone(timedelta(hours=-5))
PREREG = REPO / "config" / "research" / "preregistration_forward.yaml"
OUT = REPO / "data" / "forward" / "source_inventory.jsonl"

# Candidatos mas alla de los del pre-registro. Se prueban para poder sustituir las URL
# muertas con algo OBSERVADO, no con otra suposicion.
EXTRA_CANDIDATES = {
    "larepublica_economia": "https://www.larepublica.co/rss/economia",
    "larepublica_globoeconomia": "https://www.larepublica.co/rss/globoeconomia",
    "portafolio_home": "https://www.portafolio.co/rss",
    "banrep_home": "https://www.banrep.gov.co/es/rss.xml",
    "eltiempo_economia": "https://www.eltiempo.com/rss/economia.xml",
}


def probe(name: str, url: str, cutoff_hour: int = 8) -> dict:
    """Mide una fuente: alcanzable, con fechas, y cuántos ítems caen pre-apertura."""
    import feedparser

    row = {"name": name, "url": url, "http": None, "items": 0, "dated": 0,
           "pre_open": 0, "newest_cot": None, "error": None}
    try:
        from curl_cffi import requests

        resp = requests.get(url, impersonate="chrome", timeout=25)
        row["http"] = resp.status_code
        if resp.status_code != 200:
            return row
        feed = feedparser.parse(resp.content)
    except Exception as exc:                       # una fuente caida no aborta el inventario
        row["error"] = f"{type(exc).__name__}: {exc}"[:120]
        return row

    row["items"] = len(feed.entries)
    stamps = []
    for entry in feed.entries:
        t = entry.get("published_parsed") or entry.get("updated_parsed")
        if t is None:
            continue
        stamps.append(datetime(*t[:6], tzinfo=timezone.utc).astimezone(COT))

    row["dated"] = len(stamps)
    row["pre_open"] = sum(1 for s in stamps if s.hour < cutoff_hour)
    row["newest_cot"] = max(stamps).isoformat(timespec="minutes") if stamps else None
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--append", action="store_true",
                    help="acumula la observacion de hoy en el registro de 5 dias")
    ap.add_argument("--all-candidates", action="store_true",
                    help="prueba tambien las fuentes candidatas, no solo las del pre-registro")
    args = ap.parse_args()

    spec = yaml.safe_load(PREREG.read_text(encoding="utf-8"))
    cutoff_hour = spec["session"]["cutoff_hour_cot"]
    min_docs = spec["corpus"]["min_docs"]

    sources = {s["name"]: s["url"] for s in spec["corpus"]["sources"]}
    if args.all_candidates:
        sources.update(EXTRA_CANDIDATES)

    print(f"cutoff {cutoff_hour}:00 COT · min_docs {min_docs} · {len(sources)} fuente(s)\n")
    print(f"{'fuente':<28} {'HTTP':>5} {'items':>6} {'con fecha':>10} "
          f"{'PRE-APERTURA':>13}  mas reciente (COT)")

    rows, total_pre = [], 0
    for name, url in sources.items():
        row = probe(name, url, cutoff_hour)
        rows.append(row)
        total_pre += row["pre_open"]
        flag = "" if row["http"] == 200 else "  <- inalcanzable"
        print(f"{name:<28} {str(row['http'] or '-'):>5} {row['items']:>6} "
              f"{row['dated']:>10} {row['pre_open']:>13}  "
              f"{row['newest_cot'] or '-'}{flag}")
        if row["error"]:
            print(f"{'':<28} {row['error']}")

    print(f"\nTOTAL pre-apertura hoy: {total_pre} documento(s)")
    if total_pre < min_docs:
        print(f"POR DEBAJO de min_docs={min_docs}: hoy el brazo se ABSTENDRIA "
              "(no puntuaria 0, que es distinto).")
    else:
        print(f"Por encima de min_docs={min_docs}: hoy el brazo tendria material.")

    if args.append:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        entry = {"observed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "cutoff_hour_cot": cutoff_hour, "total_pre_open": total_pre,
                 "sources": rows}
        with OUT.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        days = sum(1 for _ in OUT.open(encoding="utf-8"))
        print(f"\nregistrado -> {OUT.relative_to(REPO)} ({days} observacion/es de 5)")
        if days >= 5:
            import statistics

            totals = [json.loads(x)["total_pre_open"]
                      for x in OUT.read_text(encoding="utf-8").splitlines()]
            med = statistics.median(totals)
            print(f"MEDIANA de {len(totals)} dias: {med:.1f} documentos pre-apertura")
            print("VEREDICTO:", "el brazo puede arrancar" if med >= min_docs
                  else "CERRAR el brazo — la regla se escribio antes de mirar")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
