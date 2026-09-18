#!/usr/bin/env python
"""Aparta del ledger las barras en las que el proveedor NO respondio, para volver a pedirlas.

Una fila con `unavailable: true` (`APIConnectionError` / `APITimeoutError`) no contiene una
decision del modelo: contiene un hueco de red. El ledger es append-only y rechaza
`decision_id` repetidos, asi que mientras esas filas sigan ahi `--resume` las da por hechas y la
sesion queda perdida para siempre.

Este script las mueve a `<ledger>.unavailable.jsonl` (no las borra: la evidencia de la caida
tambien es evidencia) y reescribe el ledger sin ellas. El siguiente `--resume` las vuelve a
pedir de verdad.

Medido el 2026-09-12: una caida local tumbo 882 barras en DeepSeek y 882 en Azure **en el mismo
instante** -- 14 sesiones completas por brazo. Que los dos proveedores fallen a la vez es la
firma de un problema local, no de un proveedor.

**Se niega a tocar un ledger que alguien esta escribiendo.** Reescribir un fichero mientras el
proceso que lo genera tiene el descriptor abierto perderia las filas que escriba entre la
lectura y el reemplazo.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

QUIET_SECONDS = 120


def partition(ledger: Path) -> tuple[list[str], list[str], dict[str, int]]:
    keep: list[str] = []
    quarantine: list[str] = []
    per_session: dict[str, int] = {}
    with ledger.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("unavailable"):
                quarantine.append(line)
                date = str(row.get("session_date", "?"))
                per_session[date] = per_session.get(date, 0) + 1
            else:
                keep.append(line)
    return keep, quarantine, per_session


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", type=Path, required=True)
    ap.add_argument("--force", action="store_true",
                    help="salta la espera de inactividad (solo si SABES que nadie escribe)")
    args = ap.parse_args()

    if not args.ledger.is_file():
        print(f"no existe: {args.ledger}")
        return 2

    idle = time.time() - args.ledger.stat().st_mtime
    if idle < QUIET_SECONDS and not args.force:
        print(json.dumps({
            "refused": "ledger_activo",
            "seconds_since_last_write": round(idle, 1),
            "required_idle_seconds": QUIET_SECONDS,
            "why": ("el brazo sigue escribiendo; reescribir ahora perderia las filas que anada "
                    "entre la lectura y el reemplazo"),
        }, ensure_ascii=False))
        return 3

    keep, quarantine, per_session = partition(args.ledger)
    if not quarantine:
        print(json.dumps({"quarantined": 0, "kept": len(keep)}))
        return 0

    backup = args.ledger.with_suffix(args.ledger.suffix + ".before_quarantine")
    shutil.copy2(args.ledger, backup)
    dest = args.ledger.with_suffix(args.ledger.suffix + ".unavailable")
    with dest.open("a", encoding="utf-8") as handle:
        handle.writelines(quarantine)
    tmp = args.ledger.with_suffix(args.ledger.suffix + ".tmp")
    tmp.write_text("".join(keep), encoding="utf-8")
    tmp.replace(args.ledger)

    print(json.dumps({
        "quarantined": len(quarantine),
        "kept": len(keep),
        "sessions_touched": len(per_session),
        "fully_lost_sessions": sum(1 for n in per_session.values() if n == 59),
        "quarantine_file": str(dest),
        "backup": str(backup),
        "next": "vuelve a lanzar el brazo con --resume para pedir de nuevo esas barras",
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
