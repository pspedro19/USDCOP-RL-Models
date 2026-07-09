#!/usr/bin/env python3
"""Verify an exchange API key is TRADE-ONLY (no withdraw) — run after rotating keys.

Anti-withdraw gate (rbac.md §5 / SFC-GATE-CHECKLIST §D): the platform must never hold a
key that can move funds. This probes the exchange's account/permission surface and gives
a three-way verdict per key:

  SAFE      — exchange proves the key CANNOT withdraw (ok to mark 'verified')
  DANGEROUS — exchange reports withdraw ENABLED (reject / rotate immediately)
  UNKNOWN   — cannot prove either way (keep 'pending'; fail-closed, do not trade)

Usage (keys via env, never argv — argv leaks into shell history):
  MEXC_API_KEY=... MEXC_API_SECRET=... python -m scripts.validation.verify_key_permissions --exchange mexc
  BINANCE_API_KEY=... BINANCE_API_SECRET=... python -m scripts.validation.verify_key_permissions --exchange binance

Read-only: only auth'd GET endpoints; never places orders, never withdraws.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys


async def check(exchange_id: str, api_key: str, secret: str) -> tuple[str, str]:
    import ccxt.async_support as ccxt

    ex = getattr(ccxt, exchange_id)({
        "apiKey": api_key, "secret": secret,
        "options": {"defaultType": "spot"},  # SOLO SPOT
        "enableRateLimit": True,
    })
    try:
        # 1. Credentials valid at all?
        await ex.fetch_balance()

        # 2. Permission probes (exchange-specific; fail CLOSED on ambiguity).
        if exchange_id == "binance":
            # GET /sapi/v1/account/apiRestrictions — authoritative permission map.
            r = await ex.sapi_get_account_apirestrictions()
            wd = r.get("enableWithdrawals")
            trade = r.get("enableSpotAndMarginTrading")
            detail = f"withdrawals={wd} spotTrade={trade} ipRestrict={r.get('ipRestrict')}"
            if str(wd).lower() == "true":
                return "DANGEROUS", detail
            if str(wd).lower() == "false":
                return "SAFE", detail
            return "UNKNOWN", detail
        if exchange_id == "mexc":
            # MEXC spot: GET /api/v3/account carries `permissions`; withdrawals ride a
            # separate wallet scope not present when the key is trade-only.
            acct = await ex.private_get_account()
            perms = acct.get("permissions") or []
            detail = f"permissions={perms}"
            lowered = {str(p).lower() for p in perms}
            if any("withdraw" in p for p in lowered):
                return "DANGEROUS", detail
            if lowered and lowered <= {"spot"}:
                return "SAFE", detail
            return "UNKNOWN", detail
        return "UNKNOWN", f"no permission probe implemented for {exchange_id}"
    finally:
        await ex.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", required=True, choices=["mexc", "binance"])
    a = ap.parse_args()
    key = os.environ.get(f"{a.exchange.upper()}_API_KEY", "")
    sec = os.environ.get(f"{a.exchange.upper()}_API_SECRET", "")
    if not key or not sec:
        print(f"set {a.exchange.upper()}_API_KEY / _API_SECRET env vars (never argv)")
        return 2
    verdict, detail = asyncio.run(check(a.exchange, key, sec))
    print(f"[{a.exchange}] verdict={verdict}  {detail}")
    print({"SAFE": "-> ok para marcar 'verified' en user_exchange_keys",
           "DANGEROUS": "-> RECHAZAR y ROTAR YA (la llave puede mover fondos)",
           "UNKNOWN": "-> mantener 'pending' (fail-closed): restringe permisos en el exchange y reintenta"}[verdict])
    return 0 if verdict == "SAFE" else 1


if __name__ == "__main__":
    sys.exit(main())
