#!/usr/bin/env python3
"""
Export the GOVERNANCE projection the Control Tower reads (BL-32, CTR-PASSPORT-001).
==================================================================================

Why this script exists
----------------------
FABRIC §24.6: *"todos los archivos de frontend son proyecciones regenerables, jamás
fuente de verdad"*. The Control Tower needs three things that live OUTSIDE the
dashboard's ``public/`` tree and therefore are not reachable from a BFF route in a
container:

1. ``registries/ledger.jsonl``       — the machine form of the trial ledger (FT/AT, N×3)
2. ``registries/families/*.yaml``    — declared hypothesis families
3. ``.claude/specs/assets/**``       — ``HYPOTHESIS-REGISTRY`` front-matter (n_trials_total,
                                       the SSOT the DSR is deflated with) and the signed
                                       ``WITHDRAWAL-PROTOCOL`` files

This script projects them — **verbatim, no recomputation of any statistic** — into
``data/control-tower/governance.json``.

Why NOT ``public/`` (C-006 precedent, CODEX P0)
-----------------------------------------------
The projection carries **trials, DSR inputs and gate state: INTERNALS**. Anything under
``usdcop-trading-dashboard/public/`` is served by the ``/data/**`` static path, which the
edge middleware gates with *a session only* — so a ``free``/``subscriber`` could fetch it
directly and bypass the ``research:read`` the Passport requires (``rbac.md`` §"nada
monetizado anónimo" + §8 "subscribers ven OUTPUTS, no INTERNALS"). Same defect CODEX
rejected in C-006 for the SHAP artifacts, same remedy: the artifact lives OUTSIDE
``public/`` (like ``data/interpretability/``) and its only reader is the server-side
composer behind ``/api/passport/**``. Override the location with
``CONTROL_TOWER_DATA_DIR`` if the dashboard runs from another root.

What it is NOT
--------------
It does not compute a Sharpe, a DSR, a p-value or any performance figure. It COPIES
counts that other systems already published and COUNTS lines/cells. Performance keeps
coming from the published bundles, read live by the BFF composer. **0 trials.**

Run:  ``python scripts/pipeline/export_control_tower.py [--check]``
      ``--check`` recomputes and fails (exit 1) if the published file is stale.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.contracts.passport import (  # noqa: E402
    N_MAX_TRIALS,
    PASSPORT_CONTRACT_ID,
    PASSPORT_CONTRACT_VERSION,
)
from src.contracts.strategy_schema import safe_json_dumps  # noqa: E402

LEDGER = ROOT / "registries" / "ledger.jsonl"
FAMILIES_DIR = ROOT / "registries" / "families"
SPECS_ASSETS = ROOT / ".claude" / "specs" / "assets"
# FUERA de public/: la única vía de lectura es el composer server-side detrás de
# /api/passport/** (research:read). Ver el docstring — precedente C-006.
OUT = ROOT / "data" / "control-tower" / "governance.json"

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)


def _rel(path: Path) -> str:
    """Repo-root-relative POSIX path — the `source.path` every field carries."""
    return path.relative_to(ROOT).as_posix()


def _frontmatter(path: Path) -> dict:
    if not path.exists():
        return {}
    match = FRONTMATTER_RE.match(path.read_text(encoding="utf-8", errors="replace"))
    if not match:
        return {}
    return yaml.safe_load(match.group(1)) or {}


# ---------------------------------------------------------------------------
# Ledger (BL-09/10/11) — counts only, never a statistic
# ---------------------------------------------------------------------------


def read_ledger() -> dict:
    """Aggregate ``registries/ledger.jsonl`` per asset and per family.

    Every value here is a COUNT of ledger lines or a verbatim copy of the running
    counters the ledger itself wrote (``N_family``/``N_cluster``/``N_global``). The
    ledger is append-only and hash-chained; we only read it.
    """
    if not LEDGER.exists():
        return {"available": False, "path": None, "reason": "registries/ledger.jsonl no existe"}

    by_asset: dict[str, dict] = defaultdict(lambda: {"forecast": 0, "action": 0, "total": 0})
    by_family: dict[str, dict] = {}
    n_global_max = 0
    lines = 0

    with LEDGER.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            lines += 1
            asset = row.get("asset") or "unknown"
            kind = row.get("kind") or "unknown"
            bucket = by_asset[asset]
            bucket["total"] += 1
            if kind in ("forecast", "action"):
                bucket[kind] += 1
            for key in ("N_family", "N_cluster", "N_global"):
                value = row.get(key)
                if isinstance(value, int):
                    bucket[key.lower()] = max(bucket.get(key.lower(), 0), value)
            n_global = row.get("N_global")
            if isinstance(n_global, int):
                n_global_max = max(n_global_max, n_global)

            family = row.get("family")
            if family:
                fam = by_family.setdefault(family, {
                    "family_id": family, "kind": kind,
                    "cluster_id": row.get("cluster"), "n_trials": 0, "closed": False,
                })
                fam["n_trials"] += 1

    return {
        "available": True,
        "path": _rel(LEDGER),
        "lines": lines,
        "n_global": n_global_max,
        "by_asset": dict(by_asset),
        "by_family": by_family,
    }


def read_families(by_family: dict) -> list[dict]:
    """Merge declared families (`registries/families/*.yaml`) with ledger counts.

    A family declared but never charged still appears (n_trials from the ledger = 0):
    the whole point of §9.4 is that families are declared BEFORE looking.
    """
    families: list[dict] = []
    if not FAMILIES_DIR.is_dir():
        return families
    for path in sorted(FAMILIES_DIR.glob("*.yaml")):
        try:
            spec = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        family_id = spec.get("family_id") or path.stem
        ledger_row = by_family.get(family_id, {})
        families.append({
            "family_id": family_id,
            "kind": spec.get("kind"),
            "cluster_id": spec.get("cluster_id"),
            "n_trials": ledger_row.get("n_trials", 0),
            "n_cells_declared": len(spec.get("cells") or []),
            "closed": bool(spec.get("closed", False)),
            "closure_note": spec.get("closure_note"),
            "declared_at": str(spec.get("declared_at")) if spec.get("declared_at") else None,
            "path": _rel(path),
        })
    # Families present in the ledger but with no declaration file — surfaced, not hidden.
    declared = {f["family_id"] for f in families}
    for family_id, row in sorted(by_family.items()):
        if family_id not in declared:
            families.append({
                "family_id": family_id, "kind": row.get("kind"),
                "cluster_id": row.get("cluster_id"), "n_trials": row.get("n_trials", 0),
                "n_cells_declared": None, "closed": False,
                "closure_note": "SIN ARCHIVO DE FAMILIA DECLARADO (registries/families/)",
                "declared_at": None, "path": None,
            })
    return families


# ---------------------------------------------------------------------------
# Per-asset governance: HYPOTHESIS-REGISTRY + WITHDRAWAL-PROTOCOL
# ---------------------------------------------------------------------------

#: Where each asset's narrative SSOT lives. The registry front-matter is the
#: authoritative `n_trials_total` (the ledger is its machine form) — we copy it.
ASSET_DOCS: dict[str, dict[str, str]] = {
    "usdcop": {
        "hypothesis": ".claude/specs/assets/usdcop/HYPOTHESIS-REGISTRY.md",
        "withdrawal": ".claude/specs/assets/usdcop/WITHDRAWAL-PROTOCOL.md",
    },
    "xauusd": {
        "hypothesis": ".claude/specs/assets/xauusd/HYPOTHESIS-REGISTRY.md",
        "withdrawal": ".claude/specs/assets/xauusd/WITHDRAWAL-PROTOCOL-XAU.md",
    },
    "btcusdt": {
        "hypothesis": ".claude/specs/assets/btcusdt/design/HYPOTHESIS-REGISTRY.md",
        "withdrawal": ".claude/specs/assets/btcusdt/WITHDRAWAL-PROTOCOL-BTC.md",
    },
    "spx500": {
        "hypothesis": ".claude/specs/assets/spx500/HYPOTHESIS-REGISTRY.md",
        "withdrawal": ".claude/specs/assets/spx500/WITHDRAWAL-PROTOCOL-SPX.md",
    },
}


def read_assets(ledger: dict) -> dict:
    """Per-asset governance block: trials, sigma grid, withdrawal protocol presence."""
    by_asset = ledger.get("by_asset", {}) if ledger.get("available") else {}
    out: dict[str, dict] = {}
    for asset_id, docs in ASSET_DOCS.items():
        hyp_path = ROOT / docs["hypothesis"]
        wd_path = ROOT / docs["withdrawal"]
        fm = _frontmatter(hyp_path)
        counts = by_asset.get(asset_id, {})
        out[asset_id] = {
            # Narrative SSOT (authoritative for the DSR deflation).
            "n_trials_total": fm.get("n_trials_total"),
            "n_trials_scenarios": fm.get("n_trials_scenarios"),
            "sigma_trials": fm.get("sigma_trials"),
            "sigma_trials_grid": fm.get("sigma_trials_grid"),
            "hypothesis_registry": _rel(hyp_path) if hyp_path.exists() else None,
            # Machine form (ledger) — disclosed side by side so a divergence is visible.
            "ledger_forecast_trials": counts.get("forecast"),
            "ledger_action_trials": counts.get("action"),
            "ledger_total": counts.get("total"),
            "n_family": counts.get("n_family"),
            "n_cluster": counts.get("n_cluster"),
            "n_global": counts.get("n_global"),
            # Withdrawal protocol: PRESENCE only. Its thresholds are prose today —
            # a machine-readable protocol is BL-33/BL-25 territory, not this script's.
            "withdrawal_protocol": _rel(wd_path) if wd_path.exists() else None,
            "withdrawal_protocol_signed": wd_path.exists(),
        }
    return out


# ---------------------------------------------------------------------------
# Build + write
# ---------------------------------------------------------------------------


def build() -> dict:
    ledger = read_ledger()
    families = read_families(ledger.get("by_family", {}) if ledger.get("available") else {})
    return {
        "contract": PASSPORT_CONTRACT_ID,
        "contract_version": PASSPORT_CONTRACT_VERSION,
        "kind": "governance_projection",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "ledger": ledger.get("path"),
            "families_dir": _rel(FAMILIES_DIR) if FAMILIES_DIR.is_dir() else None,
        },
        "ledger_available": bool(ledger.get("available")),
        "n_global": ledger.get("n_global") if ledger.get("available") else None,
        # §9.7: SPEND CAP ONLY. It never enters the DSR — disclosed so the reader
        # sees how much of the budget has been burned, nothing else.
        "n_max_trials": N_MAX_TRIALS,
        "assets": read_assets(ledger),
        "families": families,
        "notes": [
            "Proyección regenerable (FABRIC §24.6). NO es fuente de verdad: lo son "
            "registries/ledger.jsonl y los HYPOTHESIS-REGISTRY por activo.",
            "0 trials: este script cuenta líneas y copia contadores ya publicados; "
            "no calcula ningún estadístico.",
            "N_MAX=989 es cota de GASTO (§9.7) y JAMÁS entra en el DSR.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="no escribe; falla si el artefacto publicado está desactualizado")
    args = parser.parse_args()

    payload = build()
    rendered = safe_json_dumps(payload, ensure_ascii=False)

    if args.check:
        if not OUT.exists():
            print(f"FAIL: {_rel(OUT)} no existe — corre el exportador", file=sys.stderr)
            return 1
        current = json.loads(OUT.read_text(encoding="utf-8"))
        fresh = json.loads(rendered)
        # generated_at always differs; compare everything else.
        current.pop("generated_at", None)
        fresh.pop("generated_at", None)
        if current != fresh:
            print(f"FAIL: {_rel(OUT)} está desactualizado — corre el exportador", file=sys.stderr)
            return 1
        print(f"OK: {_rel(OUT)} al día")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(rendered + "\n", encoding="utf-8")
    assets = payload["assets"]
    print(f"escrito {_rel(OUT)}")
    print(f"  ledger: {'disponible' if payload['ledger_available'] else 'AUSENTE'} "
          f"· N_global={payload['n_global']} / N_MAX={payload['n_max_trials']}")
    print(f"  activos: {len(assets)} · familias: {len(payload['families'])}")
    for asset_id, block in assets.items():
        print(f"    {asset_id}: n_trials_total={block['n_trials_total']} "
              f"ledger={block['ledger_total']} "
              f"retiro={'firmado' if block['withdrawal_protocol_signed'] else 'AUSENTE'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
