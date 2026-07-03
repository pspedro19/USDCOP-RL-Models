"""DAG EXIT contract: register a strategy bundle into the dynamic registry.

Implements the "regla de salida" of CTR-STRAT-REGISTRY-001
(see .claude/rules/sdd-strategy-lifecycle-registry.md §6.2).

Every strategy pipeline's FINAL task calls `register_strategy_bundle(...)`. It (re)builds the
per-strategy manifest.json and the registry.json index from the artifacts already written to the
dashboard data dir. Only validated, discoverable bundles reach the registry — so the frontend
never renders a half-baked strategy.

Decoupling: this util loads the manifest contract (src/contracts/strategy_manifest.py) as a
standalone leaf module by file path, so it does NOT drag in the ML/forecasting stack (which the
`src` package __init__ eager-imports). Safe to import from any Airflow DAG.

Idempotent: re-running simply refreshes the registry from current artifacts.
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve()


def _find_manifest_path() -> Path:
    """Locate src/contracts/strategy_manifest.py across host and container layouts.

    Host:      <repo>/airflow/dags/utils/register_bundle.py -> <repo>/src/...
    Container: /opt/airflow/dags/utils/register_bundle.py   -> /opt/airflow/src/...
    """
    candidates = [
        _HERE.parents[3] / "src" / "contracts" / "strategy_manifest.py",  # host repo root
        Path("/opt/airflow/src/contracts/strategy_manifest.py"),          # airflow container
    ]
    # Also walk up looking for a `src/contracts/strategy_manifest.py` sibling.
    for parent in _HERE.parents:
        candidates.append(parent / "src" / "contracts" / "strategy_manifest.py")
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"strategy_manifest.py not found from {_HERE}")


def _load_manifest_module():
    """Load strategy_manifest.py as a leaf module (bypasses the src package __init__ chain)."""
    if "strategy_manifest" in sys.modules:
        return sys.modules["strategy_manifest"]
    spec = importlib.util.spec_from_file_location("strategy_manifest", _find_manifest_path())
    module = importlib.util.module_from_spec(spec)
    sys.modules["strategy_manifest"] = module  # required so @dataclass resolves annotations
    spec.loader.exec_module(module)
    return module


def _default_data_dir() -> Path:
    """Dashboard public/data dir, resolving both host and container (/opt/airflow) layouts."""
    candidates = [
        _HERE.parents[3] / "usdcop-trading-dashboard" / "public" / "data",  # host repo root
        Path("/opt/airflow/usdcop-trading-dashboard/public/data"),          # airflow container mount
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def register_strategy_bundle(data_dir: str | Path | None = None) -> dict[str, Any]:
    """Rebuild manifests + registry.json from current dashboard artifacts.

    Returns a summary dict (strategy count, asset count, default) for logging / XCom.
    """
    manifest = _load_manifest_module()
    data_path = Path(data_dir) if data_dir else _default_data_dir()
    if not data_path.exists():
        raise FileNotFoundError(f"[register_bundle] dashboard data dir not found: {data_path}")

    builder = manifest.RegistryBuilder(data_path, generated_at=datetime.now(timezone.utc).isoformat())
    index = builder.build(write_manifests=True)
    out = builder.write(index)

    summary = {
        "registry_path": str(out),
        "strategies": [s.strategy_id for s in index.strategies],
        "asset_count": len(index.assets),
        "default": index.default,
    }
    print(f"[register_bundle] published {len(index.strategies)} strateg(ies) -> {out}")
    return summary


if __name__ == "__main__":
    register_strategy_bundle()
