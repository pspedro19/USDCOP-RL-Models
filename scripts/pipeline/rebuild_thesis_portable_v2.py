#!/usr/bin/env python
"""Rebuild the research portable dataset only after all v2 gates pass."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--macro", type=Path, required=True, help="macro parquet whose identity was reconciled"
    )
    parser.add_argument("--macro-identity", type=Path, required=True)
    parser.add_argument(
        "--sanity",
        type=Path,
        default=ROOT / "outputs" / "thesis-repair" / "sanity_protocol_v2.json",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data" / "thesis" / "research_data_portable_v2.pkl"
    )
    parser.add_argument(
        "--diagnostic-output",
        type=Path,
        help="optional retrospective copy written with the same v2 identity",
    )
    parser.add_argument(
        "--export-config",
        action="store_true",
        help="export the frozen scaler and portable HMM used by live_spec",
    )
    parser.add_argument(
        "--partition", type=Path, help="versioned partition config; sets THESIS_PARTITION_CONFIG"
    )
    parser.add_argument("--scaler-output", type=Path, help="optional separate frozen scaler output")
    parser.add_argument("--regime-output", type=Path, help="optional separate portable HMM output")
    args = parser.parse_args()
    if args.partition:
        os.environ["THESIS_PARTITION_CONFIG"] = str(args.partition.resolve())
    macro = args.macro.resolve()
    availability = ROOT / "config" / "research" / "macro_availability.yaml"
    try:
        from src.research.sanity_gate import require_macro_identity, require_sanity_pass

        require_sanity_pass(args.sanity.resolve())
        require_macro_identity(
            args.macro_identity.resolve(), availability=availability, clean=macro
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"v2_rebuild_blocked: {exc}", file=sys.stderr)
        return 2
    output = args.output.resolve()
    scaler_output = args.scaler_output.resolve() if args.scaler_output else None
    regime_output = args.regime_output.resolve() if args.regime_output else None
    # Set all lane inputs before importing research modules: module-level paths are
    # part of dataset identity and must point to the verified v4 artifacts.
    os.environ["THESIS_MACRO_CLEAN"] = str(macro)
    # Keep identity computation and subsequent consumers on the exact v4 artifacts;
    # never overwrite or silently validate against the historical canonical files.
    if scaler_output:
        os.environ["THESIS_SCALER_CONFIG"] = str(scaler_output)
    if regime_output:
        os.environ["THESIS_REGIME_CONFIG"] = str(regime_output)
    # The active observation contract has four posterior slots.  Apply the
    # predeclared v4 K universe only in this builder process; no truncation is
    # permitted and the sanity recipe remains unchanged.
    import src.research.dataset as dataset_module
    from src.research.dataset import load_or_build, save_portable
    _fit_frozen = dataset_module.fit_frozen
    dataset_module.fit_frozen = lambda obs: _fit_frozen(obs, k_candidates=(2, 3, 4))

    data = load_or_build(rebuild=True, verbose=True)
    if scaler_output or regime_output:
        # Materialize the v4 numeric artifacts from this exact build before the
        # portable identity is captured.  This avoids a missing-file race and keeps
        # v1/v2 frozen files untouched.
        from src.research.live_spec import export_scaler
        from src.research.regime_portable import export_from_frozen

        if scaler_output:
            scaler_output.parent.mkdir(parents=True, exist_ok=True)
            export_scaler(data, scaler_output)
    if regime_output:
        regime_output.parent.mkdir(parents=True, exist_ok=True)
        export_from_frozen(data.regime_model, regime_output)
    # Export helpers bind the identity before both lane files exist.  Rebind the
    # back-reference after materialization (identity calculation excludes this
    # field, so no circular hash is introduced).
    if scaler_output or regime_output:
        import json

        from src.research.dataset import dataset_identity

        identity = dataset_identity()
        for artifact in (scaler_output, regime_output):
            if artifact:
                payload = json.loads(artifact.read_text(encoding="utf-8"))
                payload["dataset_identity"] = identity
                artifact.write_text(
                    json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
                )
    save_portable(data, output, scaler_path=scaler_output, regime_path=regime_output)
    if args.diagnostic_output:
        save_portable(
            data,
            args.diagnostic_output.resolve(),
            scaler_path=scaler_output,
            regime_path=regime_output,
        )
    if args.export_config:
        from src.research.live_spec import export_scaler
        from src.research.regime_portable import export_from_frozen

        export_scaler(data, scaler_output) if scaler_output else export_scaler(data)
        export_from_frozen(
            data.regime_model, regime_output
        ) if regime_output else export_from_frozen(data.regime_model)
        print("v2 frozen scaler and portable HMM exported")
    print(f"v2 portable written: {output} ({data.summary()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
