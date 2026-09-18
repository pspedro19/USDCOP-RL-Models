#!/usr/bin/env python
"""Offline validation/freeze and bounded prospective LLM pilot entrypoint.

Default and --validate-only NEVER read credentials, create a ledger or call APIs.
--freeze writes a new immutable manifest; --execute requires that manifest plus
live timestamped contexts. Environment secrets are consumed only by transport.
No .env loading, historical replay, implicit fallbacks, or overwrite options.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.llm_experiment_v2 import (  # noqa: E402
    PROVIDERS,
    VARIANTS,
    ExplicitChatTransport,
    PilotBlocked,
    PilotRunner,
    PilotStore,
    canonical,
    cohort_status,
    freeze_file,
    prepare_manifest,
    safe_path,
    utc_now,
    validate_context,
    verify_manifest,
)


def read_json(path: Path):
    return json.loads(safe_path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/research/llm_pilot_v2.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--calendar", type=Path)
    parser.add_argument("--dictionary", type=Path)
    parser.add_argument("--artifact", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--contexts", type=Path)
    parser.add_argument("--store", type=Path, default=ROOT / "data/thesis/llm/pilot_v2_shared.sqlite")
    parser.add_argument("--sidecars", type=Path, default=ROOT / "data/thesis/llm/pilot_v2_responses")
    args = parser.parse_args(argv)
    try:
        for path in (args.config, args.manifest, args.calendar, args.dictionary,
                     args.contexts, args.store, args.sidecars):
            if path is not None:
                safe_path(path)
        if args.manifest and args.manifest.is_file() and not args.freeze:
            manifest = read_json(args.manifest)
            verify_manifest(manifest)
        else:
            config = yaml.safe_load(safe_path(args.config).read_text(encoding="utf-8"))
            if not args.calendar or not args.dictionary:
                raise PilotBlocked("calendar, scale dictionary, current pricing and immutable freeze required; no API called")
            artifacts = {}
            for value in args.artifact:
                name, separator, path = value.partition("=")
                if not separator or not name or name in artifacts:
                    raise PilotBlocked("artifact must be a unique NAME=PATH")
                artifacts[name] = Path(path)
            manifest = prepare_manifest(config, read_json(args.calendar), read_json(args.dictionary),
                                        artifacts, now=utc_now())
        if args.freeze:
            if not args.manifest:
                raise PilotBlocked("--manifest destination required for freeze")
            freeze_file(args.manifest, manifest)
            print(canonical({"status": "FROZEN_NOT_EXECUTED", "manifest_sha256": manifest["manifest_sha256"],
                             "network_called": False, "reserve_usd": manifest["allocation_micro_usd"] / 1e6}))
            return 0
        if not args.execute:
            validated = 0
            if args.contexts:
                with args.contexts.open(encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            validate_context(json.loads(line), manifest, utc_now())
                            validated += 1
            print(canonical({"status": "VALIDATED_NOT_EXECUTED", "network_called": False,
                             "contexts_validated": validated, "confirmatory": False}))
            return 0
        if not args.manifest or not args.manifest.is_file() or not args.contexts:
            raise PilotBlocked("live execution requires an existing immutable manifest and incoming contexts")
        if args.store.resolve() != (ROOT / "data/thesis/llm/pilot_v2_shared.sqlite").resolve():
            raise PilotBlocked("live mode must use the single workspace-wide shared budget ledger")
        store = PilotStore(args.store)
        runner = PilotRunner(manifest, store, args.sidecars, ExplicitChatTransport())
        # A JSONL input is a transport, not a historical dataset: every row must
        # still be inside its live decision deadline when dispatched.
        with args.contexts.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                context = json.loads(line)
                for provider in PROVIDERS:
                    for variant in VARIANTS:
                        runner.decide(context, provider, variant)
        report = cohort_status(manifest, store.records(manifest))
        print(canonical(report))
        return 0 if report["status"] == "COMPLETE_PILOT" else 2
    except (PilotBlocked, OSError, ValueError, KeyError, TypeError) as exc:
        # Configuration errors have controlled messages. Other exception strings
        # are deliberately suppressed; HTTP errors can embed private material.
        reason = str(exc) if isinstance(exc, PilotBlocked) else type(exc).__name__
        print(canonical({"status": "BLOCKED", "reason": reason,
                         "no_confirmatory_claim": True}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
