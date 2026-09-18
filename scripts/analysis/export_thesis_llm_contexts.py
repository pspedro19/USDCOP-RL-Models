#!/usr/bin/env python
"""Export causal LLM contexts from the frozen research portable dataset.

The exporter never calls a model.  It serializes only observations available at the
decision bar (the last 24 bars up to and including ``bar``), daily T-1 macro/regime
context, and a frozen prompt.  Hold-out export is blocked unless explicitly labelled
retrospective, so producing contexts cannot silently become a confirmatory trial.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.dataset import PORTABLE, load_portable  # noqa: E402
from src.research.features import FEATURE_ORDER, GROUPS  # noqa: E402

SYSTEM_PROMPT = """Eres un analista cuantitativo de USD/COP. Responde exclusivamente JSON con las claves:
direccion (short, flat o long), tamano (0, 0.5 o 1) y confianza (0 a 1).
Usa únicamente el contexto observado hasta la barra indicada; no inventes noticias ni datos futuros.
Si no hay una ventaja clara, responde flat con tamano 0."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prompt(session_date: str, bar: int, market: np.ndarray, context: np.ndarray) -> str:
    names = list(GROUPS["precio"] + GROUPS["volatilidad"] + GROUPS["tendencia"] + GROUPS["temporal"])
    rows = []
    start = max(0, bar - 23)
    for offset, vector in enumerate(market[start:bar + 1], start=start):
        values = ", ".join(f"{name}={float(value):.8g}" for name, value in zip(names, vector, strict=False))
        rows.append(f"barra={offset}: {values}")
    context_names = list(GROUPS["macro"] + GROUPS["regimen"])
    ctx = ", ".join(f"{name}={float(value):.8g}" for name, value in zip(context_names, context, strict=False))
    return (f"Sesión {session_date}, barra de decisión {bar}/58.\n"
            f"Contexto diario causal (disponible antes de la sesión): {ctx}\n"
            "No hay documentos de noticias en este export; no asumas ninguno.\n"
            "Observaciones de mercado hasta ahora:\n" + "\n".join(rows))


def _forward_sessions(specs_path: Path) -> tuple[list[object], str]:
    """Load the already-built post-freeze specs without rebuilding or refitting."""
    with specs_path.open("rb") as handle:
        bundle = pickle.load(handle)
    manifest = bundle.get("manifest", {})
    sessions = bundle.get("sessions", [])
    identity = manifest.get("portable_sha256")
    if not isinstance(identity, str) or len(identity) != 64 or len(sessions) == 0:
        raise ValueError("invalid forward specs manifest")
    return sessions, identity


def export_contexts(block: str, output: Path, *, portable: Path = PORTABLE,
                    allow_retrospective: bool = False,
                    retrospective: bool = False,
                    forward_specs: Path | None = None) -> int:
    if block == "holdout" and not allow_retrospective:
        raise ValueError("holdout bloqueado: usa --allow-retrospective y etiqueta el replay")
    if block == "forward":
        if forward_specs is None:
            raise ValueError("forward export requires --forward-specs")
        sessions, identity = _forward_sessions(forward_specs)
    else:
        data = load_portable(portable)
        sessions = data.block(block)
        identity = _sha256(portable)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for session in sessions:
            date_text = session.date.isoformat()
            market = np.asarray(session.market, dtype=float)
            context = np.asarray(session.context, dtype=float)
            if market.ndim != 2 or len(market) != 60:
                raise ValueError(f"{date_text}: market no tiene 60 barras")
            for bar in range(59):
                row = {
                    "session_date": date_text,
                    "bar": bar,
                    "previous_weight": 0.0,
                    "system_prompt": SYSTEM_PROMPT,
                    "user_prompt": _prompt(date_text, bar, market, context),
                    "dataset_sha256": identity,
                    "dataset_block": block,
                    "feature_order": list(FEATURE_ORDER),
                    "news_count": 0,
                    "retrospective": bool(retrospective or block == "holdout"),
                }
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", choices=("development", "selection", "holdout", "forward"), default="selection")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--portable", type=Path, default=PORTABLE)
    parser.add_argument("--allow-retrospective", action="store_true")
    parser.add_argument("--retrospective", action="store_true",
                        help="label this block as already observed; runner will require an explicit override")
    parser.add_argument("--forward-specs", type=Path,
                        help="post-freeze specs pickle required when --block forward is selected")
    args = parser.parse_args()
    try:
        count = export_contexts(args.block, args.output, portable=args.portable,
                                allow_retrospective=args.allow_retrospective,
                                retrospective=args.retrospective,
                                forward_specs=args.forward_specs)
    except (OSError, ValueError) as exc:
        print(f"context_export_error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"contexts": count, "block": args.block,
                      "output": str(args.output), "network_called": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
