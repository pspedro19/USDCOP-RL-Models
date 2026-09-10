"""End-to-end smoke run with no API key and no network.

Stubs the two external dependencies (the RSS sources and the model) and drives
the real decide -> settle -> verify path over five synthetic sessions. Run this
first: it proves the wiring before you spend a token, and it is the fastest way
to see what the ledger actually looks like.

    python scripts/smoke.py
"""

from __future__ import annotations

import random
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.research.llm_forward import corpus, decide, llm  # noqa: E402
from src.research.llm_forward.corpus import RawDoc  # noqa: E402
from src.research.llm_forward.ledger import Ledger  # noqa: E402
from src.research.llm_forward.schema import Decision, LlmUsage  # noqa: E402

SESSIONS = [
    "2026-08-17", "2026-08-18", "2026-08-19", "2026-08-20", "2026-08-21",
]

HEADLINES = [
    ("BanRep mantiene la tasa de intervencion en 9,25%",
     "La junta directiva decidio por mayoria mantener la tasa sin cambios."),
    ("Brent cae 2% por expectativas de mayor oferta de la OPEP+",
     "El crudo de referencia retrocede en la apertura europea."),
    ("Exportaciones colombianas crecen 4,1% interanual en junio",
     "El DANE reporto un aumento impulsado por el sector minero-energetico."),
]


# The session currently being scored. A module-level cursor is the least
# invasive way to make the stub session-aware without changing the real
# fetch_rss signature that production code depends on.
_CURRENT_SESSION: str = SESSIONS[0]


def fake_rss(feed_url: str, timeout: int = 20) -> list[RawDoc]:
    """Two documents per session, stamped a few hours before that session's cutoff."""
    cutoff = corpus.session_cutoff_utc(_CURRENT_SESSION, 8)
    return [
        RawDoc(
            url=f"{feed_url}/{index}",
            title=title,
            text=body,
            published_at=cutoff - timedelta(hours=index + 2),
        )
        for index, (title, body) in enumerate(HEADLINES[:2])
    ]


class FakeClient:
    """Stands in for LlmClient with the same surface."""

    provider = "stub"
    model = "stub-model"
    temperature = 0.0
    seed = 20260101

    def __init__(self, *args, **kwargs) -> None:
        pass

    def decide(self, system_prompt: str, user_prompt: str):
        score = round(random.uniform(-0.6, 0.6), 3)
        return llm.LlmResult(
            decision=Decision(
                score=score,
                direction="long" if score > 0.1 else "short" if score < -0.1 else "flat",
                confidence=round(abs(score), 3),
                rationale="Stubbed decision for the smoke run.",
            ),
            usage=LlmUsage(1450, 60, 820, 0.000253, "fp_stub"),
            raw_response={"model": "stub-model"},
        )


def write_prices(path: Path) -> None:
    """59 five-minute bars per session, a random walk around 4100."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = ["timestamp_utc,close"]
    for session in SESSIONS:
        start = datetime.fromisoformat(f"{session}T13:00:00+00:00")
        price = 4100.0
        for bar in range(59):
            price *= 1 + random.gauss(0, 0.0004)
            rows.append(f"{(start + timedelta(minutes=5 * bar)).isoformat()},{price:.4f}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    random.seed(7)

    # Las rutas salen de `paths.py`, no se reconstruyen aqui: el smoke tiene que limpiar
    # y auditar EXACTAMENTE los ficheros que escribe el codigo real. Reconstruirlas a mano
    # fue lo que hizo que la demo de manipulacion mirase un fichero inexistente y el smoke
    # terminara en verde sin haber comprobado nada.
    from src.research.llm_forward.paths import CORPUS_STORE, FWD_DATA, LEDGER_DIR
    for directory in (LEDGER_DIR, CORPUS_STORE, FWD_DATA / "prices"):
        shutil.rmtree(directory, ignore_errors=True)

    corpus.fetch_rss = fake_rss
    decide.fetch_rss = fake_rss
    decide.LlmClient = FakeClient

    print("=== JOB 1: sealing decisions ===")
    global _CURRENT_SESSION
    for session in SESSIONS:
        _CURRENT_SESSION = session
        # Pretend we are running at 07:15 COT on that session's morning.
        pretend_now = corpus.session_cutoff_utc(session, 8) - timedelta(minutes=45)
        print(f"{session}:")
        decide.run(session, now_override=pretend_now)

    price_csv = FWD_DATA / "prices" / "usdcop_5m.csv"
    write_prices(price_csv)

    print("\n=== JOB 2: settling ===")
    from src.research.llm_forward import settle
    settle.run(price_csv)

    print("\n=== AUDIT ===")
    from src.research.llm_forward import verify
    verify.main()

    print("\n=== TAMPER DEMO ===")
    path = LEDGER_DIR / "decisions.jsonl"
    lines = path.read_text().splitlines()
    before = lines[2]
    lines[2] = before.replace('"confidence":', '"confidence": 0.99, "_was":')
    assert lines[2] != before, "tamper demo must actually modify the line"
    path.write_text("\n".join(lines) + "\n")

    ok, reason = Ledger(path, "decision_id").verify()
    print(f"after editing one score -> verify says: {'OK' if ok else 'FAIL'}")
    print(f"  {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
