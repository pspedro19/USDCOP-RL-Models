"""Prompt construction and prompt versioning.

The prompt is part of the pre-registration, not an implementation detail. If you
tweak wording mid-run you have silently changed the treatment, and every session
before the tweak belongs to a different experiment. Hashing the template makes
that visible: the hash lands in every record, and a change shows up as a break in
the series rather than as nothing at all.

Practical consequence: edit ``SYSTEM_TEMPLATE`` and you must start a new arm.
"""

from __future__ import annotations

from .canonical import sha256_text
from .schema import CorpusDoc

SYSTEM_TEMPLATE = """\
You are a sell-side FX analyst covering USD/COP. You will be shown news items \
published before the session opens. Emit one directional score for the coming \
Bogota session (08:00-12:55 COT).

Rules:
- Score the USD/COP pair, not the dollar or the peso in isolation. Positive means \
you expect USD to strengthen against COP during the session.
- Judge only from the documents provided. You have no price history, no chart, \
and no knowledge of what happened after the documents were published.
- If the documents contain nothing material for USD/COP, return score 0.0, \
direction "flat", and say so. A neutral call is a valid answer and is scored as \
such. Do not manufacture a view to seem useful.
- Cite the doc_id values that drove your score in the rationale.
"""

USER_TEMPLATE = """\
Session to score: {session_date} (Bogota, 08:00-12:55 COT)
Information cutoff: {cutoff_utc} UTC
Documents available: {doc_count}

{documents}
"""

DOC_TEMPLATE = """\
--- doc_id: {doc_id} | published: {published_at_utc} ---
{title}

{body}
"""


def build_system_prompt() -> str:
    return SYSTEM_TEMPLATE


def build_user_prompt(
    session_date: str,
    cutoff_utc: str,
    docs: list[CorpusDoc],
    bodies: dict[str, str],
    max_chars_per_doc: int = 2000,
) -> str:
    """Render the user turn.

    Args:
        bodies: doc_id -> full text, read back from the content-addressed store.
        max_chars_per_doc: truncation budget. Truncation is applied uniformly and
            recorded via ``char_count`` in the ledger, so a reader can tell how
            much of each document actually reached the model.
    """
    if not docs:
        rendered = "(no documents published before the cutoff)"
    else:
        rendered = "\n".join(
            DOC_TEMPLATE.format(
                doc_id=doc.doc_id,
                published_at_utc=doc.published_at_utc,
                title=doc.title,
                body=bodies[doc.doc_id][:max_chars_per_doc],
            )
            for doc in docs
        )

    return USER_TEMPLATE.format(
        session_date=session_date,
        cutoff_utc=cutoff_utc,
        doc_count=len(docs),
        documents=rendered,
    )


def prompt_hash() -> str:
    """Hash of the frozen templates. Changes if any template text changes."""
    return sha256_text(SYSTEM_TEMPLATE + USER_TEMPLATE + DOC_TEMPLATE)
