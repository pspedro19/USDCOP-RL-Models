"""Corpus assembly with a hard information cutoff.

This module is the single point where look-ahead can enter the system, so it is
written defensively. Every document must carry a publication timestamp, and any
document at or after the cutoff is dropped — not warned about, dropped.

The asymmetry is deliberate. A document wrongly excluded costs a little signal.
A document wrongly included silently invalidates every result downstream and is
undetectable in the output. Those two errors are not worth trading off evenly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .canonical import sha256_text
from .schema import CorpusDoc

# Colombia does not observe DST, so a fixed offset is correct here and a tz
# database lookup would add a dependency for no gain. This assumption is worth
# an explicit constant so it is easy to find if it ever stops holding.
COT = timezone(timedelta(hours=-5))


@dataclass(frozen=True)
class RawDoc:
    """A document as it comes out of a source adapter, before filtering."""

    url: str
    title: str
    text: str
    published_at: datetime  # MUST be timezone-aware


class CutoffViolation(ValueError):
    """Raised when a document cannot be safely placed relative to the cutoff."""


def session_cutoff_utc(session_date: str, cutoff_hour_cot: int = 8) -> datetime:
    """Return the information cutoff for a session, in UTC.

    Args:
        session_date: ``YYYY-MM-DD`` of the session being predicted.
        cutoff_hour_cot: local hour after which nothing may be read.
    """
    local_date = datetime.strptime(session_date, "%Y-%m-%d").date()
    cutoff_local = datetime.combine(
        local_date, datetime.min.time(), tzinfo=COT
    ).replace(hour=cutoff_hour_cot)
    return cutoff_local.astimezone(timezone.utc)


def filter_by_cutoff(
    docs: list[RawDoc],
    cutoff_utc: datetime,
    lookback_days: int = 3,
) -> list[RawDoc]:
    """Keep only documents published strictly before the cutoff.

    Args:
        lookback_days: how far back to look. Unbounded history would let the
            prompt grow without limit and would blur "what is new today" with
            "what is known", which is the thing being measured.

    Raises:
        CutoffViolation: if a document has a naive timestamp. A naive datetime
            cannot be compared to the cutoff without guessing a zone, and
            guessing is exactly how look-ahead gets in.
    """
    if cutoff_utc.tzinfo is None:
        raise CutoffViolation("cutoff_utc must be timezone-aware")

    window_start = cutoff_utc - timedelta(days=lookback_days)
    kept: list[RawDoc] = []

    for doc in docs:
        if doc.published_at.tzinfo is None:
            raise CutoffViolation(
                f"{doc.url}: naive published_at. Every source adapter must "
                "attach a timezone before returning."
            )
        published_utc = doc.published_at.astimezone(timezone.utc)
        # Strict `<`: a document stamped exactly at the cutoff is ambiguous, and
        # ambiguity resolves against inclusion.
        if window_start <= published_utc < cutoff_utc:
            kept.append(doc)

    return sorted(kept, key=lambda d: d.published_at)


def store_and_reference(
    docs: list[RawDoc],
    corpus_dir: str | Path,
) -> list[CorpusDoc]:
    """Write document texts to a content-addressed store, return ledger refs.

    Content addressing means the filename *is* the hash, so the store is
    naturally deduplicated and a reference in the ledger can never point at
    text that has since changed.
    """
    corpus_path = Path(corpus_dir)
    corpus_path.mkdir(parents=True, exist_ok=True)

    references: list[CorpusDoc] = []
    for doc in docs:
        text_hash = sha256_text(doc.text)
        blob = corpus_path / f"{text_hash}.txt"
        if not blob.exists():
            blob.write_text(doc.text, encoding="utf-8")

        references.append(
            CorpusDoc(
                doc_id=text_hash[:16],
                url=doc.url,
                published_at_utc=doc.published_at.astimezone(
                    timezone.utc
                ).isoformat(timespec="seconds"),
                title=doc.title,
                text_sha256=text_hash,
                char_count=len(doc.text),
            )
        )
    return references


# ---------------------------------------------------------------------------
# Source adapters
# ---------------------------------------------------------------------------
# Each source gets its own function returning list[RawDoc]. Keeping them behind
# one shape means adding a source never touches the cutoff logic above — which is
# the code you least want to edit once it is validated.


def fetch_rss(feed_url: str, timeout: int = 20) -> list[RawDoc]:
    """Pull an RSS/Atom feed into RawDocs.

    Only entries carrying a parseable publication date are returned. An entry
    without a date cannot be placed relative to the cutoff, and this module does
    not guess.
    """
    import feedparser  # imported lazily so the core package has no hard dep

    parsed = feedparser.parse(feed_url)
    docs: list[RawDoc] = []

    for entry in parsed.entries:
        published_struct = entry.get("published_parsed") or entry.get("updated_parsed")
        if published_struct is None:
            continue

        published = datetime(*published_struct[:6], tzinfo=timezone.utc)
        body = entry.get("summary", "") or entry.get("description", "")
        docs.append(
            RawDoc(
                url=entry.get("link", ""),
                title=entry.get("title", ""),
                text=body.strip(),
                published_at=published,
            )
        )
    return docs
