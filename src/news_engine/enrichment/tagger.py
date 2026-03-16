"""
Keyword Tagger (SDD-04 §5)
============================
Extracts keywords and named entities from article text.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional


# Named entity patterns (financial context)
ENTITY_PATTERNS = [
    # Central banks
    (r"Banco\s+de\s+la\s+Rep[uú]blica", "Banco de la Republica"),
    (r"Federal\s+Reserve|Fed\b", "Federal Reserve"),
    (r"BCE|ECB|European\s+Central\s+Bank", "ECB"),
    # Colombian entities
    (r"DANE", "DANE"),
    (r"SuperFinanciera", "SuperFinanciera"),
    (r"MinHacienda", "MinHacienda"),
    (r"Fedesarrollo", "Fedesarrollo"),
    # International
    (r"FMI|IMF", "IMF"),
    (r"Banco\s+Mundial|World\s+Bank", "World Bank"),
    (r"OPEP|OPEC", "OPEC"),
    # People
    (r"Jerome\s+Powell", "Jerome Powell"),
    (r"Leonardo\s+Villar", "Leonardo Villar"),
    (r"Gustavo\s+Petro", "Gustavo Petro"),
]

# Common financial keywords to extract
FINANCIAL_KEYWORDS = [
    "dolar", "peso", "tasa de cambio", "tasa de interes",
    "inflacion", "petroleo", "embi", "riesgo pais",
    "PIB", "GDP", "desempleo", "unemployment",
    "deficit", "superavit", "remesas", "exportaciones",
    "importaciones", "TES", "bonos", "acciones",
    "devaluacion", "revaluacion", "volatilidad",
]


def extract_keywords(
    title: str,
    content: Optional[str] = None,
    max_keywords: int = 10,
) -> list[str]:
    """Extract relevant financial keywords from text.

    Returns up to max_keywords sorted by frequency.
    """
    text = (title or "").lower() + " " + (content or "").lower()

    found = []
    for kw in FINANCIAL_KEYWORDS:
        if kw.lower() in text:
            found.append(kw)

    # Also extract significant words (capitalized, >3 chars)
    words = re.findall(r"\b[A-Z][a-z]{3,}\b", (title or "") + " " + (content or ""))
    word_counts = Counter(words)
    for word, count in word_counts.most_common(5):
        if word.lower() not in {"para", "como", "este", "esta", "sobre", "desde"}:
            found.append(word)

    # Deduplicate preserving order
    seen = set()
    unique = []
    for kw in found:
        if kw.lower() not in seen:
            seen.add(kw.lower())
            unique.append(kw)

    return unique[:max_keywords]


def extract_entities(
    title: str,
    content: Optional[str] = None,
) -> list[str]:
    """Extract named entities using pattern matching."""
    text = (title or "") + " " + (content or "")

    entities = []
    for pattern, entity_name in ENTITY_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            entities.append(entity_name)

    return list(set(entities))
