"""
Article Categorizer (SDD-04 §2)
=================================
Assigns one of 9 categories based on title + content keyword matching.
"""

from __future__ import annotations

import re

# Category rules: {category: [keyword_patterns]}
CATEGORY_RULES = {
    "monetary_policy": [
        r"banco\s*(de\s*la\s*)?republica", r"tasa\s*de\s*interes",
        r"fed\b", r"federal\s*reserve", r"central\s*bank",
        r"interest\s*rate", r"rate\s*(hike|cut|decision)",
        r"politica\s*monetaria", r"monetary\s*policy",
        r"banrep", r"jdgbr", r"fomc",
    ],
    "fx_market": [
        r"dolar", r"tasa\s*de\s*cambio", r"usd\s*cop", r"usdcop",
        r"peso\s*colombiano", r"exchange\s*rate", r"forex",
        r"devaluacion", r"revaluacion", r"currency",
        r"divisa", r"paridad",
    ],
    "commodities": [
        r"petroleo", r"oil", r"wti", r"brent", r"crude",
        r"oro\b", r"gold", r"carbon\b", r"coal",
        r"cafe\b", r"coffee", r"commodity", r"materias\s*primas",
    ],
    "inflation": [
        r"inflacion", r"inflation", r"ipc\b", r"cpi\b",
        r"precios", r"cost\s*of\s*living", r"deflacion",
        r"indice\s*de\s*precios",
    ],
    "fiscal_policy": [
        r"reforma\s*tributaria", r"impuesto", r"tax",
        r"presupuesto", r"budget", r"fiscal",
        r"gasto\s*publico", r"deuda\s*publica", r"government\s*spending",
    ],
    "risk_premium": [
        r"embi", r"riesgo\s*pais", r"country\s*risk",
        r"spread", r"tes\b", r"credit\s*default\s*swap",
        r"sovereign\s*risk", r"prima\s*de\s*riesgo",
    ],
    "capital_flows": [
        r"inversion\s*extranjera", r"foreign\s*investment",
        r"fdi\b", r"ied\b", r"capital\s*flow",
        r"portafolio\s*de\s*inversion", r"hot\s*money",
    ],
    "balance_payments": [
        r"remesas", r"remittances", r"balanza\s*de\s*pagos",
        r"balance\s*of\s*payments", r"cuenta\s*corriente",
        r"current\s*account", r"exportacion", r"importacion",
    ],
    "political": [
        r"presidente", r"president", r"congreso", r"congress",
        r"eleccion", r"election", r"politico",
        r"petro\b", r"gobierno", r"government",
    ],
}

# Pre-compile patterns
_COMPILED_RULES = {
    cat: [re.compile(p, re.IGNORECASE) for p in patterns]
    for cat, patterns in CATEGORY_RULES.items()
}


def categorize_article(
    title: str,
    content: str | None = None,
    summary: str | None = None,
) -> tuple[str | None, str | None]:
    """Assign category and subcategory based on keyword matching.

    Returns:
        (category, subcategory) — subcategory is None for now.
    """
    text = (title or "") + " " + (content or "") + " " + (summary or "")
    text = text.lower()

    scores = {}
    for category, patterns in _COMPILED_RULES.items():
        score = sum(1 for p in patterns if p.search(text))
        if score > 0:
            scores[category] = score

    if not scores:
        return None, None

    # Return category with highest match count
    best = max(scores, key=scores.get)
    return best, None
