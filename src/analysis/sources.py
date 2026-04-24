"""
Deterministic Source URLs for Macro Data & News
================================================
Provides canonical URLs for all macro variable data sources and builds
a "### Fuentes" markdown section to append AFTER LLM generation.

This module is deterministic — it NEVER depends on LLM output.

Contract: Part of CTR-ANALYSIS-SCHEMA-001
"""


# Canonical mapping: variable_key -> (display_name, url)
MACRO_SOURCE_URLS: dict[str, tuple[str, str]] = {
    "dxy": (
        "DXY — Investing.com",
        "https://www.investing.com/indices/usdollar",
    ),
    "vix": (
        "VIX — CBOE",
        "https://www.investing.com/indices/volatility-s-p-500",
    ),
    "wti": (
        "WTI — NYMEX",
        "https://www.investing.com/commodities/crude-oil",
    ),
    "brent": (
        "Brent — ICE",
        "https://www.investing.com/commodities/brent-oil",
    ),
    "gold": (
        "Oro — Investing.com",
        "https://www.investing.com/commodities/gold",
    ),
    "embi_col": (
        "EMBI Colombia — BanRep",
        "https://www.banrep.gov.co/es/estadisticas/spreads-deuda-publica",
    ),
    "ust10y": (
        "UST 10Y — FRED",
        "https://fred.stlouisfed.org/series/DGS10",
    ),
    "ust2y": (
        "UST 2Y — FRED",
        "https://fred.stlouisfed.org/series/DGS2",
    ),
    "ibr": (
        "IBR — BanRep",
        "https://www.banrep.gov.co/es/estadisticas/tasas-interes-interbancarias",
    ),
    "tpm": (
        "TPM — BanRep",
        "https://www.banrep.gov.co/es/estadisticas/tasas-intervencion-politica-monetaria",
    ),
    "fedfunds": (
        "Fed Funds — FRED",
        "https://fred.stlouisfed.org/series/FEDFUNDS",
    ),
    "cpi_us": (
        "CPI USA — FRED",
        "https://fred.stlouisfed.org/series/CPIAUCSL",
    ),
    "cpi_col": (
        "IPC Colombia — DANE",
        "https://www.dane.gov.co/index.php/estadisticas-por-tema/precios-y-costos/"
        "indice-de-precios-al-consumidor-ipc",
    ),
    "coffee": (
        "Cafe — Investing.com",
        "https://www.investing.com/commodities/us-coffee-c",
    ),
    "colcap": (
        "COLCAP — BVC",
        "https://www.bvc.com.co/",
    ),
}


def get_source_url(key: str) -> tuple[str, str] | None:
    """Return (display_name, url) for a macro variable key, or None if unknown."""
    return MACRO_SOURCE_URLS.get(key)


def build_fuentes_section(
    macro_keys: list[str],
    news_highlights: list[dict[str, str]] | None = None,
) -> str:
    """Build a deterministic '### Fuentes' markdown section.

    This is appended AFTER LLM generation to guarantee correct, clickable links
    without relying on the LLM to produce them.

    Args:
        macro_keys: List of macro variable keys used in the analysis
                    (e.g. ["dxy", "vix", "wti", "embi_col"]).
        news_highlights: Optional list of news dicts, each with at least
                         "title" and optionally "url".

    Returns:
        Markdown string starting with "### Fuentes".
    """
    lines: list[str] = ["", "### Fuentes", ""]

    # --- Macro data sources ---
    macro_links: list[str] = []
    for key in macro_keys:
        source = MACRO_SOURCE_URLS.get(key)
        if source:
            name, url = source
            macro_links.append(f"[{name}]({url})")

    if macro_links:
        lines.append("**Datos Macro:**")
        for link in macro_links:
            lines.append(f"- {link}")
        lines.append("")

    # --- News sources ---
    if news_highlights:
        news_links: list[str] = []
        for item in news_highlights:
            title = item.get("title", "").strip()
            url = item.get("url", "").strip()
            if title and url:
                news_links.append(f"[{title}]({url})")
            elif title:
                news_links.append(title)

        if news_links:
            lines.append("**Noticias Destacadas:**")
            for link in news_links:
                lines.append(f"- {link}")
            lines.append("")

    return "\n".join(lines)
