"""
Political Bias Detection (Phase 3)
====================================
Two-layer bias detection:
  Layer 1: Source-based (zero LLM cost) — media bias lookup for each article
  Layer 2: Cluster narrative bias (LLM, budget-gated) — cluster-level framing analysis
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Expanded MEDIA_BIAS lookup (60+ outlets)
# (bias_label, factuality)
# 5-tier scale: left, center-left, center, center-right, right
# Factuality: high, mixed, low
# ---------------------------------------------------------------------------

MEDIA_BIAS_EXPANDED: dict[str, tuple[str, str]] = {
    # --- International (English) ---
    "reuters.com": ("center", "high"),
    "apnews.com": ("center", "high"),
    "bloomberg.com": ("center-right", "high"),
    "ft.com": ("center", "high"),
    "wsj.com": ("center-right", "high"),
    "nytimes.com": ("center-left", "high"),
    "washingtonpost.com": ("center-left", "high"),
    "bbc.com": ("center", "high"),
    "bbc.co.uk": ("center", "high"),
    "cnbc.com": ("center-right", "high"),
    "cnn.com": ("center-left", "mixed"),
    "foxnews.com": ("right", "mixed"),
    "foxbusiness.com": ("right", "mixed"),
    "economist.com": ("center", "high"),
    "theguardian.com": ("center-left", "high"),
    "aljazeera.com": ("center", "mixed"),
    "france24.com": ("center", "high"),
    "dw.com": ("center", "high"),
    "investing.com": ("center", "high"),
    "marketwatch.com": ("center", "high"),
    "yahoo.com": ("center", "mixed"),
    "finance.yahoo.com": ("center", "mixed"),
    "barrons.com": ("center-right", "high"),
    "seekingalpha.com": ("center-right", "mixed"),
    "zerohedge.com": ("right", "low"),
    "thestreet.com": ("center", "mixed"),
    "forbes.com": ("center-right", "high"),

    # --- Colombia ---
    "eltiempo.com": ("center", "mixed"),
    "portafolio.co": ("center", "high"),
    "larepublica.co": ("center", "high"),
    "semana.com": ("center-right", "mixed"),
    "elespectador.com": ("center-left", "high"),
    "dinero.com": ("center", "high"),
    "valora.com": ("center", "mixed"),
    "pulzo.com": ("center", "mixed"),
    "elcolombiano.com": ("center", "mixed"),
    "caracol.com.co": ("center", "mixed"),
    "caracoltv.com": ("center", "mixed"),
    "rcnradio.com": ("center-right", "mixed"),
    "noticiasrcn.com": ("center-right", "mixed"),
    "bluradio.com": ("center", "mixed"),
    "lafm.com.co": ("center", "mixed"),
    "wradio.com.co": ("center", "mixed"),
    "elheraldo.co": ("center", "mixed"),
    "eluniversal.com.co": ("center", "mixed"),
    "elpais.com.co": ("center", "mixed"),
    "lasillavacia.com": ("center-left", "high"),
    "razonpublica.com": ("center-left", "high"),
    "cambio.com.co": ("center-left", "mixed"),

    # --- LATAM ---
    "elfinanciero.com.mx": ("center", "high"),
    "expansion.mx": ("center", "mixed"),
    "eleconomista.com.mx": ("center", "high"),
    "americaeconomia.com": ("center", "high"),
    "infobae.com": ("center", "mixed"),

    # --- Official / Institutional ---
    "banrep.gov.co": ("center", "high"),
    "dane.gov.co": ("center", "high"),
    "minhacienda.gov.co": ("center", "high"),
    "imf.org": ("center", "high"),
    "worldbank.org": ("center", "high"),
    "federalreserve.gov": ("center", "high"),
    "ecb.europa.eu": ("center", "high"),
}


def get_media_bias_expanded(source: str) -> tuple[str, str]:
    """Look up media bias for a source domain.

    Returns:
        (bias_label, factuality). Defaults to ("unknown", "unknown").
    """
    source_lower = source.lower().strip()
    if source_lower in MEDIA_BIAS_EXPANDED:
        return MEDIA_BIAS_EXPANDED[source_lower]
    for domain, bias in MEDIA_BIAS_EXPANDED.items():
        if domain in source_lower:
            return bias
    return ("unknown", "unknown")


# ---------------------------------------------------------------------------
# PoliticalBiasDetector
# ---------------------------------------------------------------------------

class PoliticalBiasDetector:
    """Two-layer political bias detection for news articles.

    Layer 1: Source-based lookup (zero cost, always runs)
    Layer 2: Cluster narrative bias (LLM call, budget-gated, only for large clusters)
    """

    CLUSTER_MIN_ARTICLES = 5  # Only LLM-analyze clusters with >= 5 articles

    def analyze(
        self,
        articles: list[dict],
        clusters: list[dict] | None = None,
        llm_client: object | None = None,
    ) -> dict:
        """Run full bias analysis.

        Args:
            articles: Raw article dicts with at least {title, source}.
            clusters: Optional clusters from NewsIntelligenceEngine.
            llm_client: Optional LLMClient for Layer 2 cluster bias.

        Returns:
            PoliticalBiasOutput dict.
        """
        # Layer 1: Source-based bias distribution
        bias_counts = {"left": 0, "center-left": 0, "center": 0, "center-right": 0, "right": 0, "unknown": 0}
        factuality_counts = {"high": 0, "mixed": 0, "low": 0, "unknown": 0}
        flagged = 0

        for art in articles:
            source = str(art.get("source", art.get("domain", "")))
            bias_label, factuality = get_media_bias_expanded(source)

            bias_counts[bias_label] = bias_counts.get(bias_label, 0) + 1
            factuality_counts[factuality] = factuality_counts.get(factuality, 0) + 1

            if bias_label not in ("center", "unknown"):
                flagged += 1

        # Compute bias diversity score (0-1, 1 = perfectly balanced across spectrum)
        known_counts = [bias_counts[k] for k in ("left", "center-left", "center", "center-right", "right")]
        total_known = sum(known_counts)
        if total_known > 0:
            proportions = [c / total_known for c in known_counts]
            # Shannon entropy normalized to [0, 1]
            import math
            entropy = -sum(p * math.log(p + 1e-10) for p in proportions if p > 0)
            max_entropy = math.log(5)  # 5 categories
            diversity_score = round(entropy / max_entropy, 3)
        else:
            diversity_score = 0.0

        # Layer 2: Cluster narrative bias (LLM, budget-gated)
        cluster_assessments = []
        bias_narrative = ""

        if clusters and llm_client:
            large_clusters = [c for c in clusters if c.get("article_count", 0) >= self.CLUSTER_MIN_ARTICLES]

            for cluster in large_clusters[:5]:  # Max 5 clusters
                assessment = self._assess_cluster_bias(cluster, llm_client)
                if assessment:
                    cluster_assessments.append(assessment)

            if cluster_assessments:
                bias_narrative = self._generate_bias_narrative(
                    bias_counts, diversity_score, cluster_assessments, llm_client,
                )

        return {
            "source_bias_distribution": bias_counts,
            "bias_diversity_score": diversity_score,
            "factuality_distribution": factuality_counts,
            "cluster_bias_assessments": cluster_assessments,
            "flagged_articles": flagged,
            "bias_narrative": bias_narrative,
            "total_analyzed": len(articles),
        }

    def _assess_cluster_bias(self, cluster: dict, llm_client: object) -> dict | None:
        """Assess narrative bias of a single cluster using LLM."""
        titles = cluster.get("representative_titles", [])[:5]
        if not titles:
            return None

        titles_str = "\n".join(f"- {t}" for t in titles)
        label = cluster.get("label", cluster.get("dominant_category", ""))

        prompt = (
            f"Cluster: {label} ({cluster.get('article_count', 0)} articulos)\n"
            f"Titulares:\n{titles_str}\n\n"
            f"Clasifica el sesgo narrativo de este cluster:\n"
            f"- balanced: cobertura equilibrada de multiples perspectivas\n"
            f"- slightly_left: tendencia leve hacia perspectivas progresistas\n"
            f"- slightly_right: tendencia leve hacia perspectivas conservadoras\n"
            f"- left_leaning: sesgo claro hacia perspectivas progresistas\n"
            f"- right_leaning: sesgo claro hacia perspectivas conservadoras\n\n"
            f"Responde SOLO con el label y confianza (0-1).\n"
            f"Formato: LABEL|CONFIANZA"
        )

        try:
            import hashlib
            cache_key = f"bias_cluster_{hashlib.md5(titles_str.encode()).hexdigest()[:12]}"
            result = llm_client.generate(
                "Eres un analista de medios experto en sesgo editorial. Responde solo el formato pedido.",
                prompt,
                max_tokens=50,
                cache_key=cache_key,
            )
            content = result.get("content", "").strip()
            parts = content.split("|")
            bias_label = parts[0].strip().lower() if parts else "balanced"
            confidence = float(parts[1].strip()) if len(parts) > 1 else 0.5

            valid_labels = {"balanced", "slightly_left", "slightly_right", "left_leaning", "right_leaning"}
            if bias_label not in valid_labels:
                bias_label = "balanced"

            return {
                "cluster_label": label,
                "bias_label": bias_label,
                "confidence": round(confidence, 2),
                "article_count": cluster.get("article_count", 0),
            }
        except Exception as e:
            logger.warning(f"Cluster bias assessment failed: {e}")
            return None

    def _generate_bias_narrative(
        self,
        bias_counts: dict,
        diversity_score: float,
        cluster_assessments: list[dict],
        llm_client: object,
    ) -> str:
        """Generate a summary narrative of the bias landscape."""
        try:
            bias_summary = ", ".join(f"{k}: {v}" for k, v in bias_counts.items() if v > 0)
            cluster_summary = "; ".join(
                f"{a['cluster_label']}={a['bias_label']}" for a in cluster_assessments
            )

            prompt = (
                f"Distribucion de sesgo de fuentes: {bias_summary}\n"
                f"Score de diversidad: {diversity_score:.2f}/1.00\n"
                f"Sesgos por cluster: {cluster_summary}\n\n"
                f"Genera 2-3 oraciones en español resumiendo el panorama de sesgo mediatico "
                f"para la cobertura de USD/COP esta semana. Se objetivo y conciso."
            )

            result = llm_client.generate(
                "Eres un analista de medios. Escribe de forma objetiva y concisa.",
                prompt,
                max_tokens=200,
                cache_key=f"bias_narrative_{hash(bias_summary)}",
            )
            return result.get("content", "")
        except Exception as e:
            logger.warning(f"Bias narrative generation failed: {e}")
            return ""
