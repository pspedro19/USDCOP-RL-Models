"""
Hybrid Sentiment Analysis System for USDCOP News
==================================================
Combines multilingual transformer models, FinBERT (EN-only), LLM batch scoring,
GDELT tone normalization, and FX-specific impact rules into a weighted ensemble.

Architecture:
    - Multilingual: cardiffnlp/twitter-xlm-roberta-base-sentiment (ES+EN)
    - FinBERT: ProsusAI/finbert (EN-only, financial domain)
    - LLM batch: GPT-4o-mini, 1 call per week for all titles
    - GDELT tone: normalized to [-1, 1] when available
    - FX impact rules: USDCOP-specific adjustments (oil, Fed, BanRep, etc.)

Blending weights:
    EN: 0.30 multi + 0.30 FinBERT + 0.30 LLM + 0.10 GDELT
    ES: 0.50 multi + 0.00 FinBERT + 0.40 LLM + 0.10 GDELT
    Missing signals → redistribute weights proportionally.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level singleton (type annotation deferred to avoid forward ref)
_analyzer = None  # Optional[SentimentAnalyzer]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SentimentResult:
    """Result of sentiment analysis for a single article."""

    score: float = 0.0              # Blended score, -1 to +1
    label: str = "neutral"          # "positive" / "negative" / "neutral"
    fx_adjusted_score: float = 0.0  # After FX impact rules
    components: dict = field(default_factory=dict)  # {"multilingual": 0.3, ...}
    confidence: float = 0.0         # Model agreement (0-1)


# ---------------------------------------------------------------------------
# FX Impact Rules (USDCOP-specific)
# ---------------------------------------------------------------------------

class FXImpactRules:
    """USDCOP-specific sentiment adjustment rules.

    Adjustments reflect the impact on COP strength:
    - Positive adjustment = good for COP (COP strengthens, USDCOP falls)
    - Negative adjustment = bad for COP (COP weakens, USDCOP rises)
    """

    # (pattern_list, adjustment, description)
    RULES: list[tuple[list[str], float, str]] = [
        # Oil up → COP strengthens (Colombia is oil exporter)
        (["oil price surge", "oil prices rise", "oil rally", "crude up",
          "petroleo sube", "petroleo al alza", "wti sube", "brent sube",
          "oil prices surge", "crude oil rally"], +0.15, "oil_up"),
        # Oil down → COP weakens
        (["oil price drop", "oil prices fall", "crude down", "oil crash",
          "petroleo baja", "petroleo cae", "wti cae", "brent cae",
          "oil slump", "crude decline"], -0.15, "oil_down"),
        # Fed hike → stronger USD → COP weakens
        (["fed hike", "fed raises", "rate hike", "fed tightening",
          "subida de tasas fed", "fed sube tasas", "hawkish fed",
          "fed hawkish", "rate increase"], -0.20, "fed_hike"),
        # Fed cut → weaker USD → COP strengthens
        (["fed cut", "fed lowers", "rate cut", "fed easing",
          "recorte de tasas fed", "fed baja tasas", "dovish fed",
          "fed dovish"], +0.15, "fed_cut"),
        # BanRep cut → weaker COP
        (["banrep recorta", "banrep baja", "banco de la republica recorta",
          "banrep rate cut", "colombia rate cut",
          "banco de la republica baja"], -0.15, "banrep_cut"),
        # BanRep hike → attracts capital → COP strengthens
        (["banrep sube", "banrep aumenta", "banco de la republica sube",
          "banrep rate hike", "colombia rate hike"], +0.10, "banrep_hike"),
        # DXY/dollar strength → COP weakens
        (["dollar strength", "dxy rally", "dxy sube", "dolar se fortalece",
          "strong dollar", "dollar index rise", "greenback surge",
          "dolar sube"], -0.20, "usd_strength"),
        # DXY weakness → COP strengthens
        (["dollar weakness", "dxy falls", "dxy baja", "dolar se debilita",
          "weak dollar", "dollar index fall", "greenback drops",
          "dolar cae"], +0.15, "usd_weakness"),
        # EMBI spread widens → COP weakens
        (["embi sube", "riesgo pais sube", "spread widens",
          "country risk rise", "embi increase", "risk premium rise",
          "prima de riesgo sube"], -0.15, "embi_up"),
        # EM capital inflows → COP strengthens
        (["emerging market inflows", "flujos emergentes",
          "capital inflows", "inversion extranjera sube",
          "em rally", "emerging rally", "latam inflows"], +0.15, "em_inflows"),
        # EM crisis / capital flight → COP weakens
        (["emerging crisis", "em outflows", "capital flight",
          "fuga de capitales", "crisis emergentes",
          "risk off", "risk-off"], -0.20, "em_crisis"),
        # Colombia specific positive
        (["colombia upgrade", "colombia inversion", "colombia gdp up",
          "pib colombia sube", "exportaciones colombia suben"], +0.10, "col_positive"),
        # Colombia specific negative
        (["colombia downgrade", "deficit fiscal colombia",
          "colombia recession", "recession colombia",
          "devaluacion peso"], -0.15, "col_negative"),
    ]

    MAX_ADJUSTMENT = 0.3

    @classmethod
    def compute_adjustment(cls, title: str, content: Optional[str] = None) -> tuple[float, list[str]]:
        """Compute FX impact adjustment for an article.

        Returns:
            (adjustment, matched_rules) where adjustment is clamped to [-MAX, +MAX].
        """
        text = (title or "").lower()
        if content:
            text += " " + content[:500].lower()

        total_adj = 0.0
        matched = []

        for patterns, adjustment, rule_name in cls.RULES:
            for pattern in patterns:
                if pattern in text:
                    total_adj += adjustment
                    matched.append(rule_name)
                    break  # One match per rule group

        # Clamp
        total_adj = max(-cls.MAX_ADJUSTMENT, min(cls.MAX_ADJUSTMENT, total_adj))
        return total_adj, matched


# ---------------------------------------------------------------------------
# Language detection (lightweight)
# ---------------------------------------------------------------------------

_ES_MARKERS = {
    "el", "la", "los", "las", "de", "del", "en", "por", "para", "con",
    "que", "se", "un", "una", "es", "como", "más", "pero", "su", "al",
    "dólar", "precio", "tasa", "mercado", "economía", "gobierno",
}


def detect_language(text: str) -> str:
    """Simple language detection based on common Spanish markers.

    Returns 'es' or 'en'.
    """
    if not text:
        return "en"
    words = set(text.lower().split())
    es_count = len(words & _ES_MARKERS)
    return "es" if es_count >= 3 else "en"


# ---------------------------------------------------------------------------
# Core Analyzer
# ---------------------------------------------------------------------------

class SentimentAnalyzer:
    """Hybrid sentiment analyzer combining multiple signals.

    Models are lazy-loaded on first use. CPU-only inference.
    """

    # Default blending weights by language
    DEFAULT_WEIGHTS = {
        "en": {"multilingual": 0.30, "finbert": 0.30, "llm": 0.30, "gdelt": 0.10},
        "es": {"multilingual": 0.50, "finbert": 0.00, "llm": 0.40, "gdelt": 0.10},
    }

    DEFAULT_THRESHOLDS = {"positive": 0.15, "negative": -0.15}

    def __init__(self, config: Optional[dict] = None):
        self._config = config or {}
        self._multilingual_pipeline = None
        self._finbert_pipeline = None
        self._models_loaded = False

        # Extract config
        self._weights = self._config.get("weights", self.DEFAULT_WEIGHTS)
        self._thresholds = self._config.get("thresholds", self.DEFAULT_THRESHOLDS)
        self._fx_enabled = self._config.get("fx_adjustment", {}).get("enabled", True)
        self._multilingual_model_name = self._config.get(
            "multilingual_model",
            "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        )
        self._finbert_model_name = self._config.get(
            "finbert_model",
            "ProsusAI/finbert",
        )
        self._max_length = self._config.get("max_length", 512)
        self._device = self._config.get("device", "cpu")

    def _load_models(self) -> None:
        """Lazy-load transformer models. CPU-only."""
        if self._models_loaded:
            return

        # Multilingual sentiment model
        if self._multilingual_model_name:
            try:
                from transformers import pipeline as hf_pipeline
                self._multilingual_pipeline = hf_pipeline(
                    "sentiment-analysis",
                    model=self._multilingual_model_name,
                    top_k=None,
                    device=-1,  # CPU
                    truncation=True,
                    max_length=self._max_length,
                )
                logger.info(f"Loaded multilingual model: {self._multilingual_model_name}")
            except Exception as e:
                logger.warning(f"Failed to load multilingual model: {e}")

        # FinBERT (English-only financial sentiment)
        if self._finbert_model_name:
            try:
                from transformers import pipeline as hf_pipeline
                self._finbert_pipeline = hf_pipeline(
                    "sentiment-analysis",
                    model=self._finbert_model_name,
                    top_k=None,
                    device=-1,
                    truncation=True,
                    max_length=self._max_length,
                )
                logger.info(f"Loaded FinBERT model: {self._finbert_model_name}")
            except Exception as e:
                logger.warning(f"Failed to load FinBERT: {e}")

        self._models_loaded = True

    # ------------------------------------------------------------------
    # Individual model scores
    # ------------------------------------------------------------------

    def _score_multilingual(self, text: str) -> Optional[float]:
        """Score with multilingual model. Returns [-1, 1] or None."""
        if self._multilingual_pipeline is None:
            return None
        try:
            result = self._multilingual_pipeline(text[:self._max_length])
            if not result:
                return None
            scores = result[0] if isinstance(result[0], list) else result
            # cardiffnlp output: labels are "positive", "negative", "neutral"
            score_map = {}
            for item in scores:
                label = item["label"].lower()
                score_map[label] = item["score"]

            # Convert to [-1, 1]: positive - negative (neutral acts as anchor)
            pos = score_map.get("positive", score_map.get("pos", 0))
            neg = score_map.get("negative", score_map.get("neg", 0))
            return round(pos - neg, 4)
        except Exception as e:
            logger.debug(f"Multilingual scoring failed: {e}")
            return None

    def _score_finbert(self, text: str) -> Optional[float]:
        """Score with FinBERT (English only). Returns [-1, 1] or None."""
        if self._finbert_pipeline is None:
            return None
        try:
            result = self._finbert_pipeline(text[:self._max_length])
            if not result:
                return None
            scores = result[0] if isinstance(result[0], list) else result
            pos = sum(s["score"] for s in scores if s["label"] == "positive")
            neg = sum(s["score"] for s in scores if s["label"] == "negative")
            return round(pos - neg, 4)
        except Exception as e:
            logger.debug(f"FinBERT scoring failed: {e}")
            return None

    @staticmethod
    def _normalize_gdelt_tone(tone: float) -> float:
        """Normalize GDELT tone to [-1, 1]."""
        clamped = max(-20.0, min(20.0, tone))
        return round(clamped / 20.0, 4)

    # ------------------------------------------------------------------
    # Blending
    # ------------------------------------------------------------------

    def _blend_scores(
        self,
        language: str,
        multilingual_score: Optional[float] = None,
        finbert_score: Optional[float] = None,
        llm_score: Optional[float] = None,
        gdelt_score: Optional[float] = None,
    ) -> tuple[float, dict, float]:
        """Blend available scores using language-specific weights.

        Returns:
            (blended_score, component_dict, confidence)
        """
        weights = dict(self._weights.get(language, self._weights.get("en", {})))

        # Map signal names to values
        signals = {
            "multilingual": multilingual_score,
            "finbert": finbert_score,
            "llm": llm_score,
            "gdelt": gdelt_score,
        }

        # Filter to available signals
        available = {k: v for k, v in signals.items() if v is not None and weights.get(k, 0) > 0}

        if not available:
            return 0.0, {}, 0.0

        # Redistribute weights proportionally for missing signals
        total_available_weight = sum(weights.get(k, 0) for k in available)
        if total_available_weight <= 0:
            return 0.0, {}, 0.0

        scale = 1.0 / total_available_weight

        blended = 0.0
        components = {}
        for k, v in available.items():
            w = weights.get(k, 0) * scale
            blended += v * w
            components[k] = round(v, 4)

        blended = max(-1.0, min(1.0, round(blended, 4)))

        # Confidence: agreement between available signals
        values = list(available.values())
        if len(values) >= 2:
            # 1 - normalized std dev (higher agreement = higher confidence)
            std = float(__import__("statistics").stdev(values)) if len(values) > 1 else 0
            confidence = max(0.0, min(1.0, 1.0 - std))
        else:
            confidence = 0.5  # Single signal = medium confidence

        return blended, components, round(confidence, 3)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_single(
        self,
        title: str,
        content: Optional[str] = None,
        language: Optional[str] = None,
        gdelt_tone: Optional[float] = None,
        llm_score: Optional[float] = None,
    ) -> SentimentResult:
        """Analyze sentiment of a single article.

        Args:
            title: Article title (required).
            content: Article body text (optional, used for FX rules).
            language: 'en' or 'es'. Auto-detected if None.
            gdelt_tone: Raw GDELT tone value (if available).
            llm_score: Pre-computed LLM score (if available).

        Returns:
            SentimentResult with blended score and components.
        """
        self._load_models()

        if not title:
            return SentimentResult()

        lang = language or detect_language(title)
        text = title[:self._max_length]

        # Score with available models
        multi_score = self._score_multilingual(text)
        fin_score = self._score_finbert(text) if lang == "en" else None
        gdelt_score = self._normalize_gdelt_tone(gdelt_tone) if gdelt_tone is not None else None

        # Blend
        blended, components, confidence = self._blend_scores(
            lang,
            multilingual_score=multi_score,
            finbert_score=fin_score,
            llm_score=llm_score,
            gdelt_score=gdelt_score,
        )

        # FX impact adjustment
        fx_adj = 0.0
        if self._fx_enabled:
            fx_adj, matched_rules = FXImpactRules.compute_adjustment(title, content)
            if matched_rules:
                components["fx_rules"] = matched_rules

        fx_adjusted = max(-1.0, min(1.0, round(blended + fx_adj, 4)))

        # Label from thresholds
        pos_thresh = self._thresholds.get("positive", 0.15)
        neg_thresh = self._thresholds.get("negative", -0.15)
        if fx_adjusted >= pos_thresh:
            label = "positive"
        elif fx_adjusted <= neg_thresh:
            label = "negative"
        else:
            label = "neutral"

        return SentimentResult(
            score=blended,
            label=label,
            fx_adjusted_score=fx_adjusted,
            components=components,
            confidence=confidence,
        )

    def analyze_batch(
        self,
        articles: list[dict],
        llm_scores: Optional[dict[str, float]] = None,
    ) -> list[SentimentResult]:
        """Analyze sentiment for a batch of articles.

        Args:
            articles: List of dicts with at least 'title'. Optional: 'content',
                      'language', 'gdelt_tone', 'url'.
            llm_scores: Optional dict mapping title_hash -> score from LLM batch.

        Returns:
            List of SentimentResult, one per article.
        """
        self._load_models()
        results = []

        for art in articles:
            title = art.get("title", "")
            llm_score = None
            if llm_scores and title:
                # Look up by title hash or exact title
                key = _title_hash(title)
                llm_score = llm_scores.get(key, llm_scores.get(title))

            result = self.analyze_single(
                title=title,
                content=art.get("content"),
                language=art.get("language"),
                gdelt_tone=art.get("gdelt_tone"),
                llm_score=llm_score,
            )
            results.append(result)

        return results

    def score_batch_with_llm(
        self,
        articles: list[dict],
        llm_client: object,
        cache_key: str = "sentiment_batch",
    ) -> dict[str, float]:
        """Score a batch of articles using a single LLM call.

        The LLM is asked to score each title for COP impact (-1 to +1).
        Results are cached for 7 days.

        Args:
            articles: List of dicts with at least 'title'.
            llm_client: LLMClient instance with generate() method.
            cache_key: Cache key prefix for this batch.

        Returns:
            Dict mapping title_hash -> score.
        """
        if not articles:
            return {}

        max_per_call = self._config.get("llm_batch", {}).get("max_articles_per_call", 50)
        titles = []
        for art in articles:
            t = art.get("title", "").strip()
            if t and len(t) > 10:
                titles.append(t)
        titles = titles[:max_per_call]

        if not titles:
            return {}

        # Build prompt
        title_list = "\n".join(f"{i+1}. \"{t}\"" for i, t in enumerate(titles))

        system_prompt = (
            "You are a financial sentiment analyst specializing in the Colombian peso (COP) "
            "and USD/COP exchange rate.\n\n"
            "Score each article title from -1.0 (negative for COP / COP weakens) to "
            "+1.0 (positive for COP / COP strengthens).\n\n"
            "Key context:\n"
            "- Higher oil prices HELP Colombia (oil exporter) → positive for COP\n"
            "- Fed rate hikes STRENGTHEN USD → negative for COP\n"
            "- BanRep rate cuts WEAKEN COP → negative\n"
            "- Risk-off / EM outflows → negative for COP\n"
            "- Strong DXY → negative for COP\n"
            "- EMBI spread widening → negative for COP\n\n"
            "Return ONLY valid JSON. No explanation."
        )

        user_prompt = (
            f"Score these {len(titles)} titles for COP impact:\n\n"
            f"{title_list}\n\n"
            'Format: {"scores": [{"id": 1, "score": 0.3}, ...]}'
        )

        try:
            result = llm_client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=500,
                temperature=0.3,
                cache_key=cache_key,
            )

            content = result.get("content", "")
            cost = result.get("cost_usd", 0)
            logger.info(f"LLM sentiment batch: {len(titles)} titles, cost=${cost:.4f}")

            # Parse JSON response
            scores_dict = self._parse_llm_scores(content, titles)
            return scores_dict

        except Exception as e:
            logger.warning(f"LLM batch sentiment failed: {e}")
            return {}

    @staticmethod
    def _parse_llm_scores(content: str, titles: list[str]) -> dict[str, float]:
        """Parse LLM JSON response into title_hash -> score dict."""
        scores_dict = {}

        # Try to extract JSON from content
        try:
            # Handle both raw JSON and markdown-wrapped JSON
            json_str = content
            if "```" in content:
                match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
                if match:
                    json_str = match.group(1)

            parsed = json.loads(json_str)
            score_list = parsed.get("scores", [])

            for item in score_list:
                idx = int(item.get("id", 0)) - 1  # 1-indexed
                score = float(item.get("score", 0))
                score = max(-1.0, min(1.0, score))  # Clamp
                if 0 <= idx < len(titles):
                    key = _title_hash(titles[idx])
                    scores_dict[key] = round(score, 3)

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse LLM sentiment scores: {e}")

        return scores_dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _title_hash(title: str) -> str:
    """Deterministic hash for title lookup."""
    return title.strip().lower()[:100]


def _score_to_label(score: float, pos_thresh: float = 0.15, neg_thresh: float = -0.15) -> str:
    """Convert score to sentiment label."""
    if score >= pos_thresh:
        return "positive"
    elif score <= neg_thresh:
        return "negative"
    return "neutral"


# ---------------------------------------------------------------------------
# Singleton access
# ---------------------------------------------------------------------------

def get_analyzer(config: Optional[dict] = None) -> SentimentAnalyzer:
    """Get or create the module-level SentimentAnalyzer singleton.

    Config is loaded from weekly_analysis_ssot.yaml if not provided.
    """
    global _analyzer
    if _analyzer is None:
        if config is None:
            config = _load_default_config()
        _analyzer = SentimentAnalyzer(config)
    return _analyzer


def reset_analyzer() -> None:
    """Reset the singleton (for testing)."""
    global _analyzer
    _analyzer = None


def _load_default_config() -> dict:
    """Load sentiment config from weekly_analysis_ssot.yaml."""
    config_path = Path(__file__).resolve().parents[2] / "config/analysis/weekly_analysis_ssot.yaml"
    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                full_config = yaml.safe_load(f)
            return full_config.get("sentiment", {})
        except Exception as e:
            logger.warning(f"Failed to load sentiment config: {e}")
    return {}
