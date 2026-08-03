from datetime import datetime, timezone

from src.news_engine.cross_reference.engine import CrossReferenceEngine
from src.news_engine.config import CrossReferenceConfig
from src.news_engine.models import EnrichedArticle, RawArticle


def _article(article_id: int, title: str) -> EnrichedArticle:
    raw = RawArticle(
        url=f"https://example.test/{article_id}",
        title=title,
        source_id=f"source_{article_id}",
        published_at=datetime.now(timezone.utc),
    )
    # Database identity is attached by the Airflow row-to-model boundary.
    raw.id = article_id
    return EnrichedArticle(raw=raw, category="macro", relevance_score=1.0)


def test_cross_reference_carries_database_article_ids():
    engine = CrossReferenceEngine(CrossReferenceConfig(similarity_threshold=0.0))
    cluster = engine.find_clusters([
        _article(101, "Colombia central bank rate decision"),
        _article(102, "Colombia central bank rate decision"),
        _article(103, "Colombia central bank rate decision"),
    ])[0]
    assert cluster.articles == [101, 102, 103]
