"""
Unit tests for the pluggable news-source module (src/analysis/news_sources.py).

Locks in the scalability contract: new sources register cleanly, the factory
honours the SSOT order, and the facade strategies (first_nonempty / aggregate)
behave. Uses fake in-memory sources — no network.
"""

from __future__ import annotations

from datetime import date

import pytest

from src.analysis.news_sources import (
    AssetNewsFetcher,
    NewsArticle,
    NewsQuery,
    SOURCE_REGISTRY,
    build_news_sources,
)


def _q() -> NewsQuery:
    return NewsQuery(
        asset_id="test", google_query="x", gdelt_query="x",
        w_start=date(2026, 5, 11), w_end=date(2026, 5, 17),
    )


class _FakeSource:
    def __init__(self, name, arts):
        self.name = name
        self._arts = arts

    def fetch(self, q):
        return list(self._arts)


class _BoomSource:
    name = "boom"

    def fetch(self, q):
        raise RuntimeError("network down")


def test_registry_has_builtin_sources():
    assert "google_news" in SOURCE_REGISTRY
    assert "gdelt" in SOURCE_REGISTRY


def test_build_news_sources_honours_ssot_order():
    cfg = {"sources": ["gdelt", "google_news"]}
    sources = build_news_sources(cfg)
    assert [s.name for s in sources] == ["gdelt", "google_news"]


def test_build_news_sources_skips_unknown():
    cfg = {"sources": ["google_news", "does_not_exist"]}
    sources = build_news_sources(cfg)
    assert [s.name for s in sources] == ["google_news"]


def test_first_nonempty_returns_primary_then_falls_back():
    a1 = [NewsArticle(title="from-fallback", url="u1", source="s1")]
    fetcher = AssetNewsFetcher(
        sources=[_FakeSource("primary", []), _FakeSource("fallback", a1)],
        strategy="first_nonempty",
    )
    out = fetcher.fetch(_q())
    assert len(out) == 1 and out[0].title == "from-fallback"


def test_aggregate_merges_and_dedups():
    dup = NewsArticle(title="t", url="http://x/1", source="s")
    other = NewsArticle(title="t2", url="http://x/2", source="s")
    fetcher = AssetNewsFetcher(
        sources=[_FakeSource("a", [dup]), _FakeSource("b", [dup, other])],
        strategy="aggregate",
    )
    out = fetcher.fetch(_q())
    assert len(out) == 2  # dup collapsed by url key


def test_failing_source_is_isolated():
    good = [NewsArticle(title="ok", url="u", source="s")]
    fetcher = AssetNewsFetcher(
        sources=[_BoomSource(), _FakeSource("good", good)],
        strategy="first_nonempty",
    )
    out = fetcher.fetch(_q())  # boom must not propagate
    assert len(out) == 1 and out[0].title == "ok"


def test_new_source_plugs_in_via_registry():
    """Adding a source = register a class; factory picks it up. No other change."""
    SOURCE_REGISTRY["fake_test_src"] = lambda cfg: _FakeSource("fake_test_src", [])
    try:
        sources = build_news_sources({"sources": ["fake_test_src"]})
        assert len(sources) == 1 and sources[0].name == "fake_test_src"
    finally:
        SOURCE_REGISTRY.pop("fake_test_src", None)
