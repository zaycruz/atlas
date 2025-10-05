from __future__ import annotations

from atlas_main.search import SearchResult
from atlas_main.search import engines


def test_normalize_and_dedupe_preserves_first() -> None:
    items = [
        SearchResult(title="A", url="https://example.com/article?utm=1", snippet="one", source="ddg"),
        SearchResult(title="B", url="https://example.com/article?utm=2", snippet="two", source="bing"),
        SearchResult(title="C", url="https://example.org/other", snippet="three", source="ddg"),
    ]
    deduped = engines.normalize_and_dedupe(items)
    assert len(deduped) == 2
    assert deduped[0].title == "A"
    assert deduped[0].snippet == "one"


def test_federated_search_uses_cache(monkeypatch) -> None:
    captured = {}

    def fake_serp_get(key: str):
        captured["key"] = key
        return [SearchResult(title="C", url="https://cached.example", snippet="cached", source="ddg")]

    monkeypatch.setattr(engines, "serp_get", fake_serp_get)
    monkeypatch.setattr(engines, "serp_put", lambda *args, **kwargs: None)

    results = engines.federated_search("hello", sources=["ddg"], topn=3)
    assert results[0].url == "https://cached.example"
    assert captured["key"]
