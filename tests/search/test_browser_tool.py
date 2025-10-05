from __future__ import annotations

from datetime import datetime

from atlas_main.search import PageContent, RankedDoc, SearchResult
from atlas_main.search import engines, rank
from atlas_main.tools_browser import BrowserSession


class _LogCapture:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def __call__(self, event: str, data: dict) -> None:
        self.events.append((event, data))


def test_browser_session_search_and_open(monkeypatch) -> None:
    query = "atlas search"
    results = [
        SearchResult(title="Atlas", url="https://example.com/a", snippet="About Atlas", source="ddg"),
        SearchResult(title="Docs", url="https://example.com/b", snippet="Documentation", source="ddg"),
    ]

    def fake_federated(*args, **kwargs):
        return results

    monkeypatch.setattr(engines, "federated_search", fake_federated)

    def fake_rank(query: str, results, **kwargs):
        ranked = []
        for idx, item in enumerate(results):
            ranked.append(RankedDoc(
                url=item.url,
                title=item.title,
                score=10 - idx,
                reason="test",
                source=item.source,
                snippet=item.snippet,
                published_at=item.published_at,
            ))
        return ranked

    monkeypatch.setattr(rank, "rank_results", fake_rank)

    def fake_fetch(url: str, **kwargs):
        return PageContent(
            url=url,
            text=f"Content for {url}",
            html=f"<html><body><a href=\"{url}/more\">More</a></body></html>",
            meta={"title": f"Title {url}"},
            fetched_at=datetime.utcnow(),
        )

    import atlas_main.tools_browser as tools_browser

    monkeypatch.setattr(tools_browser, "fetch", fake_fetch)

    logger = _LogCapture()
    session = BrowserSession(embed_fn=None, logger=logger)

    out = session.search(query=query, topn=2, budget_tokens=400)
    assert "pageText" in out
    assert "[0]" in out["pageText"]

    opened = session.open(id=0)
    assert "Content for" in opened["pageText"]
    assert any(event == "search.query" for event, _ in logger.events)
    assert any(event == "crawl.fetch" for event, _ in logger.events)
