from __future__ import annotations

from atlas_main.search import SearchResult
from atlas_main.search import rank


def test_rank_results_prefers_lexical_match() -> None:
    results = [
        SearchResult(title="Cats and Dogs", url="https://example.com/a", snippet="A story about pets", source="ddg"),
        SearchResult(title="Cats", url="https://example.com/b", snippet="Cats only", source="ddg"),
        SearchResult(title="Unrelated", url="https://example.com/c", snippet="Nothing relevant", source="ddg"),
    ]

    ranked = rank.rank_results("cats", results, embed_fn=None)
    assert ranked[0].url in {"https://example.com/a", "https://example.com/b"}
    assert ranked[0].score >= ranked[-1].score


def test_rank_results_penalizes_duplicates() -> None:
    results = [
        SearchResult(title="Item 1", url="https://a.com/1", snippet="alpha", source="ddg"),
        SearchResult(title="Item 2", url="https://a.com/2", snippet="beta", source="ddg"),
        SearchResult(title="Item 3", url="https://b.com/3", snippet="gamma", source="ddg"),
    ]

    ranked = rank.rank_results("item", results, embed_fn=None)
    a_scores = [doc.score for doc in ranked if doc.url.startswith("https://a.com")]
    b_score = [doc.score for doc in ranked if doc.url.startswith("https://b.com")][0]
    assert max(a_scores) - min(a_scores) < 5  # penalty applied but not huge
    assert b_score >= min(a_scores)
