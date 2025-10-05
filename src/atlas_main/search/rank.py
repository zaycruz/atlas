"""Ranking utilities combining lexical and embedding signals."""
from __future__ import annotations

import math
from collections import Counter
from typing import Callable, Dict, Iterable, List, Optional
from urllib.parse import urlparse

from . import RankedDoc, SearchResult

EmbedFn = Callable[[str], Optional[Iterable[float]]]

DEFAULT_WEIGHTS = {"lexical": 0.5, "embed": 0.4, "recency": 0.1}


def _tokenize(text: str) -> List[str]:
    return [token.lower() for token in text.split() if token]


def bm25_score(query: str, text: str, *, k1: float = 1.5, b: float = 0.75) -> float:
    query_tokens = _tokenize(query)
    doc_tokens = _tokenize(text)
    if not query_tokens or not doc_tokens:
        return 0.0
    doc_len = len(doc_tokens)
    avg_doc_len = max(doc_len, 1)
    doc_freq = Counter(doc_tokens)
    score = 0.0
    for token in set(query_tokens):
        freq = doc_freq.get(token, 0)
        if freq == 0:
            continue
        numerator = freq * (k1 + 1)
        denominator = freq + k1 * (1 - b + b * (doc_len / avg_doc_len))
        score += math.log(1 + len(doc_tokens)) * numerator / denominator
    return score


def _cosine_similarity(vec_a: Iterable[float], vec_b: Iterable[float]) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


def _embed_similarity(query: str, candidate: SearchResult, embed_fn: Optional[EmbedFn]) -> float:
    if not embed_fn:
        return 0.0
    query_vec = embed_fn(query)
    candidate_vec = embed_fn(f"{candidate.title}\n{candidate.snippet}")
    if not query_vec or not candidate_vec:
        return 0.0
    return _cosine_similarity(query_vec, candidate_vec)


def _recency_score(result: SearchResult) -> float:
    published = result.published_at or ""
    if not published:
        return 0.0
    # naive heuristic: newer strings (year within last 2 years) get small boost
    for year in range(2030, 1990, -1):
        if str(year) in published:
            return max(0.0, (year - 2015) * 0.02)
    return 0.0


def _domain_penalty(result: SearchResult, seen_domains: Dict[str, int]) -> float:
    domain = urlparse(result.url).netloc
    count = seen_domains.get(domain, 0)
    penalty = 0.1 * count
    seen_domains[domain] = count + 1
    return penalty


def rank_results(
    query: str,
    results: List[SearchResult],
    *,
    embed_fn: Optional[EmbedFn] = None,
    weights: Optional[Dict[str, float]] = None,
) -> List[RankedDoc]:
    if weights is None:
        weights = DEFAULT_WEIGHTS
    seen_domains: Dict[str, int] = {}
    ranked: List[RankedDoc] = []
    for result in results:
        lexical = bm25_score(query, f"{result.title} {result.snippet}")
        semantic = _embed_similarity(query, result, embed_fn)
        recency = _recency_score(result)
        penalty = _domain_penalty(result, seen_domains)
        score = (
            weights.get("lexical", 0.0) * lexical
            + weights.get("embed", 0.0) * semantic
            + weights.get("recency", 0.0) * recency
            - penalty
        )
        reason = f"lex={lexical:.2f}, emb={semantic:.2f}, rec={recency:.2f}, penalty={penalty:.2f}"
        ranked.append(
            RankedDoc(
                url=result.url,
                title=result.title,
                score=score,
                reason=reason,
                source=result.source,
                snippet=result.snippet,
                published_at=result.published_at,
            )
        )
    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked


__all__ = ["bm25_score", "rank_results", "DEFAULT_WEIGHTS"]
