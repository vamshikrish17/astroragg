from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi


logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[2]
DEBUG_EXTRACTION_DIR = BACKEND_DIR / "debug_extraction"
DEFAULT_TOP_K = 15
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
CHROMA_CANDIDATE_MULTIPLIER = 4
SIMILARITY_THRESHOLD = 0.15
MMR_LAMBDA = 0.70
DOCUMENT_BOOST = 0.18
BOOSTED_DOCUMENTS = {
    "chandrayaan-3": "Chandrayaan-3.pdf",
    "chandrayaan-1": "Chandrayaan-1.pdf",
    "shape payload": "Chandrayaan-3.pdf",
    "shap payload": "Chandrayaan-3.pdf",
}


@dataclass(frozen=True)
class IndexedChunk:
    text: str
    source: str
    chunk_index: int
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class SearchIndex:
    chunks: list[IndexedChunk]
    tokenized_chunks: list[list[str]]
    bm25: BM25Okapi


class DocumentRetriever:
    """Lightweight BM25 retriever that works well in Vercel serverless."""

    def __init__(self, corpus_dir: Path = DEBUG_EXTRACTION_DIR) -> None:
        self.corpus_dir = corpus_dir
        self.index = load_search_index(corpus_dir)

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        document_name: str | None = None,
    ) -> list[dict[str, Any]]:
        expanded_query = expand_query(query)
        query_tokens = tokenize(expanded_query)
        if not query_tokens or not self.index.chunks:
            return []

        candidate_chunks = self.index.chunks
        if document_name:
            candidate_chunks = filter_chunks_by_document(candidate_chunks, document_name)
            if not candidate_chunks:
                return []

        scored_matches = score_chunks(self.index, candidate_chunks, query_tokens)
        rescued_matches = add_keyword_rescue_matches(query, scored_matches, candidate_chunks)
        ranked_chunks = rerank_chunks(
            rescued_matches,
            query_tokens,
            top_k=top_k,
        )
        log_chunk_previews(ranked_chunks[:10])
        return [strip_internal_fields(chunk) for chunk in ranked_chunks]


@lru_cache(maxsize=1)
def load_search_index(corpus_dir: Path = DEBUG_EXTRACTION_DIR) -> SearchIndex:
    chunks = load_corpus_chunks(corpus_dir)
    tokenized_chunks = [list(chunk.tokens) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks) if tokenized_chunks else BM25Okapi([[]])
    return SearchIndex(chunks=chunks, tokenized_chunks=tokenized_chunks, bm25=bm25)


def retrieve_relevant_chunks(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
    retriever = DocumentRetriever()
    return retriever.retrieve(query=query, top_k=top_k)


def expand_query(query: str) -> str:
    query_lower = query.lower()
    expansions = []
    if "shap payload" in query_lower or "shape payload" in query_lower:
        expansions.append(
            "SHAPE payload Spectro-polarimetry of Habitable Planet Earth Chandrayaan-3"
        )

    if not expansions:
        return query

    return f"{query} {' '.join(expansions)}"


def load_corpus_chunks(corpus_dir: Path) -> list[IndexedChunk]:
    if not corpus_dir.exists():
        logger.warning("Corpus directory does not exist: %s", corpus_dir)
        return []

    chunks: list[IndexedChunk] = []
    for text_file in sorted(corpus_dir.glob("*.txt")):
        source = f"{text_file.stem}.pdf"
        text = text_file.read_text(encoding="utf-8", errors="ignore")
        for chunk_index, chunk in enumerate(chunk_text(text)):
            tokens = tuple(tokenize(chunk))
            if not tokens:
                continue
            chunks.append(
                IndexedChunk(
                    text=chunk,
                    source=source,
                    chunk_index=chunk_index,
                    tokens=tokens,
                )
            )

    return chunks


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    clean_text = " ".join(text.split())
    if not clean_text:
        return []

    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)

    while start < len(clean_text):
        end = start + chunk_size
        chunk = clean_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def score_chunks(
    index: SearchIndex,
    chunks: list[IndexedChunk],
    query_tokens: list[str],
) -> list[dict[str, Any]]:
    if not chunks:
        return []

    raw_scores = index.bm25.get_scores(query_tokens)
    chunk_to_score = {
        indexed_chunk: float(raw_scores[i])
        for i, indexed_chunk in enumerate(index.chunks)
    }

    matching_scores = [chunk_to_score.get(chunk, 0.0) for chunk in chunks]
    max_raw_score = max(matching_scores, default=0.0)
    if max_raw_score <= 0:
        max_raw_score = 1.0

    matches: list[dict[str, Any]] = []
    for chunk in chunks:
        raw_score = chunk_to_score.get(chunk, 0.0)
        similarity_score = max(0.0, min(1.0, raw_score / max_raw_score))
        boosted_similarity_score = apply_query_document_boost(
            " ".join(query_tokens),
            chunk.source,
            similarity_score,
        )
        matches.append(
            {
                "chunk": chunk.text,
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "distance": round(max(0.0, 1.0 - boosted_similarity_score), 4),
                "similarity_score": boosted_similarity_score,
                "original_similarity_score": similarity_score,
                "tokens": list(chunk.tokens),
            }
        )

    return matches


def filter_chunks_by_document(
    chunks: list[IndexedChunk],
    document_name: str,
) -> list[IndexedChunk]:
    normalized_name = normalize_document_name(document_name)
    return [chunk for chunk in chunks if chunk.source == normalized_name]


def normalize_document_name(document_name: str) -> str:
    normalized_name = document_name.strip()
    if normalized_name and not normalized_name.lower().endswith(".pdf"):
        normalized_name = f"{normalized_name}.pdf"
    return normalized_name


def add_keyword_rescue_matches(
    query: str,
    matches: list[dict[str, Any]],
    all_chunks: list[IndexedChunk],
) -> list[dict[str, Any]]:
    query_lower = query.lower()
    if "shap payload" not in query_lower and "shape payload" not in query_lower:
        return matches

    seen_keys = {
        (match.get("source"), match.get("chunk_index"))
        for match in matches
    }
    rescue_score = max(
        (float(match.get("similarity_score", 0.0)) for match in matches),
        default=1.0,
    )

    for chunk in all_chunks:
        if chunk.source != "Chandrayaan-3.pdf":
            continue

        lowered = chunk.text.lower()
        has_shape_payload = "shape" in lowered and "payload" in lowered
        has_spectro = "spectro-polarimetry" in lowered
        if not (has_shape_payload or has_spectro):
            continue

        key = (chunk.source, chunk.chunk_index)
        if key in seen_keys:
            continue

        matches.append(
            {
                "chunk": chunk.text,
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "distance": 0.0,
                "similarity_score": rescue_score,
                "original_similarity_score": rescue_score,
                "tokens": list(chunk.tokens),
            }
        )
        seen_keys.add(key)

    return matches


def rerank_chunks(
    chunks: list[dict[str, Any]],
    query_tokens: list[str],
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    filtered = [
        chunk
        for chunk in chunks
        if chunk.get("similarity_score", 0.0) >= SIMILARITY_THRESHOLD
    ]
    deduped = remove_duplicate_chunks(filtered)
    source_scores = calculate_source_scores(deduped)
    if not source_scores:
        return []

    scored_chunks = [
        {
            **chunk,
            "source_score": source_scores.get(chunk.get("source"), 0.0),
        }
        for chunk in deduped
    ]

    similarity_ranked = sorted(
        scored_chunks,
        key=lambda chunk: (
            -chunk.get("source_score", 0.0),
            -chunk.get("similarity_score", 0.0),
            chunk.get("chunk_index") if chunk.get("chunk_index") is not None else 10**9,
        ),
    )
    return maximal_marginal_relevance(similarity_ranked, set(query_tokens), top_k=top_k)


def calculate_source_scores(chunks: list[dict[str, Any]]) -> dict[str, float]:
    grouped_scores: dict[str, list[float]] = {}
    for chunk in chunks:
        source = chunk.get("source")
        if not source:
            continue
        grouped_scores.setdefault(source, []).append(float(chunk.get("similarity_score", 0.0)))

    source_scores = {}
    for source, scores in grouped_scores.items():
        ordered_scores = sorted(scores, reverse=True)
        best_score = ordered_scores[0]
        support_score = sum(ordered_scores[1:4]) * 0.08
        source_scores[source] = round(min(1.0, best_score + support_score), 4)

    return source_scores


def remove_duplicate_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    seen_by_source: dict[str, list[str]] = {}

    for chunk in sorted(chunks, key=lambda item: -item.get("similarity_score", 0.0)):
        source = chunk.get("source", "unknown")
        text = normalize_text(chunk.get("chunk", ""))
        if not text:
            continue

        existing_texts = seen_by_source.setdefault(source, [])
        if any(is_near_duplicate(text, existing) for existing in existing_texts):
            continue

        existing_texts.append(text)
        kept.append(chunk)

    return kept


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def is_near_duplicate(text: str, existing_text: str) -> bool:
    if text == existing_text or text in existing_text or existing_text in text:
        return True

    text_words = set(text.split())
    existing_words = set(existing_text.split())
    if not text_words or not existing_words:
        return False

    overlap = len(text_words & existing_words) / min(len(text_words), len(existing_words))
    return overlap >= 0.90


def maximal_marginal_relevance(
    chunks: list[dict[str, Any]],
    query_tokens: set[str],
    top_k: int,
    lambda_mult: float = MMR_LAMBDA,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    candidates = chunks.copy()

    while candidates and len(selected) < top_k:
        best_index = 0
        best_score = -math.inf
        for index, candidate in enumerate(candidates):
            candidate_tokens = set(candidate.get("tokens", []))
            keyword_relevance = jaccard_similarity(query_tokens, candidate_tokens)
            relevance = max(candidate.get("similarity_score", 0.0), keyword_relevance)
            if selected:
                diversity_penalty = max(
                    jaccard_similarity(candidate_tokens, set(item.get("tokens", [])))
                    for item in selected
                )
            else:
                diversity_penalty = 0.0

            mmr_score = (lambda_mult * relevance) - ((1.0 - lambda_mult) * diversity_penalty)
            if mmr_score > best_score:
                best_score = mmr_score
                best_index = index

        selected.append(candidates.pop(best_index))

    return selected


def jaccard_similarity(first: set[str], second: set[str]) -> float:
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)


def build_document_filter(document_name: str | None) -> dict[str, str] | None:
    if not document_name:
        return None
    return {"source": normalize_document_name(document_name)}


def apply_query_document_boost(query: str, source: str, similarity_score: float) -> float:
    query_lower = query.lower()
    for query_term, document_name in BOOSTED_DOCUMENTS.items():
        if query_term in query_lower and source == document_name:
            return min(1.0, similarity_score + DOCUMENT_BOOST)

    return similarity_score


def strip_internal_fields(chunk: dict[str, Any]) -> dict[str, Any]:
    public_chunk = chunk.copy()
    public_chunk.pop("tokens", None)
    return public_chunk


def log_chunk_previews(chunks: list[dict[str, Any]]) -> None:
    if not logger.isEnabledFor(logging.INFO):
        return

    for index, chunk in enumerate(chunks, start=1):
        preview = " ".join(chunk.get("chunk", "").split())[:180]
        logger.info(
            "Retrieved chunk %s source=%s source_score=%.4f similarity=%.4f preview=%s",
            index,
            chunk.get("source", "unknown"),
            float(chunk.get("source_score", 0.0)),
            float(chunk.get("similarity_score", 0.0)),
            preview,
        )
