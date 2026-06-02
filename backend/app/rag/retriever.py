from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[2]
VECTORSTORE_DIR = BACKEND_DIR / "vectorstore"
COLLECTION_NAME = "domain_documents"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 15
CHROMA_CANDIDATE_MULTIPLIER = 4
SIMILARITY_THRESHOLD = 0.20
MMR_LAMBDA = 0.70
DOCUMENT_BOOST = 0.18
BOOSTED_DOCUMENTS = {
    "chandrayaan-3": "Chandrayaan-3.pdf",
    "chandrayaan-1": "Chandrayaan-1.pdf",
    "shape payload": "Chandrayaan-3.pdf",
    "shap payload": "Chandrayaan-3.pdf",
}


class DocumentRetriever:
    """Small wrapper around ChromaDB retrieval for the FastAPI layer."""

    def __init__(
        self,
        vectorstore_dir: Path = VECTORSTORE_DIR,
        collection_name: str = COLLECTION_NAME,
        model_name: str = EMBEDDING_MODEL_NAME,
    ) -> None:
        self.vectorstore_dir = vectorstore_dir
        self.collection_name = collection_name
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path=str(vectorstore_dir))
        self.collection = self.client.get_collection(collection_name)

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        document_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return thresholded, MMR-selected, de-duplicated, and re-ranked chunks."""
        expanded_query = expand_query(query)
        query_embedding = self.model.encode([expanded_query]).tolist()[0]
        where_filter = build_document_filter(document_name)
        candidate_count = min(
            self.collection.count(),
            max(top_k, top_k * CHROMA_CANDIDATE_MULTIPLIER),
        )
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": candidate_count,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            query_kwargs["where"] = where_filter

        results = self.collection.query(**query_kwargs)

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        documents, metadatas, distances = add_keyword_rescue_matches(
            query,
            documents,
            metadatas,
            distances,
            self.collection,
        )
        embeddings = get_candidate_embeddings(documents, query_embedding, self.model)

        matches = []
        for document, metadata, distance, embedding in zip(documents, metadatas, distances, embeddings):
            source = metadata.get("source", "unknown") if metadata else "unknown"
            similarity_score = cosine_distance_to_similarity(distance)
            boosted_similarity_score = apply_query_document_boost(
                query,
                source,
                similarity_score,
            )
            matches.append(
                {
                    "chunk": document,
                    "source": source,
                    "chunk_index": metadata.get("chunk_index") if metadata else None,
                    "distance": distance,
                    "similarity_score": boosted_similarity_score,
                    "original_similarity_score": similarity_score,
                    "embedding": embedding,
                }
            )

        ranked_chunks = rerank_chunks(matches, query_embedding, top_k=top_k)
        log_chunk_previews(ranked_chunks[:10])
        return [strip_internal_fields(chunk) for chunk in ranked_chunks]


def retrieve_relevant_chunks(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
    retriever = DocumentRetriever()
    return retriever.retrieve(query=query, top_k=top_k)


def expand_query(query: str) -> str:
    """Expand common acronym/typo queries so embedding search can find the right PDF."""
    query_lower = query.lower()
    expansions = []
    if "shap payload" in query_lower or "shape payload" in query_lower:
        expansions.append(
            "SHAPE payload Spectro-polarimetry of Habitable Planet Earth Chandrayaan-3"
        )

    if not expansions:
        return query

    return f"{query} {' '.join(expansions)}"


def cosine_distance_to_similarity(distance: float | int | None) -> float:
    """Convert Chroma cosine distance into a bounded similarity score."""
    if distance is None:
        return 0.0
    return max(0.0, min(1.0, 1.0 - float(distance)))


def rerank_chunks(
    chunks: list[dict[str, Any]],
    query_embedding: list[float],
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """Filter weak matches, score sources, and use MMR to balance relevance/diversity."""
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
    return maximal_marginal_relevance(similarity_ranked, query_embedding, top_k=top_k)


def calculate_source_scores(chunks: list[dict[str, Any]]) -> dict[str, float]:
    """Score each document using strongest chunk similarity plus supporting hits."""
    grouped_scores: dict[str, list[float]] = {}
    for chunk in chunks:
        source = chunk.get("source")
        if not source:
            continue
        grouped_scores.setdefault(source, []).append(chunk.get("similarity_score", 0.0))

    source_scores = {}
    for source, scores in grouped_scores.items():
        ordered_scores = sorted(scores, reverse=True)
        best_score = ordered_scores[0]
        support_score = sum(ordered_scores[1:4]) * 0.08
        source_scores[source] = round(min(1.0, best_score + support_score), 4)

    return source_scores


def remove_duplicate_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop near-duplicate chunks within the same document."""
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
    """Normalize chunk text for duplicate detection."""
    return " ".join(text.lower().split())


def is_near_duplicate(text: str, existing_text: str) -> bool:
    """Detect exact, contained, or heavily overlapping duplicate chunks."""
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
    query_embedding: list[float],
    top_k: int,
    lambda_mult: float = MMR_LAMBDA,
) -> list[dict[str, Any]]:
    """Select chunks that are relevant to the query without being redundant."""
    selected: list[dict[str, Any]] = []
    candidates = chunks.copy()

    while candidates and len(selected) < top_k:
        best_index = 0
        best_score = -math.inf
        for index, candidate in enumerate(candidates):
            embedding_relevance = max(
                0.0,
                cosine_similarity(query_embedding, candidate["embedding"]),
            )
            relevance = max(candidate.get("similarity_score", 0.0), embedding_relevance)
            if selected:
                diversity_penalty = max(
                    cosine_similarity(candidate["embedding"], item["embedding"])
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


def cosine_similarity(first: list[float], second: list[float]) -> float:
    """Calculate cosine similarity for MMR diversity scoring."""
    dot_product = sum(a * b for a, b in zip(first, second))
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm == 0 or second_norm == 0:
        return 0.0
    return dot_product / (first_norm * second_norm)


def get_candidate_embeddings(
    documents: list[str],
    query_embedding: list[float],
    model: SentenceTransformer,
) -> list[list[float]]:
    """Generate candidate embeddings used by MMR."""
    if not documents:
        return []
    return model.encode(documents).tolist()


def build_document_filter(document_name: str | None) -> dict[str, str] | None:
    """Build an exact Chroma metadata filter for source document name."""
    if not document_name:
        return None

    normalized_name = document_name.strip()
    if normalized_name and not normalized_name.lower().endswith(".pdf"):
        normalized_name = f"{normalized_name}.pdf"

    return {"source": normalized_name}


def apply_query_document_boost(query: str, source: str, similarity_score: float) -> float:
    """Boost known mission PDFs when the query explicitly names them."""
    query_lower = query.lower()
    for query_term, document_name in BOOSTED_DOCUMENTS.items():
        if query_term in query_lower and source == document_name:
            return min(1.0, similarity_score + DOCUMENT_BOOST)

    return similarity_score


def add_keyword_rescue_matches(
    query: str,
    documents: list[str],
    metadatas: list[dict[str, Any]],
    distances: list[float],
    collection: Any,
) -> tuple[list[str], list[dict[str, Any]], list[float]]:
    """Add exact keyword matches that semantic search can miss for rare acronyms."""
    query_lower = query.lower()
    if "shap payload" not in query_lower and "shape payload" not in query_lower:
        return documents, metadatas, distances

    existing_keys = {
        (metadata.get("source"), metadata.get("chunk_index"))
        for metadata in metadatas
        if metadata
    }
    items = collection.get(include=["documents", "metadatas"])

    for document, metadata in zip(items.get("documents", []), items.get("metadatas", [])):
        if not metadata:
            continue

        key = (metadata.get("source"), metadata.get("chunk_index"))
        document_lower = document.lower()
        has_shape_payload = "shape" in document_lower and "payload" in document_lower
        has_spectro = "spectro-polarimetry" in document_lower
        is_chandrayaan_3 = metadata.get("source") == "Chandrayaan-3.pdf"
        if key in existing_keys or not (is_chandrayaan_3 and (has_shape_payload or has_spectro)):
            continue

        documents.append(document)
        metadatas.append(metadata)
        distances.append(0.0)
        existing_keys.add(key)

    return documents, metadatas, distances


def strip_internal_fields(chunk: dict[str, Any]) -> dict[str, Any]:
    """Remove large internal fields before returning chunks to API callers."""
    public_chunk = chunk.copy()
    public_chunk.pop("embedding", None)
    return public_chunk


def log_chunk_previews(chunks: list[dict[str, Any]]) -> None:
    """Log the top retrieved chunks with similarity scores for debugging."""
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
