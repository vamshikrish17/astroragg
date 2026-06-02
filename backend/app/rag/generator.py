from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from groq import Groq, GroqError


NOT_FOUND_ANSWER = "I could not find that information in the documents."
SYSTEM_PROMPT = (
    "You are a RAG assistant. Answer only from the provided context. "
    "Combine relevant details across multiple context chunks when needed. "
    "Only say 'I could not find that information in the documents.' when the "
    "provided context has no relevant information for the question."
)
PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"


class MissingGroqApiKeyError(RuntimeError):
    """Raised when GROQ_API_KEY is not configured."""


class GroqGenerationError(RuntimeError):
    """Raised when Groq cannot generate an answer."""


class RagAnswerGenerator:
    """Generate grounded answers from retrieved chunks using the Groq API."""

    def __init__(
        self,
        api_key: str | None = None,
        primary_model: str = PRIMARY_MODEL,
        fallback_model: str = FALLBACK_MODEL,
    ) -> None:
        load_dotenv()
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise MissingGroqApiKeyError("GROQ_API_KEY environment variable is not set.")

        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.client = Groq(api_key=self.api_key)

    def generate_answer(self, question: str, retrieved_chunks: list[dict[str, Any]]) -> str:
        """Use retrieved context to answer a user question."""
        context = build_context(retrieved_chunks)
        if not context:
            return NOT_FOUND_ANSWER

        user_prompt = (
            "Context:\n"
            f"{context}\n\n"
            "Question:\n"
            f"{question}\n\n"
            "Answer:"
        )

        try:
            return self._chat_completion(self.primary_model, user_prompt)
        except GroqError:
            try:
                return self._chat_completion(self.fallback_model, user_prompt)
            except GroqError as exc:
                raise GroqGenerationError("Groq failed to generate an answer.") from exc

    def _chat_completion(self, model: str, user_prompt: str) -> str:
        """Call Groq chat completions and return the generated message text."""
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )

        answer = response.choices[0].message.content
        return answer.strip() if answer else ""


def build_context(retrieved_chunks: list[dict[str, Any]]) -> str:
    """Combine retrieved chunks into one context block for the LLM."""
    context_parts = []
    for index, item in enumerate(retrieved_chunks, start=1):
        source = item.get("source", "unknown")
        chunk = item.get("chunk", "").strip()
        if chunk:
            context_parts.append(f"[{index}] Source: {source}\n{chunk}")

    return "\n\n".join(context_parts)


def unique_sources(retrieved_chunks: list[dict[str, Any]]) -> list[str]:
    """Return source document names once, preserving retrieval order."""
    sources = []
    seen = set()
    for item in retrieved_chunks:
        source = item.get("source")
        if source and source not in seen:
            seen.add(source)
            sources.append(source)

    return sources


def source_scores(retrieved_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the top three source documents used in context, ranked by score."""
    best_scores: dict[str, float] = {}
    for item in retrieved_chunks:
        source = item.get("source")
        if not source:
            continue

        score = float(item.get("source_score", item.get("similarity_score", 0.0)) or 0.0)
        best_scores[source] = max(best_scores.get(source, 0.0), score)

    ranked_sources = sorted(best_scores.items(), key=lambda item: -item[1])
    return [
        {"document": source, "score": round(score, 4)}
        for source, score in ranked_sources[:3]
    ]
