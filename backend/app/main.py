from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.rag.evaluator import EvaluationError, run_rag_evaluation
from app.rag.generator import (
    GroqGenerationError,
    MissingGroqApiKeyError,
    RagAnswerGenerator,
    source_scores,
)
from app.rag.retriever import DEFAULT_TOP_K, DocumentRetriever


app = FastAPI(title="Domain RAG Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RetrieveRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question to retrieve context for.")


class RetrievedChunk(BaseModel):
    chunk: str
    source: str
    chunk_index: int | None = None
    distance: float | None = None
    similarity_score: float | None = None
    source_score: float | None = None


class RetrieveResponse(BaseModel):
    question: str
    results: list[RetrievedChunk]


class SourceScore(BaseModel):
    document: str
    score: float


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceScore]
    debug: dict[str, Any] | None = None


class EvaluationSummary(BaseModel):
    total_questions: int
    average_retrieval_score: float
    average_latency_seconds: float
    average_answer_length: float
    successful_retrievals: int
    failed_retrievals: int
    exact_match_count: int
    exact_match_rate: float
    results_file: str
    summary_file: str


@lru_cache(maxsize=1)
def get_retriever() -> DocumentRetriever:
    """Load the embedding model and Chroma collection once per server process."""
    return DocumentRetriever()


@lru_cache(maxsize=1)
def get_generator() -> RagAnswerGenerator:
    """Load the Groq client once per server process."""
    return RagAnswerGenerator()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(
    request: RetrieveRequest,
    document_name: str | None = Query(None),
) -> RetrieveResponse:
    try:
        results = get_retriever().retrieve(
            request.question,
            top_k=DEFAULT_TOP_K,
            document_name=document_name,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Vector store is not ready. Run ingestion before retrieval.",
        ) from exc

    return RetrieveResponse(question=request.question, results=results)


@app.post("/ask", response_model=AskResponse, response_model_exclude_none=True)
def ask(
    request: RetrieveRequest,
    debug: bool = Query(False),
    document_name: str | None = Query(None),
) -> AskResponse:
    """Retrieve document context and ask Groq for a grounded answer."""
    try:
        retrieved_chunks = get_retriever().retrieve(
            request.question,
            top_k=DEFAULT_TOP_K,
            document_name=document_name,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Vector store is not ready. Run ingestion before asking questions.",
        ) from exc

    if not retrieved_chunks:
        raise HTTPException(
            status_code=404,
            detail="No relevant document chunks were found for this question.",
        )

    try:
        answer = get_generator().generate_answer(request.question, retrieved_chunks)
    except MissingGroqApiKeyError as exc:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY environment variable is not set.",
        ) from exc
    except GroqGenerationError as exc:
        raise HTTPException(
            status_code=502,
            detail="Groq failed to generate an answer. Please try again later.",
        ) from exc

    debug_payload = build_debug_payload(retrieved_chunks) if debug else None
    return AskResponse(
        question=request.question,
        answer=answer,
        sources=source_scores(retrieved_chunks),
        debug=debug_payload,
    )


@app.post("/evaluate", response_model=EvaluationSummary)
def evaluate() -> EvaluationSummary:
    """Run batch RAG evaluation from evaluation/questions.txt."""
    try:
        summary = run_rag_evaluation(get_retriever(), get_generator())
    except EvaluationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except MissingGroqApiKeyError as exc:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY environment variable is not set.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Evaluation failed. Check server logs for details.",
        ) from exc

    return EvaluationSummary(**summary)


def build_debug_payload(retrieved_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Return retrieved chunk details only when /ask is called with debug=true."""
    return {
        "retrieved_chunks": [
            {
                "chunk": chunk.get("chunk", ""),
                "similarity_score": chunk.get("similarity_score", 0.0),
                "source_document": chunk.get("source", "unknown"),
                "chunk_index": chunk.get("chunk_index"),
            }
            for chunk in retrieved_chunks
        ]
    }
