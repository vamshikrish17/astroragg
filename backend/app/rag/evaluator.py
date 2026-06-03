from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.rag.generator import RagAnswerGenerator
from app.rag.retriever import DocumentRetriever


BACKEND_DIR = Path(__file__).resolve().parents[2]
QUESTIONS_FILE = BACKEND_DIR / "evaluation" / "questions.txt"
RESULTS_FILE = BACKEND_DIR / "evaluation" / "results.csv"
SUMMARY_FILE = BACKEND_DIR / "evaluation" / "evaluation_summary.json"
RESULT_COLUMNS = [
    "question",
    "generated_answer",
    "source_document",
    "retrieval_score",
    "exact_match",
    "answer_length",
    "latency_seconds",
    "source_count",
]


class EvaluationError(RuntimeError):
    """Raised when the evaluation setup is invalid."""


@dataclass(frozen=True)
class EvaluationQuestion:
    question: str
    expected_answer: str | None = None


def read_evaluation_questions(questions_file: Path = QUESTIONS_FILE) -> list[EvaluationQuestion]:
    if not questions_file.exists():
        raise EvaluationError(f"Questions file was not found: {questions_file}")

    questions = [
        parse_question_line(line.strip())
        for line in questions_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not questions:
        raise EvaluationError(f"No questions were found in: {questions_file}")

    return questions


def run_rag_evaluation(
    retriever: DocumentRetriever,
    generator: RagAnswerGenerator,
    questions_file: Path = QUESTIONS_FILE,
    results_file: Path = RESULTS_FILE,
    summary_file: Path = SUMMARY_FILE,
) -> dict[str, Any]:
    questions = read_evaluation_questions(questions_file)
    rows = []
    retrieval_scores = []
    latencies = []
    answer_lengths = []
    exact_match_values = []
    successful_retrievals = 0
    failed_retrievals = 0

    for evaluation_question in questions:
        row = evaluate_question(evaluation_question, retriever, generator)
        rows.append(row)
        latencies.append(row["latency_seconds"])
        answer_lengths.append(row["answer_length"])

        score = row["retrieval_score"]
        if score > 0:
            successful_retrievals += 1
            retrieval_scores.append(score)
        else:
            failed_retrievals += 1

        if row["exact_match"] != "":
            exact_match_values.append(bool(row["exact_match"]))

    results_file.parent.mkdir(parents=True, exist_ok=True)
    with results_file.open("w", newline="", encoding="utf-8") as results_handle:
        writer = csv.DictWriter(results_handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    average_score = round(sum(retrieval_scores) / len(retrieval_scores), 4) if retrieval_scores else 0.0
    exact_match_count = sum(1 for value in exact_match_values if value)
    exact_match_rate = (
        round(exact_match_count / len(exact_match_values), 4)
        if exact_match_values
        else 0.0
    )

    summary = {
        "total_questions": len(questions),
        "average_retrieval_score": average_score,
        "average_latency_seconds": round(sum(latencies) / len(latencies), 4),
        "average_answer_length": round(sum(answer_lengths) / len(answer_lengths), 2),
        "successful_retrievals": successful_retrievals,
        "failed_retrievals": failed_retrievals,
        "exact_match_count": exact_match_count,
        "exact_match_rate": exact_match_rate,
        "results_file": str(results_file),
        "summary_file": str(summary_file),
    }

    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def evaluate_question(
    evaluation_question: EvaluationQuestion,
    retriever: DocumentRetriever,
    generator: RagAnswerGenerator,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    question = evaluation_question.question

    try:
        retrieved_chunks = retriever.retrieve(question)
    except Exception:
        latency_seconds = elapsed_seconds(start_time)
        return build_result_row(
            evaluation_question,
            "Retrieval failed.",
            "",
            0.0,
            latency_seconds,
            0,
        )

    if not retrieved_chunks:
        latency_seconds = elapsed_seconds(start_time)
        return build_result_row(
            evaluation_question,
            "I could not find that information in the documents.",
            "",
            0.0,
            latency_seconds,
            0,
        )

    source_document = get_top_source_document(retrieved_chunks)
    retrieval_score = get_top_retrieval_score(retrieved_chunks)
    source_count = get_source_count(retrieved_chunks)

    try:
        generated_answer = generator.generate_answer(question, retrieved_chunks)
    except Exception:
        generated_answer = "Answer generation failed."

    latency_seconds = elapsed_seconds(start_time)
    return build_result_row(
        evaluation_question,
        generated_answer,
        source_document,
        retrieval_score,
        latency_seconds,
        source_count,
    )


def parse_question_line(line: str) -> EvaluationQuestion:
    for delimiter in ("\t", "||", "::"):
        if delimiter in line:
            question, expected_answer = line.split(delimiter, 1)
            return EvaluationQuestion(
                question=question.strip(),
                expected_answer=expected_answer.strip() or None,
            )

    return EvaluationQuestion(question=line)


def get_top_source_document(retrieved_chunks: list[dict[str, Any]]) -> str:
    source_scores: dict[str, float] = {}
    for chunk in retrieved_chunks:
        source = chunk.get("source")
        if not source:
            continue
        score = float(chunk.get("source_score", chunk.get("similarity_score", 0.0)) or 0.0)
        source_scores[source] = max(source_scores.get(source, 0.0), score)

    ranked_sources = sorted(source_scores.items(), key=lambda item: -item[1])
    return "; ".join(source for source, _score in ranked_sources[:3])


def get_top_retrieval_score(retrieved_chunks: list[dict[str, Any]]) -> float:
    top_chunk = retrieved_chunks[0]
    score = top_chunk.get("source_score", top_chunk.get("similarity_score", 0.0))
    return round(float(score or 0.0), 4)


def get_source_count(retrieved_chunks: list[dict[str, Any]]) -> int:
    return len({chunk.get("source") for chunk in retrieved_chunks if chunk.get("source")})


def calculate_exact_match(generated_answer: str, expected_answer: str | None) -> bool | str:
    if expected_answer is None:
        return ""

    return normalize_answer(generated_answer) == normalize_answer(expected_answer)


def normalize_answer(answer: str) -> str:
    return " ".join(answer.lower().split())


def elapsed_seconds(start_time: float) -> float:
    return round(time.perf_counter() - start_time, 4)


def build_result_row(
    evaluation_question: EvaluationQuestion,
    generated_answer: str,
    source_document: str,
    retrieval_score: float,
    latency_seconds: float,
    source_count: int,
) -> dict[str, Any]:
    return {
        "question": evaluation_question.question,
        "generated_answer": generated_answer,
        "source_document": source_document,
        "retrieval_score": retrieval_score,
        "exact_match": calculate_exact_match(
            generated_answer,
            evaluation_question.expected_answer,
        ),
        "answer_length": len(generated_answer),
        "latency_seconds": latency_seconds,
        "source_count": source_count,
    }
