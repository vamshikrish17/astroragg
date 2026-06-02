from __future__ import annotations

import logging
from copy import copy
from dataclasses import dataclass
from pathlib import Path

import chromadb
from pypdf import PageObject
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[2]
PDF_DIR = BACKEND_DIR / "data" / "pdf"
PDF_FALLBACK_DIR = BACKEND_DIR / "data" / "pdfs"
VECTORSTORE_DIR = BACKEND_DIR / "vectorstore"
COLLECTION_NAME = "domain_documents"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


@dataclass(frozen=True)
class DocumentChunk:
    text: str
    source: str
    chunk_index: int


def resolve_pdf_dir() -> Path:
    """Return the configured PDF directory, with support for the repo's current folder name."""
    if PDF_DIR.exists():
        return PDF_DIR
    return PDF_FALLBACK_DIR


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from a PDF while allowing ingestion to continue on bad files."""
    try:
        reader = PdfReader(str(pdf_path))
        page_texts = []
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                page_texts.append(extract_page_text_in_reading_order(page))
            except Exception as exc:
                logger.warning(
                    "Skipping unreadable page %s in %s: %s",
                    page_number,
                    pdf_path.name,
                    exc,
                )
        return "\n".join(page_texts)
    except Exception as exc:
        logger.warning("Skipping unreadable PDF %s: %s", pdf_path.name, exc)
        return ""


def extract_page_text_in_reading_order(page: PageObject) -> str:
    """Extract normal pages directly and split wide two-up pages left-to-right."""
    width = float(page.mediabox.width)
    height = float(page.mediabox.height)
    if width <= height * 1.1:
        return page.extract_text() or ""

    left_page, right_page = split_page_vertically(page)
    left_text = left_page.extract_text() or ""
    right_text = right_page.extract_text() or ""
    return "\n".join(part for part in (left_text, right_text) if part.strip())


def split_page_vertically(page: PageObject) -> tuple[PageObject, PageObject]:
    """Create left and right cropped page copies for landscape/two-up PDFs."""
    left_page = copy(page)
    right_page = copy(page)

    lower_left_x = float(page.mediabox.left)
    lower_left_y = float(page.mediabox.bottom)
    upper_right_x = float(page.mediabox.right)
    upper_right_y = float(page.mediabox.top)
    midpoint_x = lower_left_x + ((upper_right_x - lower_left_x) / 2)

    left_page.mediabox.lower_left = (lower_left_x, lower_left_y)
    left_page.mediabox.upper_right = (midpoint_x, upper_right_y)
    left_page.cropbox.lower_left = (lower_left_x, lower_left_y)
    left_page.cropbox.upper_right = (midpoint_x, upper_right_y)

    right_page.mediabox.lower_left = (midpoint_x, lower_left_y)
    right_page.mediabox.upper_right = (upper_right_x, upper_right_y)
    right_page.cropbox.lower_left = (midpoint_x, lower_left_y)
    right_page.cropbox.upper_right = (upper_right_x, upper_right_y)

    return left_page, right_page


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks of roughly chunk_size characters."""
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


def load_pdf_chunks(pdf_dir: Path | None = None) -> list[DocumentChunk]:
    """Load every PDF and return source-aware chunks for vector indexing."""
    source_dir = pdf_dir or resolve_pdf_dir()
    if not source_dir.exists():
        logger.warning("PDF directory does not exist: %s", source_dir)
        return []

    chunks: list[DocumentChunk] = []
    for pdf_path in sorted(source_dir.glob("*.pdf")):
        text = extract_pdf_text(pdf_path)
        for index, chunk in enumerate(chunk_text(text)):
            chunks.append(
                DocumentChunk(
                    text=chunk,
                    source=pdf_path.name,
                    chunk_index=index,
                )
            )

    return chunks


def ingest_documents() -> dict[str, int | str]:
    """Generate embeddings for PDF chunks and persist them in ChromaDB."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    chunks = load_pdf_chunks()
    if not chunks:
        logger.warning("No PDF chunks were found to ingest.")
        return {"status": "empty", "chunks": 0, "source_dir": str(resolve_pdf_dir())}

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))

    # Recreate the collection so repeated ingestion mirrors the current PDF folder.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    texts = [chunk.text for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    collection.add(
        ids=[f"{chunk.source}-{chunk.chunk_index}" for chunk in chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {"source": chunk.source, "chunk_index": chunk.chunk_index}
            for chunk in chunks
        ],
    )

    logger.info("Ingested %s chunks into %s", len(chunks), VECTORSTORE_DIR)
    return {
        "status": "ok",
        "chunks": len(chunks),
        "source_dir": str(resolve_pdf_dir()),
        "vectorstore_dir": str(VECTORSTORE_DIR),
    }


if __name__ == "__main__":
    result = ingest_documents()
    print(result)
