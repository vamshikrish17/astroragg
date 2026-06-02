from __future__ import annotations

import logging
from pathlib import Path

from app.rag.ingest import extract_pdf_text, resolve_pdf_dir


BACKEND_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BACKEND_DIR / "debug_extraction"
PREVIEW_CHARS = 1000


def debug_pdf_extraction() -> None:
    """Extract every PDF exactly like ingestion and save one text file per PDF."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    pdf_dir = resolve_pdf_dir()
    if not pdf_dir.exists():
        print(f"PDF directory not found: {pdf_dir}")
        return

    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDF files found in: {pdf_dir}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Reading PDFs from: {pdf_dir}")
    print(f"Writing extracted text to: {OUTPUT_DIR}")

    for pdf_path in pdf_paths:
        text = extract_pdf_text(pdf_path)
        output_path = OUTPUT_DIR / f"{pdf_path.stem}.txt"
        output_path.write_text(text, encoding="utf-8")

        print("\n" + "=" * 80)
        print(f"PDF: {pdf_path.name}")
        print(f"TXT: {output_path}")
        print("-" * 80)
        print(text[:PREVIEW_CHARS])


if __name__ == "__main__":
    debug_pdf_extraction()
