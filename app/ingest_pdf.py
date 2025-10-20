"""
ingest_pdf.py — one function to ingest a PDF into Chroma.

Steps:
1) Extract page texts → Document objects with metadata.
2) Chunk into ~1000-char chunks with overlap.
3) Add to persistent Chroma vector store and persist.
"""

from pathlib import Path
from typing import Optional
from langchain_core.documents import Document
from .chromaSetup import get_vectorstore
from .utils import extract_pdf_pages, chunk_documents

def ingest_pdf(
        pdf_path: str,
        collection_name: str="nadbook",
        persist_directory: str="chroma_store/handbook",
        chunk_size: int=1000,
        chunk_overlap: int=150,
) -> int:
    """
       Ingests a PDF into a Chroma collection.
       Returns how many chunks were added.
       """
    p = Path(pdf_path)
    assert p.exists(), f"File not found: {p}"
    page_docs= list(extract_pdf_pages(p))
    if not page_docs:
        raise RuntimeError("No selectable text found. If it's scanned, you need OCR")

    chunks = chunk_documents(page_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    vectordb = get_vectorstore(collection_name, persist_directory)
    vectordb.add_documents(chunks)
    # vectordb.persist()
    return len(chunks)