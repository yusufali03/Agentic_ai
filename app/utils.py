"""
utils.py  - small helpers: PDF text extraction, chunking, and formatting
"""
from typing import Iterable, List
from pathlib import Path
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sympy.utilities.iterables import sequence_partitions_empty


def extract_pdf_pages(pdf_path: Path) -> Iterable[Document]:
    """
        Yields Document objects, one per page.
        - Extracts selectable text from each page (no OCR here).
        - Cleans hyphenation and newlines for better embeddings.
        Metadata: {"source": filename, "page": page_number}
        """
    reader = PdfReader(pdf_path)
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.replace("-\n", "").replace("\n", " ").strip()
        if text:
            yield Document(page_content=text, metadata={"source": pdf_path.name, "page": i})


def chunk_documents(
        docs: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
) -> List[Document]:
    """
        Splits Documents into semantically stable chunks.
        - chunk_size ~ 1000 chars (~600–800 tokens)
        - overlap keeps topic continuity across chunk boundaries.
        """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # seperator=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)

def format_citations(docs: List[Document]) -> str:
    """
        Turns retrieved docs into a single string block with inline citations.
        Example header: [ModuleHandbook.pdf p.12]
        """
    parts = []
    for d in docs:
        src = d.metadata.get("source","?")
        pg = d.metadata.get("page","?")
        parts.append(f"[{src} p.{pg}]\n{d.page_content}")
    return "\n\n---\n".join(parts)