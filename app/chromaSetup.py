"""
chromaSetup.py — all Chroma-related setup in one place.

What this module does:
1) Returns a single *embedding function* so index and query use the same model.
2) Returns a *persistent* Chroma vector store (collection) given a name + folder.
3) Hides the import differences between new/old langchain-chroma packages.
"""

from pathlib import Path
import os

from langchain_community.embeddings import  HuggingFaceEmbeddings

try:
    from langchain_chroma import Chroma
except Exception:
    from langchain_community.vectorstores import Chroma

def get_embedding():
    """
    Create and return an embedding function used for Both investigation and retrieval.
    We pick a strong multilingual model
    """
    model_name = os.getenv("EMBEDDING_MODEL")
    return HuggingFaceEmbeddings(model_name=model_name)

def get_vectorstore(collection_name: str, persist_directory: str) -> Chroma:
    """
        Creates (or gets) a persistent Chroma collection using our embedding.

        Args:
          collection_name: Logical bucket name inside Chroma (e.g., "handbook").
          persist_directory: Folder where Chroma stores its data on disk.

        Returns:
          A ready-to-use LangChain Chroma VectorStore instance.
        """
    Path(persist_directory).mkdir(parents=True, exist_ok=True)
    embed = get_embedding()
    return Chroma(collection_name=collection_name,
                  embedding_function=embed,
                  persist_directory=persist_directory
                  )