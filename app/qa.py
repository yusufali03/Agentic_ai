"""
qa.py — retrieve top-k chunks from Chroma and ask the LLM to answer with citations.

RAG flow:
User question -> Vector search (Chroma retriever) -> Build prompt with context -> LLM -> Answer
"""
from typing import Optional


from .chromaSetup import get_vectorstore
from .utils import format_citations
from .llm_provider import get_llm

def answer_question(
        question: str,
        collection_name: str = "handbook",
        persist_directory: str = "chroma_store/handbook",
        source_filter: Optional[str]=None,
        k: int = 5,
        provider: str = "ollama",
        model: Optional[str] = None,
) -> str:
    """
        Retrieves k relevant chunks and asks the LLM to answer using ONLY that context.
        - source_filter lets you restrict to a particular filename inside the collection.
        Returns the answer text from the LLM.
        """
    vectordb = get_vectorstore(collection_name, persist_directory)
    search_kwags = {"k": k}
    if source_filter:
        search_kwags["filter"] = {"source": source_filter}

    retriever = vectordb.as_retriever(search_kwags = search_kwags)
    docs = retriever.invoke(question)
    if not docs:
        return "I couldn't find relevant context. Try increasing --k or check the --source filename."

    context = format_citations(docs)

    system_prompt = (
        "Answer using ONLY the context below. If the answer is not in the context say "
        "'I don't know'. Always include citations like [file p.page]"
    )
    user_prompt = f"Question: {question}\n\nContext: \n{context}"

    llm = get_llm(provider=provider, model=model)

    resp = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])

    return resp.content if hasattr(resp, "content") else str(resp)


class QASession:
    """
        QASession — initialize once (vectordb + retriever + LLM), then ask many questions quickly.

        Why:
          - Avoids rebuilding retriever/LLM for every question in a chat loop.
          - Uses provider/model from .env (via get_llm) so you don't pass args each time.
        """
    def __init__(self,
                 collection_name: str = "handbook",
                 persist_directory: str = "chroma_store/handbook",
                 source_filter: Optional[str] = None,
                 k: int = 5,
                 provider: Optional[str] = None,
                 model: Optional[str] = None,
                 ):
        self.vectordb = get_vectorstore(collection_name, persist_directory)
        search_kwags = {"k": k}
        if source_filter:
            search_kwags["filter"] = {"source": source_filter}

        self.retriever = self.vectordb.as_retriever(search_kwags = search_kwags)
        self.llm = get_llm(provider=provider, model=model)

        self.system_prompt = (
            "Answer using ONLY the context below. If the answer is not in the context say "
            "I don't know. Always include citations like [file p.page]"
        )

    def ask(self, question: str) -> str:
        docs = self.retriever.invoke(question)
        if not docs:
            return "I couldn't find relevant context. Try increasing --k or check the --source filename."

        context = format_citations(docs)
        message = [
            {"role": "system", "content": context},
            {"role": "user", "content": f"Question: {question}\n\nContext: \n{context}"},
        ]

        resp = self.llm.invoke(message)
        return resp.content if hasattr(resp, "content") else str(resp)
