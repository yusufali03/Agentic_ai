"""
llm_provider.py — choose an LLM (Ollama or OpenAI) behind a simple function.

- Default: Ollama (local). Change with provider="openai" and set OPENAI_API_KEY.
"""

import os
from dotenv import load_dotenv

load_dotenv()

def get_llm(provider: str | None = None, model: str | None = None):
    """
        Returns a LangChain chat model based on provider/model.
        provider: "ollama" (default) or "openai"
        model: e.g., "llama3.1:8b" for Ollama, or "gpt-4o-mini" for OpenAI
        """
    provider = provider or os.getenv("LLM_PROVIDER", "ollama").lower()
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        m = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=m, temperature=0)
    from langchain_ollama import ChatOllama
    m = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    return ChatOllama(model=m, temperature=0)