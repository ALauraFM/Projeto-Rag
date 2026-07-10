"""
Retrieval layer: embeddings, Qdrant vector store access and document retrieval.
Isolates everything related to fetching and formatting context from the
knowledge base, keeping generation (prompt + LLM) in app/rag.py.
"""
from __future__ import annotations

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.config import settings


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def get_vector_store(client: QdrantClient | None = None) -> QdrantVectorStore:
    if client is None:
        client = get_qdrant_client()
    return QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=get_embeddings(),
    )


def build_context(docs: list) -> str:
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "desconhecido")
        parts.append(f"[Fonte: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def extract_sources(docs: list) -> list[str]:
    seen: set[str] = set()
    sources: list[str] = []
    for doc in docs:
        src = doc.metadata.get("source", "desconhecido")
        if src not in seen:
            seen.add(src)
            sources.append(src)
    return sources


def retrieve_with_sources(vector_store: QdrantVectorStore, pergunta: str) -> dict:
    retriever = vector_store.as_retriever(
        search_kwargs={"k": settings.retrieval_top_k}
    )
    docs = retriever.invoke(pergunta)
    return {
        "context": build_context(docs),
        "fontes": extract_sources(docs),
        "pergunta": pergunta,
    }
