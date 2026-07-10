from __future__ import annotations

import os

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from qdrant_client import QdrantClient

from app.config import settings

SYSTEM_PROMPT = """Você é o assistente virtual do YAITEC Atende.
Responda exclusivamente com base nos trechos de documentação fornecidos.
Cite sempre o documento de origem usando o formato [Fonte: <nome_do_arquivo>].
Se a informação não estiver nos trechos, diga que não encontrou nas fontes disponíveis.

Trechos de documentação:
{context}
"""


def _build_context(docs: list) -> str:
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "desconhecido")
        parts.append(f"[Fonte: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _extract_sources(docs: list) -> list[str]:
    seen: set[str] = set()
    sources: list[str] = []
    for doc in docs:
        src = doc.metadata.get("source", "desconhecido")
        if src not in seen:
            seen.add(src)
            sources.append(src)
    return sources


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


def load_and_split_docs():
    loader = DirectoryLoader(
        settings.docs_path,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    raw_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(raw_docs)

    for chunk in chunks:
        chunk.metadata["source"] = os.path.basename(chunk.metadata.get("source", ""))

    return chunks


def build_rag_chain(vector_store: QdrantVectorStore):
    retriever = vector_store.as_retriever(
        search_kwargs={"k": settings.retrieval_top_k}
    )

    llm = ChatOpenAI(
        model=settings.llm_model,
        openai_api_key=settings.openai_api_key,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{pergunta}"),
        ]
    )

    def retrieve_with_sources(pergunta: str) -> dict:
        docs = retriever.invoke(pergunta)
        return {
            "context": _build_context(docs),
            "fontes": _extract_sources(docs),
            "pergunta": pergunta,
        }

    answer_chain = (
        RunnablePassthrough.assign(context=lambda x: x["context"])
        | prompt
        | llm
        | StrOutputParser()
    )

    def full_chain(pergunta: str) -> dict:
        retrieved = retrieve_with_sources(pergunta)
        resposta = answer_chain.invoke(retrieved)
        return {"resposta": resposta, "fontes": retrieved["fontes"]}

    return full_chain
