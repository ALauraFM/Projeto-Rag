from __future__ import annotations

from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.config import settings
from app.retrieval_pipeline import retrieve_with_sources

SYSTEM_PROMPT = """Você é o assistente virtual do YAITEC Atende.
Responda exclusivamente com base nos trechos de documentação fornecidos.
Cite sempre o documento de origem usando o formato [Fonte: <nome_do_arquivo>].
Se a informação não estiver nos trechos, diga que não encontrou nas fontes disponíveis.

Trechos de documentação:
{context}
"""


def build_rag_chain(vector_store: QdrantVectorStore):
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

    answer_chain = (
        RunnablePassthrough.assign(context=lambda x: x["context"])
        | prompt
        | llm
        | StrOutputParser()
    )

    def full_chain(pergunta: str) -> dict:
        retrieved = retrieve_with_sources(vector_store, pergunta)
        resposta = answer_chain.invoke(retrieved)
        return {"resposta": resposta, "fontes": retrieved["fontes"]}

    return full_chain
