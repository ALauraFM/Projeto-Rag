"""
Ingestion script: loads docs from /docs, chunks them, embeds and upserts into Qdrant.
Uses QdrantVectorStore.add_documents so payloads are stored in LangChain's expected format.
Idempotent: skips if the collection already has points.
"""
import logging
import sys

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.config import settings
from app.rag import get_embeddings, load_and_split_docs

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def ingest() -> None:
    client = QdrantClient(url=settings.qdrant_url)

    collections = [c.name for c in client.get_collections().collections]

    if settings.qdrant_collection in collections:
        count = client.count(settings.qdrant_collection).count
        if count > 0:
            logger.info(
                "Collection '%s' already has %d points — skipping ingestion.",
                settings.qdrant_collection,
                count,
            )
            return

    logger.info("Loading and splitting documents from '%s'...", settings.docs_path)
    chunks = load_and_split_docs()
    logger.info("Loaded %d chunks.", len(chunks))

    logger.info("Embedding and upserting via QdrantVectorStore...")
    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection,
    )

    logger.info(
        "Ingested %d chunks into collection '%s'.",
        len(chunks),
        settings.qdrant_collection,
    )


if __name__ == "__main__":
    ingest()
    sys.exit(0)
