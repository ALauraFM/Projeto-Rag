from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "yaitec_knowledge"
    docs_path: str = "docs"
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    retrieval_top_k: int = 4
    chunk_size: int = 500
    chunk_overlap: int = 50
    stt_model: str = "whisper-1"
    tts_model: str = "gpt-4o-mini-tts"
    tts_voice: str = "alloy"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
