"""Application configuration loaded from environment / .env file."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized settings, populated from environment variables or .env.

    Defaults mirror .env.example so the app runs out of the box against a
    local Ollama instance.
    """

    # LLM provider selection: "ollama" or "groq"
    llm_provider: str = "ollama"

    # Ollama (local) configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    # Groq (hosted) configuration
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Vector store
    chroma_persist_dir: str = "chroma_db"
    collection_name: str = "qa_documents"

    # Retrieval
    top_k: int = 4

    # Source documents
    docs_dir: str = "data/docs"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 150

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
