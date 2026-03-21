import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load .env file
env_path = Path(__file__).parent.parent / "config.env"
load_dotenv(env_path)


class Config:
    """Configuration management for RAG Document Chat"""

    # Alibaba Cloud DashScope API
    DASHSCOPE_API_KEY: str = os.getenv("DASHSCOPE_API_KEY", "")

    # Cohere Reranking
    COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")
    USE_COHERE_RERANK: bool = os.getenv("USE_COHERE_RERANK", "false").lower() == "true"

    # Ollama Local Mode (for future extension)
    USE_OLLAMA: bool = os.getenv("USE_OLLAMA", "false").lower() == "true"
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Storage
    STORAGE_PATH: str = os.getenv("STORAGE_PATH", "./storage")

    # Chunk settings for indexing
    CHUNK_SIZE: int = 768
    CHUNK_OVERLAP: int = 150

    # Retrieval settings
    DEFAULT_TOP_K: int = 5
    RERANK_TOP_K: int = 20  # Initial retrieval count before reranking

    # LLM settings
    LLM_MODEL: str = "qwen3.5-plus"
    EMBEDDING_MODEL: str = "text-embedding-v3"

    @classmethod
    def get_storage_path(cls) -> Path:
        """Get absolute path for storage directory"""
        path = Path(cls.STORAGE_PATH)
        if not path.is_absolute():
            path = Path(__file__).parent.parent / path
        return path


config = Config()
