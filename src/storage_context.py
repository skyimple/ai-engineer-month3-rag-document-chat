from pathlib import Path
from typing import Optional
import chromadb
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config import config


class StorageContextManager:
    """Manage persistent storage context for Chroma and LlamaIndex"""

    def __init__(self, collection_name: str = "rag_documents"):
        self.collection_name = collection_name
        self._storage_context: Optional[StorageContext] = None
        self._vector_store: Optional[ChromaVectorStore] = None

    def get_storage_path(self) -> Path:
        """Get the base storage path"""
        return config.get_storage_path()

    def get_chroma_path(self) -> Path:
        """Get Chroma persistence path"""
        return self.get_storage_path() / "chroma"

    def initialize(self) -> StorageContext:
        """Initialize or load persistent storage"""
        storage_path = self.get_storage_path()
        chroma_path = self.get_chroma_path()

        storage_path.mkdir(parents=True, exist_ok=True)
        chroma_path.mkdir(parents=True, exist_ok=True)

        chroma_client = chromadb.PersistentClient(path=str(chroma_path))

        chroma_collection = chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        self._vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        self._storage_context = StorageContext.from_defaults(
            vector_store=self._vector_store,
            persist_dir=str(storage_path)
        )

        return self._storage_context

    def get(self) -> Optional[StorageContext]:
        """Get the storage context, initializing if needed"""
        if self._storage_context is None:
            return self.initialize()
        return self._storage_context

    def persist(self):
        """Persist storage context"""
        if self._storage_context:
            self._storage_context.persist(persist_dir=str(self.get_storage_path()))

    def is_initialized(self) -> bool:
        """Check if storage is initialized"""
        return self._storage_context is not None


storage_manager = StorageContextManager()
