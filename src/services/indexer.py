from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode
from llama_index.core.storage import StorageContext
from llama_index.core.vector_stores.types import VectorStore
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from src.config import config
from src.services.pdf_processor import pdf_processor, clean_text


class Indexer:
    """Build and manage RAG index using LlamaIndex and Chroma"""

    def __init__(self, collection_name: str = "rag_documents"):
        self.collection_name = collection_name
        self._storage_context: Optional[StorageContext] = None
        self._vector_store: Optional[VectorStore] = None
        self._chroma_client: Optional[chromadb.PersistentClient] = None
        self._index = None
        self._chunk_splitter = SentenceSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )

    def _get_storage_path(self) -> Path:
        """Get storage directory path"""
        return config.get_storage_path()

    def _get_chroma_path(self) -> Path:
        """Get Chroma persistence path"""
        return self._get_storage_path() / "chroma"

    def initialize_storage(self) -> StorageContext:
        """Initialize or load persistent storage"""
        storage_path = self._get_storage_path()
        chroma_path = self._get_chroma_path()

        # Create directories
        storage_path.mkdir(parents=True, exist_ok=True)
        chroma_path.mkdir(parents=True, exist_ok=True)

        # Initialize Chroma client with persistence
        self._chroma_client = chromadb.PersistentClient(path=str(chroma_path))

        # Get or create collection
        chroma_collection = self._chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Create vector store
        self._vector_store = ChromaVectorStore(
            chroma_collection=chroma_collection
        )

        # Create storage context with explicit docstore and index store
        from llama_index.core.storage.docstore import SimpleDocumentStore
        from llama_index.core.storage.index_store import SimpleIndexStore

        docstore = SimpleDocumentStore()
        index_store = SimpleIndexStore()

        self._storage_context = StorageContext.from_defaults(
            vector_store=self._vector_store,
            docstore=docstore,
            index_store=index_store,
            persist_dir=str(storage_path)
        )

        return self._storage_context

    def _create_document_from_page(
        self,
        page_content: Dict[str, Any],
        file_name: str
    ) -> Document:
        """Create a LlamaIndex Document from page content"""
        text = page_content["text"]
        metadata = page_content["metadata"]

        # Clean the text
        cleaned_text = clean_text(text)

        # Skip empty pages
        if not cleaned_text or len(cleaned_text) < 10:
            return None

        # Create document with metadata
        doc = Document(
            text=cleaned_text,
            metadata={
                "file_name": file_name,
                "page_label": str(metadata.get("page_label", "")),
                "page_number": metadata.get("page_number", 0),
                "upload_date": datetime.now().isoformat(),
                "file_type": "pdf",
            },
            metadata_separator="\n",
            text_template="{content}"
        )

        return doc

    def index_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        Index a PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            Indexing statistics
        """
        # Initialize storage if not already done
        if self._storage_context is None:
            self.initialize_storage()

        file_path = Path(file_path)
        file_name = file_path.name

        # Process PDF
        pdf_data = pdf_processor.process_pdf(str(file_path), file_name)

        # Create documents from pages
        documents = []
        for page in pdf_data["pages"]:
            doc = self._create_document_from_page(page, file_name)
            if doc:
                documents.append(doc)

        if not documents:
            return {
                "status": "skipped",
                "reason": "No valid content found in PDF",
                "file_name": file_name
            }

        # Create new index - always rebuild to ensure Chroma persistence
        from llama_index.core import VectorStoreIndex
        from llama_index.embeddings.dashscope import DashScopeEmbedding

        embed_model = DashScopeEmbedding(
            model_name=config.EMBEDDING_MODEL,
            api_key=config.DASHSCOPE_API_KEY
        )

        # Process documents in batches to respect DashScope batch size limit (max 10)
        batch_size = 10
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            if i == 0:
                # First batch - create new index
                self._index = VectorStoreIndex.from_documents(
                    batch,
                    storage_context=self._storage_context,
                    embed_model=embed_model,
                    show_progress=True
                )
            else:
                # Subsequent batches - insert into existing index
                for doc in batch:
                    self._index.insert(doc)

        # Chroma PersistentClient auto-persist is enabled, no manual call needed

        return {
            "status": "success",
            "file_name": file_name,
            "pages_processed": len(documents),
            "total_pages": pdf_data["total_pages"],
            "has_images": pdf_data["has_images"],
        }

    def index_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """Index a list of pre-processed documents"""
        if self._storage_context is None:
            self.initialize_storage()

        if not documents:
            return {"status": "skipped", "reason": "No documents provided"}

        from llama_index.core import VectorStoreIndex
        from llama_index.embeddings.dashscope import DashScopeEmbedding

        embed_model = DashScopeEmbedding(
            model_name=config.EMBEDDING_MODEL,
            api_key=config.DASHSCOPE_API_KEY
        )

        try:
            self._index = load_index_from_storage(self._storage_context)
            for doc in documents:
                self._index.insert(doc)
        except Exception:
            self._index = VectorStoreIndex.from_documents(
                documents,
                storage_context=self._storage_context,
                embed_model=embed_model,
                show_progress=True
            )

        return {
            "status": "success",
            "documents_indexed": len(documents),
        }

    def get_index(self):
        """Get or create the index"""
        if self._storage_context is None:
            self.initialize_storage()

        if self._index is None:
            from llama_index.core import VectorStoreIndex
            from llama_index.embeddings.dashscope import DashScopeEmbedding

            embed_model = DashScopeEmbedding(
                model_name=config.EMBEDDING_MODEL,
                api_key=config.DASHSCOPE_API_KEY
            )

            try:
                from llama_index.core import load_index_from_storage
                self._index = load_index_from_storage(self._storage_context)
            except Exception:
                self._index = VectorStoreIndex.from_documents(
                    [],
                    storage_context=self._storage_context,
                    embed_model=embed_model,
                )

        return self._index

    def delete_file_index(self, file_name: str) -> bool:
        """Delete all nodes associated with a file"""
        if self._vector_store is None:
            return False

        try:
            # Get all items and filter
            collection = self._vector_store._collection
            all_items = collection.get(include=["metadatas"])

            ids_to_delete = []
            for idx, metadata in enumerate(all_items.get("metadatas", [])):
                if metadata and metadata.get("file_name") == file_name:
                    ids_to_delete.append(all_items["ids"][idx])

            if ids_to_delete:
                collection.delete(ids_to_delete)

            return True
        except Exception as e:
            print(f"Error deleting index: {e}")
            return False


indexer = Indexer()
