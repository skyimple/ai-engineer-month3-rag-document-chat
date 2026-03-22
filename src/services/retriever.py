from typing import List, Optional
import json
import cohere
import chromadb

from src.config import config
from src.models import Source


class Retriever:
    """Retrieval with ChromaDB direct query and optional Cohere reranking"""

    def __init__(self):
        self._chroma_client = None
        self._collection = None
        self._embed_model = None
        self._cohere_client = None

        if config.USE_COHERE_RERANK and config.COHERE_API_KEY:
            try:
                self._cohere_client = cohere.Client(config.COHERE_API_KEY)
            except Exception as e:
                print(f"Warning: Failed to initialize Cohere client: {e}")

    def _get_collection(self):
        """Get or create ChromaDB collection"""
        if self._collection is None:
            from llama_index.embeddings.dashscope import DashScopeEmbedding

            chroma_path = config.get_storage_path() / "chroma"
            self._chroma_client = chromadb.PersistentClient(path=str(chroma_path))
            self._collection = self._chroma_client.get_or_create_collection(
                name="rag_documents",
                metadata={"hnsw:space": "cosine"}
            )

            # Initialize embedding model for query
            self._embed_model = DashScopeEmbedding(
                model_name=config.EMBEDDING_MODEL,
                api_key=config.DASHSCOPE_API_KEY
            )

        return self._collection

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for query text"""
        if self._embed_model is None:
            self._get_collection()
        return self._embed_model.get_text_embedding(query)

    def _parse_node_content(self, node_content: str) -> str:
        """Parse node content from JSON string to extract text"""
        try:
            node_data = json.loads(node_content)
            return node_data.get("text", node_content)
        except (json.JSONDecodeError, TypeError):
            return node_content

    def _rerank_with_cohere(
        self,
        query: str,
        results: List,
        top_n: int
    ) -> List:
        """Rerank results using Cohere"""
        if not self._cohere_client or not results:
            return results[:top_n]

        try:
            doc_texts = [r["document"] for r in results]

            response = self._cohere_client.rerank(
                query=query,
                documents=doc_texts,
                top_n=top_n,
                model="rerank-multilingual-v2.0"
            )

            reranked = []
            for result in response.results:
                reranked.append(results[result.index])

            return reranked

        except Exception as e:
            print(f"Cohere reranking failed: {e}")
            return results[:top_n]

    def retrieve(
        self,
        query: str,
        file_name: Optional[str] = None,
        top_k: int = 5
    ) -> List[Source]:
        """Retrieve relevant sources for a query using ChromaDB directly"""
        collection = self._get_collection()

        # Get query embedding
        query_embedding = self._get_query_embedding(query)

        # Calculate how many to retrieve (3x if reranking)
        retrieve_k = top_k * 3 if config.USE_COHERE_RERANK else top_k

        # Query ChromaDB with pre-computed embedding
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=retrieve_k,
            include=["documents", "metadatas", "distances"]
        )

        sources = []
        if results and results["ids"]:
            ids = results["ids"][0]
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]

            for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                # Filter by file_name if specified
                if file_name and metadata.get("file_name") != file_name:
                    continue

                # Parse text from node content or use document directly
                if metadata.get("_node_content"):
                    text = self._parse_node_content(metadata["_node_content"])
                else:
                    text = document

                # Convert distance to similarity score (cosine distance)
                score = 1.0 - distance if distance is not None else 0.0

                source = Source(
                    content=text[:500] if text else document[:500],
                    file_name=metadata.get("file_name", "unknown"),
                    page_label=str(metadata.get("page_label", "")),
                    score=float(score)
                )
                sources.append(source)

                # Stop if we have enough
                if len(sources) >= top_k:
                    break

        # Apply reranking if enabled
        if config.USE_COHERE_RERANK and self._cohere_client and sources:
            # Re-query without file filter for reranking
            all_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=retrieve_k,
                include=["documents", "metadatas", "distances"]
            )

            all_sources = []
            if all_results and all_results["ids"]:
                for i, (doc_id, document, metadata, distance) in enumerate(zip(
                    all_results["ids"][0], all_results["documents"][0],
                    all_results["metadatas"][0], all_results["distances"][0]
                )):
                    if file_name and metadata.get("file_name") != file_name:
                        continue

                    if metadata.get("_node_content"):
                        text = self._parse_node_content(metadata["_node_content"])
                    else:
                        text = document

                    score = 1.0 - distance if distance is not None else 0.0

                    all_sources.append(Source(
                        content=text[:500] if text else document[:500],
                        file_name=metadata.get("file_name", "unknown"),
                        page_label=str(metadata.get("page_label", "")),
                        score=float(score)
                    ))

            sources = self._rerank_with_cohere(query, all_sources, top_n=top_k)

        return sources


retriever = Retriever()
