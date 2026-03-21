from typing import List, Optional, Tuple
import cohere
from llama_index.core import QueryBundle
from llama_index.core.retrievers import VectorIndexRetriever

from src.config import config
from src.services.indexer import indexer
from src.models import Source


class Retriever:
    """Retrieval with optional Cohere reranking"""

    def __init__(self):
        self._cohere_client = None
        if config.USE_COHERE_RERANK and config.COHERE_API_KEY:
            try:
                self._cohere_client = cohere.Client(config.COHERE_API_KEY)
            except Exception as e:
                print(f"Warning: Failed to initialize Cohere client: {e}")

    def _create_retriever(self, top_k: int) -> VectorIndexRetriever:
        """Create a base vector retriever"""
        index = indexer.get_index()

        return VectorIndexRetriever(
            index=index,
            similarity_top_k=top_k * 3 if config.USE_COHERE_RERANK else top_k,
            filters=None,
        )

    def _filter_by_file_name(
        self,
        nodes: List,
        file_name: Optional[str] = None
    ) -> List:
        """Filter retrieved nodes by file name if specified"""
        if not file_name:
            return nodes

        return [
            node for node in nodes
            if node.metadata.get("file_name") == file_name
        ]

    def _rerank_with_cohere(
        self,
        query: str,
        nodes: List,
        top_n: int
    ) -> List:
        """Rerank results using Cohere"""
        if not self._cohere_client or not nodes:
            return nodes[:top_n]

        try:
            doc_texts = [node.get_content() for node in nodes]

            response = self._cohere_client.rerank(
                query=query,
                documents=doc_texts,
                top_n=top_n,
                model="rerank-multilingual-v2.0"
            )

            reranked = []
            for result in response.results:
                reranked.append(nodes[result.index])

            return reranked

        except Exception as e:
            print(f"Cohere reranking failed: {e}")
            return nodes[:top_n]

    def retrieve(
        self,
        query: str,
        file_name: Optional[str] = None,
        top_k: int = 5
    ) -> List[Source]:
        """Retrieve relevant sources for a query"""
        retriever = self._create_retriever(top_k)
        query_bundle = QueryBundle(query_str=query)

        nodes = retriever.retrieve(query_bundle)
        nodes = self._filter_by_file_name(nodes, file_name)

        if config.USE_COHERE_RERANK and self._cohere_client:
            nodes = self._rerank_with_cohere(query, nodes, top_n=top_k)
        else:
            nodes = nodes[:top_k]

        sources = []
        for node in nodes:
            source = Source(
                content=node.get_content()[:500],
                file_name=node.metadata.get("file_name", "unknown"),
                page_label=str(node.metadata.get("page_label", "")),
                score=float(node.get_score()) if hasattr(node, 'get_score') else 0.0
            )
            sources.append(source)

        return sources


retriever = Retriever()
