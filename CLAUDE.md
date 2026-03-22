# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PDF-based RAG (Retrieval-Augmented Generation) system that enables intelligent question answering over PDF documents using Qwen LLM via Alibaba Cloud DashScope. It uses LlamaIndex for indexing/retrieval and ChromaDB for vector storage.

## Common Commands

### Start the Server
```bash
python -m src.main
```
Server runs at `http://localhost:8000` with auto-reload enabled.

### Run Tests
```bash
pytest
pytest tests/test_api.py::TestHealthCheck  # Run specific test class
```

### Configuration
Copy `config.env` to `.env` and set required API keys:
- `DASHSCOPE_API_KEY` (required) - Alibaba Cloud DashScope API key for Qwen LLM
- `COHERE_API_KEY` (optional) - For Cohere reranking
- `USE_COHERE_RERANK=false` - Enable/disable reranking
- `USE_OLLAMA=false` - Use Ollama local LLM instead of DashScope
- `OLLAMA_BASE_URL=http://localhost:11434` - Ollama server URL
- `OLLAMA_MODEL=qwen3.5-plus:0.8b` - Ollama model name

## Architecture

### Service Layer (Singleton Pattern)
Services are instantiated at module level as singletons in `src/services/`:
- `src.services.indexer` - Manages ChromaDB vector storage and document indexing (Indexer class)
- `src.services.retriever` - Handles vector search with optional Cohere reranking
- `src.services.generator` - LLM answer generation (DashScope API or Ollama local model via `USE_OLLAMA=true`)
- `src.services.pdf_processor` - PDF text extraction using PyMuPDF

### RAG Pipeline Flow
1. **Upload** (`POST /api/upload-pdf`) → PDFProcessor extracts text → Indexer chunks and embeds → ChromaDB stores
2. **Query** (`POST /api/query`) → Retriever searches vector store → Optional reranking → Generator creates answer using Qwen
3. **Delete** (`DELETE /api/files/{file_name}`) → Removes file's chunks from ChromaDB

### Key Files
- `src/config.py` - Configuration class loading from `config.env`
- `src/models.py` - Pydantic request/response models
- `src/storage_context.py` - ChromaDB storage context management
- `src/api/routes.py` - FastAPI endpoints
- `src/main.py` - FastAPI app initialization
- `evaluations/eval.py` - RAG evaluation script using Ragas

### Storage
Data persists in `./storage` directory (configurable via `STORAGE_PATH`):
- `storage/chroma/` - ChromaDB vector store
- Uses cosine similarity with `text-embedding-v3` embeddings

### Indexing Parameters
- Chunk size: 768 tokens with 150 overlap
- Default top_k: 5 results
- Initial retrieval fetches 3x results when Cohere reranking is enabled
