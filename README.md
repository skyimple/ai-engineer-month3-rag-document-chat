# RAG Document Chat

A PDF-based Retrieval-Augmented Generation (RAG) system that enables intelligent question answering over PDF documents using advanced LLM capabilities.

## Features

- **PDF Processing**: Extract and index text content from PDF documents with page-level metadata
- **Vector Storage**: Persistent Chroma vector store for efficient similarity search
- **RAG Pipeline**: Retrieve relevant context and generate accurate answers using Qwen LLM
- **Optional Reranking**: Cohere-powered reranking for improved result quality
- **RESTful API**: FastAPI-based API for easy integration
- **Incremental Indexing**: Support for adding and removing documents from the index

## Tech Stack

| Component | Technology |
|-----------|------------|
| API Framework | FastAPI + Uvicorn |
| Vector Store | ChromaDB |
| LLM Framework | LlamaIndex |
| LLM Model | Qwen3.5-plus (Alibaba Cloud DashScope) |
| Embedding Model | text-embedding-v3 |
| PDF Processing | PyMuPDF |
| Optional Reranking | Cohere |
| Configuration | python-dotenv |

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ai-engineer-month3-rag-document-chat
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp config.env .env
# Edit .env with your API keys
```

## Configuration

Create a `config.env` file (or set environment variables) with the following:

```env
# Alibaba Cloud DashScope API Key (Required for Qwen LLM)
DASHSCOPE_API_KEY=your_api_key_here

# Cohere Reranking (Optional)
COHERE_API_KEY=your_cohere_key_here
USE_COHERE_RERANK=false

# Ollama Local Mode (Optional, for future extension)
USE_OLLAMA=false
OLLAMA_BASE_URL=http://localhost:11434

# Storage
STORAGE_PATH=./storage
```

### Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DASHSCOPE_API_KEY` | Yes | - | Alibaba Cloud DashScope API key for Qwen LLM |
| `COHERE_API_KEY` | No | - | Cohere API key for reranking |
| `USE_COHERE_RERANK` | No | `false` | Enable Cohere reranking |
| `USE_OLLAMA` | No | `false` | Use Ollama for local LLM (future) |
| `OLLAMA_BASE_URL` | No | `http://localhost:11434` | Ollama server URL |
| `STORAGE_PATH` | No | `./storage` | Path for persistent vector storage |

## API Endpoints

### Health Check

```
GET /api/health
```

Check the service health and configuration.

**Response:**
```json
{
  "status": "healthy",
  "storage_initialized": true,
  "storage_path": "./storage",
  "use_cohere_rerank": false,
  "use_ollama": false,
  "llm_model": "qwen3.5-plus"
}
```

### Upload PDF

```
POST /api/upload-pdf
Content-Type: multipart/form-data
```

Upload and index one or more PDF files.

**Request:**
- `files`: PDF files (multiple files supported)

**Response:**
```json
{
  "status": "success",
  "indexed_files": ["document.pdf"],
  "message": "Indexed 1 file(s)"
}
```

**Example with curl:**
```bash
curl -X POST "http://localhost:8000/api/upload-pdf" \
  -F "files=@document.pdf"
```

### Query

```
POST /api/query
Content-Type: application/json
```

Query the RAG system with a question.

**Request Body:**
```json
{
  "question": "What is the main topic of the document?",
  "file_name": "document.pdf",
  "top_k": 5
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `question` | string | Yes | - | The question to ask |
| `file_name` | string | No | - | Filter by specific file |
| `top_k` | integer | No | 5 | Number of results (1-20) |

**Response:**
```json
{
  "answer": "Based on the document...",
  "sources": [
    {
      "content": "Extracted text content...",
      "file_name": "document.pdf",
      "page_label": "1",
      "score": 0.85
    }
  ],
  "question": "What is the main topic of the document?"
}
```

**Example with curl:**
```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?"}'
```

### Delete File Index

```
DELETE /api/files/{file_name}
```

Remove a file's index from the vector store.

**Response:**
```json
{
  "status": "success",
  "message": "Deleted index for document.pdf"
}
```

## Usage Example

### Step 1: Start the Server

```bash
cd ai-engineer-month3-rag-document-chat
python -m src.main
```

The server starts at `http://localhost:8000`. API documentation is available at `http://localhost:8000/docs`.

### Step 2: Upload a PDF

Using the Swagger UI at `/docs` or curl:

```bash
curl -X POST "http://localhost:8000/api/upload-pdf" \
  -F "files=@your-document.pdf"
```

### Step 3: Ask Questions

```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Summarize the main points",
    "top_k": 3
  }'
```

### Step 4: Delete a File (Optional)

```bash
curl -X DELETE "http://localhost:8000/api/files/your-document.pdf"
```

## Project Structure

```
ai-engineer-month3-rag-document-chat/
├── config.env              # Configuration file
├── requirements.txt        # Python dependencies
├── .gitignore
├── README.md
└── src/
    ├── __init__.py
    ├── main.py             # FastAPI application entry
    ├── config.py           # Configuration management
    ├── models.py           # Pydantic data models
    ├── storage_context.py  # Chroma storage management
    ├── api/
    │   ├── __init__.py
    │   └── routes.py       # API endpoints
    └── services/
        ├── __init__.py
        ├── pdf_processor.py    # PDF text extraction
        ├── indexer.py          # LlamaIndex indexing
        ├── retriever.py        # Vector retrieval + reranking
        └── generator.py        # LLM answer generation
```

## Evaluation

To run the evaluation system:

```bash
# Install evaluation dependencies (already in requirements.txt)
pip install ragas>=0.1.0

# Run evaluation script
python eval.py
```

Note: The evaluation module requires:
- Indexed test documents
- Ground truth question-answer pairs
- Configured LLM API access

## Storage

The system persists data in the `./storage` directory (configurable):

```
storage/
├── chroma/              # ChromaDB vector store
│   └── ...
└── (additional persisted data)
```

Vector embeddings use cosine similarity with the `text-embedding-v3` model.
