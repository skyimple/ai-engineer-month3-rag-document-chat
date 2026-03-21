import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse

from src.models import UploadResponse, QueryRequest, RAGResponse
from src.services.indexer import indexer
from src.services.retriever import retriever
from src.services.generator import generator
from src.config import config

router = APIRouter(prefix="/api", tags=["RAG"])

# Temporary directory for uploaded files
UPLOAD_TEMP_DIR = Path(tempfile.gettempdir()) / "rag_uploads"
UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload-pdf", response_model=UploadResponse)
async def upload_pdf(files: List[UploadFile]) -> UploadResponse:
    """
    Upload and index PDF files.

    - Accepts multiple PDF files in a single request
    - Extracts text and builds vector index
    - Returns list of successfully indexed files
    """
    indexed_files = []
    errors = []

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            errors.append(f"{file.filename}: Not a PDF file")
            continue

        try:
            # Save to temp directory
            temp_path = UPLOAD_TEMP_DIR / file.filename

            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Index the PDF
            result = indexer.index_pdf(str(temp_path))

            if result.get("status") == "success":
                indexed_files.append(file.filename)
            else:
                errors.append(f"{file.filename}: {result.get('reason', 'Unknown error')}")

            # Clean up temp file
            temp_path.unlink(missing_ok=True)

        except Exception as e:
            errors.append(f"{file.filename}: {str(e)}")
            # Clean up on error
            (UPLOAD_TEMP_DIR / file.filename).unlink(missing_ok=True)

    if not indexed_files and errors:
        return UploadResponse(
            status="partial",
            indexed_files=[],
            message=f"Errors: {'; '.join(errors)}"
        )

    return UploadResponse(
        status="success",
        indexed_files=indexed_files,
        message=f"Indexed {len(indexed_files)} file(s)" if indexed_files else None
    )


@router.post("/query", response_model=RAGResponse)
async def query(request: QueryRequest) -> RAGResponse:
    """
    Query the RAG system.

    - Takes a question and optional file_name filter
    - Retrieves relevant context and generates answer
    - Returns answer with source references
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        # Retrieve relevant sources
        sources = retriever.retrieve(
            query=request.question,
            file_name=request.file_name,
            top_k=request.top_k or config.DEFAULT_TOP_K
        )

        # Generate answer
        result = generator.generate_with_sources(
            question=request.question,
            sources=sources
        )

        return RAGResponse(
            answer=result["answer"],
            sources=sources,
            question=request.question
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    storage_path = config.get_storage_path()
    chroma_path = storage_path / "chroma"

    return {
        "status": "healthy",
        "storage_initialized": chroma_path.exists(),
        "storage_path": str(storage_path),
        "use_cohere_rerank": config.USE_COHERE_RERANK,
        "use_ollama": config.USE_OLLAMA,
        "llm_model": config.LLM_MODEL,
    }


@router.delete("/files/{file_name}")
async def delete_file_index(file_name: str):
    """Delete index for a specific file"""
    try:
        success = indexer.delete_file_index(file_name)
        if success:
            return {"status": "success", "message": f"Deleted index for {file_name}"}
        else:
            return {"status": "partial", "message": "Could not delete index"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
