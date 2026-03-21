from typing import Optional, List
from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """Response for PDF upload endpoint"""
    status: str = "success"
    indexed_files: List[str] = Field(default_factory=list)
    message: Optional[str] = None


class QueryRequest(BaseModel):
    """Request for query endpoint"""
    question: str = Field(..., min_length=1, description="The question to ask")
    file_name: Optional[str] = Field(None, description="Filter by specific file name")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to retrieve")


class Source(BaseModel):
    """Source document reference"""
    content: str = Field(..., description="Source text content")
    file_name: str = Field(..., description="Source file name")
    page_label: str = Field(..., description="Page label number")
    score: float = Field(..., description="Relevance score")


class RAGResponse(BaseModel):
    """RAG query response"""
    answer: str = Field(..., description="Generated answer")
    sources: List[Source] = Field(default_factory=list, description="Source references")
    question: str = Field(..., description="Original question")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = "healthy"
    storage_initialized: bool = False
    llm_available: bool = False
