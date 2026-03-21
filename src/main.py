import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.config import config

app = FastAPI(
    title="RAG Document Chat",
    description="PDF-based RAG system with LlamaIndex and Qwen LLM",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "RAG Document Chat API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health"
    }


def main():
    """Run the application"""
    import uvicorn

    print(f"Starting RAG Document Chat API...")
    print(f"Storage path: {config.get_storage_path()}")
    print(f"LLM Model: {config.LLM_MODEL}")
    print(f"Cohere Rerank: {config.USE_COHERE_RERANK}")
    print(f"Ollama Mode: {config.USE_OLLAMA}")

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
