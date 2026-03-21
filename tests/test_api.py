import pytest
from io import BytesIO
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


class TestHealthCheck:
    """Tests for /api/health endpoint."""

    def test_health_check(self, client):
        """Test health endpoint returns healthy status."""
        response = client.get("/api/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "storage_path" in data
        assert "llm_model" in data


class TestRootEndpoint:
    """Tests for / endpoint."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "RAG Document Chat API"
        assert data["version"] == "1.0.0"
        assert "docs" in data
        assert "health" in data


class TestUploadPDF:
    """Tests for /api/upload-pdf endpoint."""

    def test_upload_pdf_validation(self, client):
        """Test that non-PDF files are rejected."""
        # Create a fake text file
        file_content = b"This is not a PDF"
        files = {
            "files": ("test.txt", BytesIO(file_content), "text/plain")
        }

        response = client.post("/api/upload-pdf", files=files)

        assert response.status_code == 200
        data = response.json()
        # Non-PDF should be rejected with errors
        assert len(data.get("indexed_files", [])) == 0
        assert "test.txt" in data.get("message", "").lower() or "not a pdf" in data.get("message", "").lower()

    @patch("src.api.routes.indexer")
    def test_upload_pdf_success(self, mock_indexer, client):
        """Test successful PDF upload with mocked indexer."""
        mock_indexer.index_pdf.return_value = {"status": "success"}

        # Create a fake PDF content
        pdf_content = b"%PDF-1.4 fake pdf content"
        files = {
            "files": ("test.pdf", BytesIO(pdf_content), "application/pdf")
        }

        response = client.post("/api/upload-pdf", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "test.pdf" in data["indexed_files"]


class TestQuery:
    """Tests for /api/query endpoint."""

    def test_query_empty_question(self, client):
        """Test that empty question returns 422 validation error."""
        payload = {"question": ""}

        response = client.post("/api/query", json=payload)

        # FastAPI returns 422 for Pydantic validation errors
        assert response.status_code == 422

    def test_query_whitespace_question(self, client):
        """Test that whitespace-only question returns 400 error."""
        payload = {"question": "   "}

        response = client.post("/api/query", json=payload)

        assert response.status_code == 400

    @patch("src.api.routes.generator")
    @patch("src.api.routes.retriever")
    def test_query_with_question(self, mock_retriever, mock_generator, client):
        """Test query endpoint with mocked retriever and generator."""
        # Mock empty sources
        mock_retriever.retrieve.return_value = []

        # Mock generator response
        mock_generator.generate_with_sources.return_value = {
            "answer": "No relevant documents found.",
            "sources": [],
            "question": "What is AI?"
        }

        payload = {"question": "What is AI?"}

        response = client.post("/api/query", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert data["question"] == "What is AI?"


class TestDeleteFile:
    """Tests for /api/files/{file_name} endpoint."""

    @patch("src.api.routes.indexer")
    def test_delete_file_index(self, mock_indexer, client):
        """Test delete endpoint exists and returns success."""
        mock_indexer.delete_file_index.return_value = True

        file_name = "test.pdf"
        response = client.delete(f"/api/files/{file_name}")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["success", "partial"]

    @patch("src.api.routes.indexer")
    def test_delete_file_index_not_found(self, mock_indexer, client):
        """Test delete returns partial status when file not found."""
        mock_indexer.delete_file_index.return_value = False

        file_name = "nonexistent.pdf"
        response = client.delete(f"/api/files/{file_name}")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "partial"
