import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def client():
    with patch("src.orchestration.pipeline.build_rag_graph") as mock_graph:
        mock_graph.return_value = MagicMock()
        from fastapi.testclient import TestClient
        from src.api.main import app

        return TestClient(app)


class TestHealthEndpoint:
    def test_returns_ok(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_includes_version(self, client):
        response = client.get("/api/v1/health")
        assert "version" in response.json()


class TestIngestEndpoint:
    @patch("src.api.routes._pipeline")
    def test_success_returns_201(self, mock_pipeline, client):
        mock_pipeline.ingest.return_value = 5
        response = client.post("/api/v1/ingest", json={"sources": ["doc.txt"]})
        assert response.status_code == 201
        assert response.json()["chunks_indexed"] == 5

    @patch("src.api.routes._pipeline")
    def test_unsupported_file_type_returns_422(self, mock_pipeline, client):
        mock_pipeline.ingest.side_effect = ValueError("Unsupported file type '.xyz'")
        response = client.post("/api/v1/ingest", json={"sources": ["bad.xyz"]})
        assert response.status_code == 422

    def test_missing_sources_field_returns_422(self, client):
        response = client.post("/api/v1/ingest", json={})
        assert response.status_code == 422


class TestQueryEndpoint:
    @patch("src.api.routes._pipeline")
    def test_success_returns_answer(self, mock_pipeline, client):
        mock_pipeline.query.return_value = {
            "query": "What is RAG?",
            "original_query": "What is RAG?",
            "answer": "RAG is Retrieval-Augmented Generation.",
            "sources": ["doc.txt"],
            "latency_ms": 120,
        }
        response = client.post("/api/v1/query", json={"question": "What is RAG?"})
        assert response.status_code == 200
        assert response.json()["answer"] == "RAG is Retrieval-Augmented Generation."

    def test_empty_question_returns_422(self, client):
        response = client.post("/api/v1/query", json={"question": ""})
        assert response.status_code == 422

    def test_missing_question_field_returns_422(self, client):
        response = client.post("/api/v1/query", json={})
        assert response.status_code == 422


class TestMetricsEndpoint:
    @patch("src.api.routes.get_stats", return_value={"total_queries": 0})
    @patch("src.api.routes.get_recent_queries", return_value=[])
    def test_returns_stats_and_recent_queries(self, mock_recent, mock_stats, client):
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200
        body = response.json()
        assert "stats" in body
        assert "recent_queries" in body


class TestDeleteCollectionEndpoint:
    @patch("src.api.routes.clear_collection")
    def test_returns_204_on_success(self, mock_clear, client):
        response = client.delete("/api/v1/collection")
        assert response.status_code == 204
        mock_clear.assert_called_once()
