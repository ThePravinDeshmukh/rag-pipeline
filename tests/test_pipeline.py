import pytest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document


class TestRAGPipelineIngest:
    @patch("src.orchestration.pipeline.build_rag_graph")
    @patch("src.orchestration.pipeline.add_documents", return_value=3)
    @patch("src.orchestration.pipeline.chunk_documents")
    @patch("src.orchestration.pipeline.clean_documents")
    @patch("src.orchestration.pipeline.load_sources")
    def test_returns_chunk_count(
        self, mock_load, mock_clean, mock_chunk, mock_add, mock_graph
    ):
        from src.orchestration.pipeline import RAGPipeline

        mock_load.return_value = [MagicMock()]
        mock_clean.return_value = [MagicMock()]
        mock_chunk.return_value = [MagicMock(), MagicMock(), MagicMock()]
        mock_graph.return_value = MagicMock()

        pipeline = RAGPipeline()
        assert pipeline.ingest(["doc.txt"]) == 3

    @patch("src.orchestration.pipeline.build_rag_graph")
    @patch("src.orchestration.pipeline.add_documents", return_value=0)
    @patch("src.orchestration.pipeline.chunk_documents", return_value=[])
    @patch("src.orchestration.pipeline.clean_documents", return_value=[])
    @patch("src.orchestration.pipeline.load_sources", return_value=[])
    def test_empty_sources_returns_zero(
        self, mock_load, mock_clean, mock_chunk, mock_add, mock_graph
    ):
        from src.orchestration.pipeline import RAGPipeline

        mock_graph.return_value = MagicMock()
        pipeline = RAGPipeline()
        assert pipeline.ingest([]) == 0


class TestRAGPipelineQuery:
    @patch("src.orchestration.pipeline.log_query")
    @patch("src.orchestration.pipeline.build_rag_graph")
    def test_returns_expected_keys(self, mock_graph, mock_log):
        from src.orchestration.pipeline import RAGPipeline

        mock_compiled = MagicMock()
        mock_compiled.invoke.return_value = {
            "query": "test question",
            "answer": "test answer",
            "documents": [Document(page_content="ctx", metadata={"source": "doc.txt"})],
            "rewrite_count": 0,
        }
        mock_graph.return_value = mock_compiled

        pipeline = RAGPipeline()
        result = pipeline.query("test question")

        assert result["answer"] == "test answer"
        assert result["sources"] == ["doc.txt"]
        assert "latency_ms" in result
        assert isinstance(result["latency_ms"], int)

    @patch("src.orchestration.pipeline.log_query")
    @patch("src.orchestration.pipeline.build_rag_graph")
    def test_logs_query_after_invocation(self, mock_graph, mock_log):
        from src.orchestration.pipeline import RAGPipeline

        mock_compiled = MagicMock()
        mock_compiled.invoke.return_value = {
            "query": "q",
            "answer": "a",
            "documents": [],
            "rewrite_count": 0,
        }
        mock_graph.return_value = mock_compiled

        RAGPipeline().query("q")
        mock_log.assert_called_once()
