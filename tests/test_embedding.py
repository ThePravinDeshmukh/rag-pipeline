import pytest
from unittest.mock import MagicMock, patch

from src.embedding.embedder import get_embedder, OpenAIEmbedder, HuggingFaceEmbedder


class TestGetEmbedder:
    def test_raises_for_unknown_provider(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_PROVIDER", "nonexistent")
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            get_embedder()

    @patch.object(OpenAIEmbedder, "get_embeddings")
    def test_selects_openai_provider(self, mock_embed, monkeypatch):
        monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
        mock_embed.return_value = MagicMock()
        result = get_embedder()
        assert result is not None
        mock_embed.assert_called_once()

    @patch.object(HuggingFaceEmbedder, "get_embeddings")
    def test_selects_huggingface_provider(self, mock_embed, monkeypatch):
        monkeypatch.setenv("EMBEDDING_PROVIDER", "huggingface")
        mock_embed.return_value = MagicMock()
        result = get_embedder()
        assert result is not None
        mock_embed.assert_called_once()
