import pytest
from langchain_core.documents import Document

from src.config import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Ensures env-var changes in tests don't bleed across test cases."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def sample_documents() -> list[Document]:
    return [
        Document(
            page_content="LangChain is a framework for building LLM applications.",
            metadata={"source": "langchain_docs.txt"},
        ),
        Document(
            page_content=(
                "RAG stands for Retrieval-Augmented Generation. "
                "It combines retrieval with language models."
            ),
            metadata={"source": "rag_overview.txt"},
        ),
        Document(
            page_content="ChromaDB is an open-source vector database for storing embeddings.",
            metadata={"source": "chromadb_docs.txt"},
        ),
    ]


@pytest.fixture
def sample_text_file(tmp_path) -> str:
    content = "Artificial intelligence enables machines to simulate human reasoning and learning."
    f = tmp_path / "sample.txt"
    f.write_text(content, encoding="utf-8")
    return str(f)
