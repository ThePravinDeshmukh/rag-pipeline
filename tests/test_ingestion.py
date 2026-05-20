import pytest
from langchain_core.documents import Document

from src.ingestion.cleaners import clean_documents, clean_text
from src.ingestion.chunkers import chunk_documents
from src.ingestion.loaders import load_file, load_sources, SUPPORTED_EXTENSIONS


class TestCleanText:
    def test_strips_extra_spaces(self):
        assert clean_text("hello   world") == "hello world"

    def test_removes_null_chars(self):
        assert "\x00" not in clean_text("hello\x00world")

    def test_collapses_excessive_newlines(self):
        result = clean_text("line1\n\n\n\nline2")
        assert "\n\n\n" not in result

    def test_normalises_unicode(self):
        # NFKC: fullwidth Latin letter → ASCII equivalent
        assert clean_text("ａ") == "a"


class TestCleanDocuments:
    def test_filters_blank_documents(self):
        docs = [
            Document(page_content="   ", metadata={}),
            Document(page_content="valid content", metadata={}),
        ]
        result = clean_documents(docs)
        assert len(result) == 1
        assert result[0].page_content == "valid content"

    def test_preserves_metadata(self, sample_documents):
        result = clean_documents(sample_documents)
        assert len(result) == len(sample_documents)
        for doc in result:
            assert "source" in doc.metadata

    def test_empty_input(self):
        assert clean_documents([]) == []


class TestChunkDocuments:
    def test_splits_long_document(self):
        doc = Document(page_content=" ".join(["word"] * 500), metadata={"source": "x"})
        chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1

    def test_preserves_metadata(self, sample_documents):
        chunks = chunk_documents(sample_documents, chunk_size=100, chunk_overlap=10)
        for chunk in chunks:
            assert "source" in chunk.metadata

    def test_empty_input_returns_empty(self):
        assert chunk_documents([]) == []

    def test_chunk_size_is_respected(self):
        doc = Document(page_content="A" * 1000, metadata={})
        chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=0)
        for chunk in chunks:
            # splitter may overshoot slightly due to separator logic
            assert len(chunk.page_content) <= 250


class TestLoadFile:
    def test_loads_text_file(self, sample_text_file):
        docs = load_file(sample_text_file)
        assert len(docs) >= 1
        assert "artificial intelligence" in docs[0].page_content.lower()

    def test_raises_for_unsupported_extension(self, tmp_path):
        bad_file = tmp_path / "data.xyz"
        bad_file.write_text("content")
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_file(str(bad_file))


class TestLoadSources:
    def test_loads_multiple_files(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("Document A content")
        f2.write_text("Document B content")
        docs = load_sources([str(f1), str(f2)])
        assert len(docs) >= 2

    def test_loads_directory(self, tmp_path):
        (tmp_path / "c.txt").write_text("Document C")
        docs = load_sources([str(tmp_path)])
        assert len(docs) >= 1
