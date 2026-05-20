import pytest
from langchain_core.documents import Document

from src.retrieval.prompt_builder import format_context, RAG_PROMPT


class TestFormatContext:
    def test_includes_source_and_content(self):
        docs = [Document(page_content="Hello world", metadata={"source": "test.txt"})]
        context = format_context(docs)
        assert "test.txt" in context
        assert "Hello world" in context

    def test_empty_docs_returns_empty_string(self):
        assert format_context([]) == ""

    def test_falls_back_to_unknown_source(self):
        docs = [Document(page_content="content", metadata={})]
        assert "unknown" in format_context(docs)

    def test_separates_multiple_docs(self):
        docs = [
            Document(page_content="Doc 1", metadata={"source": "a.txt"}),
            Document(page_content="Doc 2", metadata={"source": "b.txt"}),
        ]
        context = format_context(docs)
        assert "Doc 1" in context
        assert "Doc 2" in context
        assert "---" in context


class TestRAGPrompt:
    def test_prompt_contains_context_and_question_variables(self):
        input_variables = RAG_PROMPT.input_variables
        assert "context" in input_variables
        assert "question" in input_variables
