import pytest
from langchain_core.documents import Document

from src.evaluation.metrics import (
    answer_coverage_score,
    context_relevance_score,
    evaluate,
)


class TestContextRelevanceScore:
    def test_identical_text_scores_one(self):
        docs = [Document(page_content="machine learning AI", metadata={})]
        assert context_relevance_score("machine learning AI", docs) == 1.0

    def test_no_overlap_scores_zero(self):
        docs = [Document(page_content="unrelated topic here", metadata={})]
        assert context_relevance_score("machine learning AI", docs) == 0.0

    def test_empty_docs_scores_zero(self):
        assert context_relevance_score("any query", []) == 0.0

    def test_partial_overlap_between_zero_and_one(self):
        docs = [Document(page_content="machine learning concepts", metadata={})]
        score = context_relevance_score("machine learning AI", docs)
        assert 0.0 < score < 1.0


class TestAnswerCoverageScore:
    def test_answer_present_in_context_scores_high(self):
        docs = [Document(page_content="The capital of France is Paris", metadata={})]
        score = answer_coverage_score("Paris is the capital of France", docs)
        assert score > 0.5

    def test_empty_answer_scores_zero(self):
        docs = [Document(page_content="some context", metadata={})]
        assert answer_coverage_score("", docs) == 0.0

    def test_empty_docs_scores_zero(self):
        assert answer_coverage_score("some answer", []) == 0.0


class TestEvaluate:
    def test_returns_all_required_keys(self):
        docs = [Document(page_content="relevant information about query", metadata={})]
        result = evaluate("query", "answer about query", docs)
        assert "context_relevance" in result
        assert "answer_coverage" in result
        assert "num_docs_retrieved" in result

    def test_num_docs_matches_input(self):
        docs = [
            Document(page_content="doc1", metadata={}),
            Document(page_content="doc2", metadata={}),
        ]
        assert evaluate("q", "a", docs)["num_docs_retrieved"] == 2

    def test_scores_are_rounded(self):
        docs = [Document(page_content="partial overlap words", metadata={})]
        result = evaluate("words here", "some words", docs)
        # rounded to 4dp — check they are floats within [0, 1]
        for key in ("context_relevance", "answer_coverage"):
            assert 0.0 <= result[key] <= 1.0
