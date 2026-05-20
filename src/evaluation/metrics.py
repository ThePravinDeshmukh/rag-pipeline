from typing import List

from langchain_core.documents import Document


def _jaccard(text_a: str, text_b: str) -> float:
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def context_relevance_score(query: str, documents: List[Document]) -> float:
    """Jaccard similarity between query tokens and the combined retrieved context."""
    if not documents:
        return 0.0
    combined = " ".join(doc.page_content for doc in documents)
    return _jaccard(query, combined)


def answer_coverage_score(answer: str, documents: List[Document]) -> float:
    """Fraction of answer tokens that appear in the retrieved context."""
    if not documents or not answer.strip():
        return 0.0
    combined = " ".join(doc.page_content for doc in documents)
    return _jaccard(answer, combined)


def evaluate(query: str, answer: str, documents: List[Document]) -> dict:
    return {
        "context_relevance": round(context_relevance_score(query, documents), 4),
        "answer_coverage": round(answer_coverage_score(answer, documents), 4),
        "num_docs_retrieved": len(documents),
    }
