from typing import List, Optional, Tuple

from langchain_core.documents import Document

from src.config import get_settings
from src.embedding.indexer import get_store


def retrieve_documents(query: str, top_k: Optional[int] = None) -> List[Document]:
    settings = get_settings()
    return get_store().similarity_search(query, k=top_k or settings.top_k)


def retrieve_with_scores(
    query: str, top_k: Optional[int] = None
) -> List[Tuple[Document, float]]:
    settings = get_settings()
    return get_store().similarity_search_with_relevance_scores(
        query, k=top_k or settings.top_k
    )


def retrieve_above_threshold(
    query: str,
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
) -> List[Document]:
    settings = get_settings()
    threshold = score_threshold if score_threshold is not None else settings.score_threshold
    return [doc for doc, score in retrieve_with_scores(query, top_k) if score >= threshold]
