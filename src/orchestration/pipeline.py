import time
from typing import List, Optional

from src.embedding.indexer import add_documents, clear_collection
from src.evaluation.monitor import log_query
from src.ingestion.chunkers import chunk_documents
from src.ingestion.cleaners import clean_documents
from src.ingestion.loaders import load_sources
from src.orchestration.graph import RAGState, build_rag_graph


class RAGPipeline:
    def __init__(self) -> None:
        self._graph = build_rag_graph()

    def ingest(
        self,
        sources: List[str],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> int:
        raw_docs = load_sources(sources)
        cleaned = clean_documents(raw_docs)
        chunks = chunk_documents(cleaned, chunk_size, chunk_overlap)
        return add_documents(chunks)

    def query(self, question: str) -> dict:
        start = time.perf_counter()
        initial: RAGState = {
            "query": question,
            "documents": [],
            "answer": "",
            "rewrite_count": 0,
        }
        final = self._graph.invoke(initial)
        latency_ms = int((time.perf_counter() - start) * 1000)

        log_query(
            query=question,
            answer=final["answer"],
            num_docs=len(final["documents"]),
            latency_ms=latency_ms,
        )

        return {
            "original_query": question,
            "query": final["query"],
            "answer": final["answer"],
            "sources": [doc.metadata.get("source", "unknown") for doc in final["documents"]],
            "latency_ms": latency_ms,
        }

    def clear(self) -> None:
        clear_collection()
