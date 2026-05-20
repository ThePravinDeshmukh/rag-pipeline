from fastapi import APIRouter, HTTPException, status

from src.api.schemas import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    MetricsResponse,
    QueryRequest,
    QueryResponse,
)
from src.embedding.indexer import clear_collection
from src.evaluation.monitor import get_recent_queries, get_stats
from src.orchestration.pipeline import RAGPipeline

router = APIRouter()

# Module-level singleton — one pipeline per worker process.
_pipeline = RAGPipeline()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
def ingest_documents(request: IngestRequest) -> IngestResponse:
    try:
        count = _pipeline.ingest(
            sources=request.sources,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    return IngestResponse(chunks_indexed=count, message=f"Successfully indexed {count} chunks.")


@router.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest) -> QueryResponse:
    try:
        result = _pipeline.query(request.question)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    return QueryResponse(
        question=request.question,
        answer=result["answer"],
        sources=result["sources"],
        latency_ms=result["latency_ms"],
    )


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics() -> MetricsResponse:
    return MetricsResponse(stats=get_stats(), recent_queries=get_recent_queries())


@router.delete("/collection", status_code=status.HTTP_204_NO_CONTENT)
def delete_collection() -> None:
    try:
        clear_collection()
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
