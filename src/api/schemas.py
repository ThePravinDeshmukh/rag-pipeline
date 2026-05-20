from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    sources: List[str] = Field(..., description="File paths, directory paths, or URLs to ingest")
    chunk_size: Optional[int] = Field(None, gt=0, description="Characters per chunk")
    chunk_overlap: Optional[int] = Field(None, ge=0, description="Overlap between consecutive chunks")


class IngestResponse(BaseModel):
    chunks_indexed: int
    message: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural-language question")


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    latency_ms: int


class MetricsResponse(BaseModel):
    stats: Dict[str, Any]
    recent_queries: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
