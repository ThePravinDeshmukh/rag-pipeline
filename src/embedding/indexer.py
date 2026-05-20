from pathlib import Path
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import get_settings
from src.embedding.embedder import get_embedder


def _get_store(collection_name: Optional[str] = None) -> Chroma:
    settings = get_settings()
    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection_name or settings.collection_name,
        embedding_function=get_embedder(),
        persist_directory=settings.chroma_persist_dir,
    )


def add_documents(documents: List[Document], collection_name: Optional[str] = None) -> int:
    if not documents:
        return 0
    _get_store(collection_name).add_documents(documents)
    return len(documents)


def get_store(collection_name: Optional[str] = None) -> Chroma:
    return _get_store(collection_name)


def clear_collection(collection_name: Optional[str] = None) -> None:
    _get_store(collection_name).delete_collection()
