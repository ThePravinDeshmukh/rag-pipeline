from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    WebBaseLoader,
)

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".txt", ".md", ".csv"})


def load_pdf(path: str) -> List[Document]:
    return PyPDFLoader(path).load()


def load_text(path: str) -> List[Document]:
    return TextLoader(path, encoding="utf-8").load()


def load_csv(path: str) -> List[Document]:
    return CSVLoader(path).load()


def load_url(url: str) -> List[Document]:
    return WebBaseLoader(url).load()


def load_directory(directory: str) -> List[Document]:
    docs: List[Document] = []
    for path in Path(directory).rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                docs.extend(load_file(str(path)))
            except Exception:
                pass
    return docs


def load_file(path: str) -> List[Document]:
    extension = Path(path).suffix.lower()
    _loaders = {
        ".pdf": load_pdf,
        ".txt": load_text,
        ".md": load_text,
        ".csv": load_csv,
    }
    loader_fn = _loaders.get(extension)
    if loader_fn is None:
        raise ValueError(
            f"Unsupported file type '{extension}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    return loader_fn(path)


def load_sources(sources: List[str]) -> List[Document]:
    """Load documents from a mixed list of file paths, directories, and URLs."""
    documents: List[Document] = []
    for source in sources:
        if source.startswith(("http://", "https://")):
            documents.extend(load_url(source))
        elif Path(source).is_dir():
            documents.extend(load_directory(source))
        else:
            documents.extend(load_file(source))
    return documents
