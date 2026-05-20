import re
import unicodedata
from typing import List

from langchain_core.documents import Document

_RE_MULTIPLE_NEWLINES = re.compile(r"\n{3,}")
_RE_MULTIPLE_SPACES = re.compile(r" {2,}")
_RE_NULL_CHARS = re.compile(r"\x00")


def _normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def _normalize_whitespace(text: str) -> str:
    text = _RE_NULL_CHARS.sub("", text)
    text = _RE_MULTIPLE_NEWLINES.sub("\n\n", text)
    text = _RE_MULTIPLE_SPACES.sub(" ", text)
    return text.strip()


def clean_text(text: str) -> str:
    return _normalize_whitespace(_normalize_unicode(text))


def clean_document(doc: Document) -> Document:
    return Document(page_content=clean_text(doc.page_content), metadata=doc.metadata)


def clean_documents(docs: List[Document]) -> List[Document]:
    return [clean_document(doc) for doc in docs if doc.page_content.strip()]
