from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

_SYSTEM = """You are a helpful assistant that answers questions using only the provided context.
If the context does not contain enough information, say so clearly. Never fabricate facts."""

_HUMAN = """Context:
{context}

Question: {question}

Answer:"""

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _SYSTEM), ("human", _HUMAN)]
)


def format_context(documents: List[Document]) -> str:
    if not documents:
        return ""
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in documents
    )
