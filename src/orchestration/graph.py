from typing import List, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from src.config import get_settings
from src.orchestration._llm import get_llm
from src.retrieval.prompt_builder import RAG_PROMPT, format_context
from src.retrieval.retriever import retrieve_documents

_NO_DOCS_ANSWER = (
    "I could not find relevant information in the knowledge base to answer your question."
)


class RAGState(TypedDict):
    query: str
    documents: List[Document]
    answer: str
    rewrite_count: int


def _retrieve(state: RAGState) -> RAGState:
    docs = retrieve_documents(state["query"])
    return {**state, "documents": docs}


def _generate(state: RAGState) -> RAGState:
    if not state["documents"]:
        return {**state, "answer": _NO_DOCS_ANSWER}

    chain = RAG_PROMPT | get_llm()
    response = chain.invoke(
        {"context": format_context(state["documents"]), "question": state["query"]}
    )
    answer = response.content if hasattr(response, "content") else str(response)
    return {**state, "answer": answer}


def _rewrite_query(state: RAGState) -> RAGState:
    llm = get_llm()
    prompt = (
        "Rewrite this search query to retrieve more relevant documents. "
        "Be specific and focus on key concepts.\n\n"
        f"Original query: {state['query']}\n\nRewritten query:"
    )
    response = llm.invoke(prompt)
    new_query = response.content if hasattr(response, "content") else str(response)
    return {**state, "query": new_query.strip(), "rewrite_count": state.get("rewrite_count", 0) + 1}


def _route_after_retrieve(state: RAGState) -> str:
    settings = get_settings()
    has_docs = bool(state.get("documents"))
    exceeded_rewrites = state.get("rewrite_count", 0) >= settings.max_query_rewrites
    return "generate" if (has_docs or exceeded_rewrites) else "rewrite"


def build_rag_graph() -> StateGraph:
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", _retrieve)
    graph.add_node("generate", _generate)
    graph.add_node("rewrite", _rewrite_query)

    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges(
        "retrieve",
        _route_after_retrieve,
        {"generate": "generate", "rewrite": "rewrite"},
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)

    return graph.compile()
