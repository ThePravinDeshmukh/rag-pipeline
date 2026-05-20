from abc import ABC, abstractmethod

from langchain_core.embeddings import Embeddings

from src.config import get_settings


class BaseEmbedder(ABC):
    @abstractmethod
    def get_embeddings(self) -> Embeddings: ...


class OpenAIEmbedder(BaseEmbedder):
    def get_embeddings(self) -> Embeddings:
        from langchain_openai import OpenAIEmbeddings

        settings = get_settings()
        return OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )


class HuggingFaceEmbedder(BaseEmbedder):
    def get_embeddings(self) -> Embeddings:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        settings = get_settings()
        return HuggingFaceEmbeddings(model_name=settings.embedding_model)


_PROVIDERS: dict[str, BaseEmbedder] = {
    "openai": OpenAIEmbedder(),
    "huggingface": HuggingFaceEmbedder(),
}


def get_embedder() -> Embeddings:
    settings = get_settings()
    provider = _PROVIDERS.get(settings.embedding_provider)
    if provider is None:
        raise ValueError(
            f"Unknown embedding provider: '{settings.embedding_provider}'. "
            f"Available: {list(_PROVIDERS)}"
        )
    return provider.get_embeddings()
