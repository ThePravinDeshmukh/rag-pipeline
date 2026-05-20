from langchain_core.language_models import BaseChatModel

from src.config import get_settings

_PROVIDER_MAP = {
    "openai": "_build_openai",
    "anthropic": "_build_anthropic",
}


def _build_openai() -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    s = get_settings()
    return ChatOpenAI(model=s.llm_model, temperature=s.temperature, openai_api_key=s.openai_api_key)


def _build_anthropic() -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    s = get_settings()
    return ChatAnthropic(
        model=s.llm_model, temperature=s.temperature, anthropic_api_key=s.anthropic_api_key
    )


def get_llm() -> BaseChatModel:
    settings = get_settings()
    builder_name = _PROVIDER_MAP.get(settings.llm_provider)
    if builder_name is None:
        raise ValueError(
            f"Unknown LLM provider: '{settings.llm_provider}'. Available: {list(_PROVIDER_MAP)}"
        )
    return globals()[builder_name]()
