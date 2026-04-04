from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from langchain_openai import ChatOpenAI
import openai

class Settings(BaseSettings):
    # --- Campi obbligatori ---
    openai_api_key              : str
    groq_api_key                : str
    pinecone_api_key            : str
    pinecone_index              : str
    postgres_url                : str
    jwt_secret_key              : str
    google_service_account_path : str

    # --- Campi opzionali con default ---
    llm_model                   : str  = "gpt-4o-mini"
    fallback_model              : str  = "llama-3.1-8b-instant"
    fallback_base_url           : str  = "https://api.groq.com/openai/v1"
    embedding_model             : str  = "text-embedding-3-small"
    retriever_k                 : int  = 5
    jwt_algorithm               : str  = "HS256"
    access_token_expire_minutes : int  = 60
    refresh_token_expire_days   : int  = 7
    summary_threshold           : int  = 10
    app_name                    : str  = "Enterprise Agent Platform"
    app_version                 : str  = "1.0.0"
    use_prompt_repetition       : bool = True

    model_config = SettingsConfigDict(
        env_file          = Path(__file__).parent.parent / ".env",
        env_file_encoding = "utf-8",
        extra             = "ignore",
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()

# testo OpenAi, in caso di fallback uso Groq
# ha stessa struttura api di OpenAi, quindi posso usare oggetto ChatOpenAI per entrambi
def get_llm(temperature: float = 0) -> ChatOpenAI:
    settings = get_settings()
    try:
        llm = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=temperature,
        )
        return llm
    except (openai.AuthenticationError, openai.RateLimitError):
        return ChatOpenAI(
            model=settings.fallback_model,
            api_key=settings.groq_api_key,
            base_url=settings.fallback_base_url,
            temperature=temperature,
        )