import base64
import os
import tempfile
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
    google_calendar_id          : str = "primary"

    # --- Credenziali Google: una delle due forme è richiesta ---
    # Locale/Docker: path al file JSON Service Account montato come volume
    google_service_account_path : str = ""
    # Railway/cloud: contenuto JSON Service Account codificato in base64
    # (i provider cloud non permettono volumi per file segreti)
    google_credentials_b64      : str = ""

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


# =============================================================================
# Google Service Account — risoluzione path multi-ambiente
# =============================================================================

# Path del file temporaneo materializzato dal contenuto base64.
# Cache module-level: la decodifica avviene una volta sola per processo,
# le chiamate successive riusano lo stesso file.
_GOOGLE_CREDS_TMP_PATH: str | None = None


def get_google_credentials_path() -> str:
    """
    Restituisce il path al JSON Service Account di Google, scegliendo
    automaticamente la fonte in base all'ambiente:

      1. Se GOOGLE_CREDENTIALS_B64 è valorizzato (deploy cloud, es. Railway):
         decodifica il contenuto da base64 e lo materializza in un file
         temporaneo (/tmp/google_credentials.json o equivalente OS).
         Restituisce il path del file temporaneo.

      2. Se GOOGLE_SERVICE_ACCOUNT_PATH è valorizzato (locale/Docker con
         volume montato): restituisce il path direttamente, senza modifiche.

      3. Se nessuno dei due è impostato: solleva ValueError con messaggio
         esplicito su come configurare l'ambiente.

    Il path tmp è cachato a livello di modulo per non riscrivere il file
    ad ogni chiamata — la decodifica avviene solo alla prima invocazione.
    """
    global _GOOGLE_CREDS_TMP_PATH

    settings = get_settings()

    # ── Caso 1: credenziali in base64 (cloud deploy) ─────────────────────────
    if settings.google_credentials_b64:
        if _GOOGLE_CREDS_TMP_PATH and os.path.exists(_GOOGLE_CREDS_TMP_PATH):
            return _GOOGLE_CREDS_TMP_PATH

        try:
            decoded_bytes = base64.b64decode(settings.google_credentials_b64)
        except Exception as e:
            raise ValueError(
                f"GOOGLE_CREDENTIALS_B64 non è una stringa base64 valida: {e}"
            ) from e

        # Scrive in /tmp su Linux, in %TEMP% su Windows — gestito da tempfile
        tmp_dir  = tempfile.gettempdir()
        tmp_path = os.path.join(tmp_dir, "google_credentials.json")
        with open(tmp_path, "wb") as f:
            f.write(decoded_bytes)

        _GOOGLE_CREDS_TMP_PATH = tmp_path
        return tmp_path

    # ── Caso 2: path locale (Docker volume) ──────────────────────────────────
    if settings.google_service_account_path:
        return settings.google_service_account_path

    # ── Caso 3: nessuna credenziale configurata ──────────────────────────────
    raise ValueError(
        "Credenziali Google mancanti. Imposta una di queste variabili "
        "d'ambiente:\n"
        "  - GOOGLE_CREDENTIALS_B64: contenuto del JSON Service Account "
        "codificato in base64 (consigliato per deploy cloud come Railway)\n"
        "  - GOOGLE_SERVICE_ACCOUNT_PATH: path al file JSON Service Account "
        "(consigliato per esecuzione locale/Docker con volume montato)"
    )

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