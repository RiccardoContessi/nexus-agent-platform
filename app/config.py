from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from langchain_openai import ChatOpenAI

class Settings(BaseSettings):
    # --- Campi obbligatori ---
    openai_api_key              : str
    pinecone_api_key            : str
    pinecone_index              : str
    postgres_url                : str
    jwt_secret_key              : str

    # --- Campi opzionali con default ---
    llm_model                   : str  = "gpt-4o-mini"
    embedding_model             : str  = "text-embedding-3-small"
    retriever_k                 : int  = 5
    jwt_algorithm               : str  = "HS256"
    access_token_expire_minutes : int  = 60
    refresh_token_expire_days   : int  = 7
    summary_threshold           : int  = 10
    app_name                    : str  = "Enterprise Agent Platform"
    app_version                 : str  = "1.0.0"
    use_prompt_repetition       : bool = True

    # --- Flashrank (reranking locale) ---
    # Il default di flashrank è la stringa letterale "/tmp", che su Windows
    # risolve in C:\tmp — una directory alla radice del disco, esposta a
    # qualsiasi pulizia. Il modello (~170 MB) va tenuto in un path durevole
    # fuori dal repo. "~" viene espanso da get_flashrank_cache_dir().
    flashrank_cache_dir         : str  = "~/.flashrank"
    flashrank_model             : str  = "ms-marco-TinyBERT-L-2-v2"

    # Soglia di rilevanza sotto la quale un brano non raggiunge l'LLM.
    # Se nessun brano la supera, il sistema rifiuta SENZA chiamare l'LLM:
    # un modello che non riceve contesto e' un modello che puo' inventare.
    #
    # CALIBRATA su corpus reale, 10 domande (5 documentate, 5 fuori corpus):
    #
    #     documentate    0.9891  0.9934  0.9940  0.9967  0.9991
    #     fuori corpus   0.0008  0.0135  0.1212  0.4273  0.7419
    #                                                    ^^^^^^ "prezzi di listino"
    #
    # Le due classi sono nettamente separate: gap 0.2472 fra 0.7419 e 0.9891.
    # 0.90 sta dentro quel vuoto, tarato STRETTO: margine 0.158 sopra la
    # domanda fuori corpus piu' insidiosa, 0.089 sotto la documentata piu'
    # debole. La scelta e' deliberatamente asimmetrica — un rifiuto di troppo
    # e' recuperabile, una risposta inventata no.
    #
    # La separazione esiste solo DOPO il chunking a campo delle schede
    # tecniche (vedi scripts/ingestion.py): sul chunk-blob precedente le due
    # classi erano interlacciate e NESSUNA soglia le separava.
    rerank_score_threshold      : float = 0.90

    model_config = SettingsConfigDict(
        env_file          = Path(__file__).parent.parent / ".env",
        env_file_encoding = "utf-8",
        extra             = "ignore",
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()


# =============================================================================
# Flashrank — cache del modello di reranking
# =============================================================================

def get_flashrank_cache_dir() -> Path:
    """
    Restituisce il path assoluto della cache flashrank, espandendo "~".

    Non verifica la presenza del modello: quello è compito di
    ensure_flashrank_model(), che va chiamata prima di costruire il Ranker.
    """
    return Path(get_settings().flashrank_cache_dir).expanduser().resolve()


def ensure_flashrank_model() -> Path:
    """
    Verifica che il modello di reranking sia già presente su disco e
    restituisce il path della sua directory.

    Serve a rendere esplicito un fallimento che altrimenti sarebbe silenzioso:
    flashrank, se non trova la directory del modello, la scarica al volo
    (Ranker._prepare_model_dir). In una sala riunioni senza rete quel download
    fallisce all'import, con un errore che non dice cosa manca.

    Solleva RuntimeError con istruzioni di ripristino se il modello è assente.
    """
    settings   = get_settings()
    cache_dir  = get_flashrank_cache_dir()
    model_dir  = cache_dir / settings.flashrank_model

    if not model_dir.is_dir():
        raise RuntimeError(
            f"Modello flashrank '{settings.flashrank_model}' non trovato in "
            f"{model_dir}.\n"
            f"L'applicazione NON lo scarica automaticamente: senza rete "
            f"l'import fallirebbe in modo opaco.\n"
            f"Per ripristinarlo, con rete disponibile:\n"
            f"  python -c \"from flashrank import Ranker; "
            f"Ranker(model_name='{settings.flashrank_model}', "
            f"cache_dir=r'{cache_dir}')\"\n"
            f"Oppure imposta FLASHRANK_CACHE_DIR su una directory che lo contiene."
        )

    return cache_dir


# =============================================================================
# LLM
# =============================================================================

def get_llm(temperature: float = 0) -> ChatOpenAI:
    """
    Costruisce il client LLM.

    Nota: qui esisteva un fallback su Groq avvolto in
    try/except (openai.AuthenticationError, openai.RateLimitError) attorno al
    costruttore di ChatOpenAI. Non ha mai potuto funzionare: il costruttore non
    effettua alcuna chiamata di rete, quindi quelle eccezioni non vengono
    sollevate lì ma a .invoke(). Il ramo except era codice morto.
    È stato rimosso insieme alla dichiarazione corrispondente nel README.
    """
    settings = get_settings()
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        temperature=temperature,
    )