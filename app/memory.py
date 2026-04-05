# =============================================================================
# app/memory.py — Summary Memory e persistenza su PostgreSQL
# =============================================================================
# Gestisce il ciclo di vita della memoria conversazionale:
import uuid
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from app.config import get_settings, get_llm
from app.models import Conversation, Message
from app.prompts import SUMMARIZE_PROMPT

settings = get_settings()


# =============================================================================
# 1. SHOULD SUMMARIZE — quando comprimere
# =============================================================================

def should_summarize(state: dict) -> bool:
    """
    Controlla se la history ha superato la soglia di compressione.

    Viene chiamata come arco condizionale nel grafo del Supervisor:
      → se True  → summarize_node → END
      → se False → END direttamente

    settings.summary_threshold (default 10): ogni 10 messaggi comprimi.
    Puoi abbassarlo per risparmiare token, alzarlo per più contesto diretto.
    """
    messages = state.get("messages", [])
    return len(messages) >= settings.summary_threshold


# =============================================================================
# 2. SUMMARIZE NODE — compressione cumulativa
# =============================================================================

def summarize_node(state: dict) -> dict:
    """
    Nodo LangGraph che comprime la history in un riassunto cumulativo.

    Strategia:
      - Prende TUTTI i messaggi tranne gli ultimi 4
      - Li passa all'LLM insieme al summary precedente
      - Il nuovo summary integra quello vecchio + i messaggi compressi
      - Restituisce lo state aggiornato con summary nuovo e history ridotta

    Gli ultimi messaggi sono il contesto immediato della conversazione corrente.
    Comprimerli subito farebbe perdere il filo del dialogo in corso.
    4 = 2 turni completi (human + ai) — sufficiente per la continuità.

    Returns:
        dict con i campi dello state aggiornati — LangGraph li merge automaticamente
    """
    messages    = state.get("messages", [])
    old_summary = state.get("summary", "")

    # Messaggi da comprimere: tutti tranne gli ultimi 4
    messages_to_compress = messages[:-4]
    messages_to_keep     = messages[-4:]

    if not messages_to_compress:
        # Nulla da comprimere — non dovrebbe succedere, ma gestito per sicurezza
        return {}

    # Costruisce il testo della history da comprimere
    history_text = _messages_to_text(messages_to_compress)

    # Prompt cumulativo: passa il summary precedente + nuovi messaggi
    # L'LLM integra entrambi nel nuovo riassunto — nulla viene perso
    prompt = f"""{SUMMARIZE_PROMPT}

Riassunto esistente della conversazione (può essere vuoto):
{old_summary if old_summary else "Nessun riassunto precedente."}

Nuovi messaggi da integrare nel riassunto:
{history_text}

Produci il nuovo riassunto cumulativo:"""

    llm          = get_llm(temperature=0)
    new_summary  = llm.invoke(prompt).content.strip()

    print(f"[Memory] Compressi {len(messages_to_compress)} messaggi → summary aggiornato")
    print(f"[Memory] Messaggi mantenuti: {len(messages_to_keep)}")

    # Restituisce solo i campi che cambiano — LangGraph fa il merge con il resto
    return {
        "messages": messages_to_keep,    # history ridotta
        "summary" : new_summary,         # riassunto cumulativo aggiornato
    }


def _messages_to_text(messages: list[BaseMessage]) -> str:
    """
    Converte una lista di LangChain messages in testo leggibile per l'LLM.
    Funzione helper interna — non esposta.
    """
    lines = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            lines.append(f"Utente: {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"Assistente: {msg.content}")
        else:
            # ToolMessage o altri tipi — include il contenuto grezzo
            lines.append(f"Sistema: {msg.content}")
    return "\n".join(lines)


# =============================================================================
# 3. SAVE TO DB — persistenza su PostgreSQL
# =============================================================================

async def save_to_db(
    db             : AsyncSession,
    conversation_id: uuid.UUID,
    new_messages   : list[dict],
    summary        : str,
) -> None:
    """
    Salva i nuovi messaggi e aggiorna il summary nella Conversation.

    Chiamata da main.py dopo ogni invocazione del Supervisor:
      1. Inserisce i nuovi messaggi nella tabella Message
      2. Aggiorna summary e updated_at nella tabella Conversation

    Args:
        db:              sessione PostgreSQL (da Depends(get_db))
        conversation_id: UUID della conversazione corrente
        new_messages:    lista di dict con role, content, agente_usato, tools_usati
                         (i messaggi generati in questa invocazione del Supervisor)
        summary:         il summary aggiornato dallo state (può essere "" se non compresso)
    """
    # ── Inserisce i nuovi messaggi ────────────────────────────────────────────
    for msg in new_messages:
        db_message = Message(
            conversation_id=conversation_id,
            role           =msg["role"],           # "human" o "ai"
            content        =msg["content"],
            agente_usato   =msg.get("agente_usato"),   # nullable
            tools_usati    =msg.get("tools_usati"),    # JSON nullable
        )
        db.add(db_message)

    # ── Aggiorna il summary nella Conversation ────────────────────────────────
    # updated_at viene aggiornato automaticamente da onupdate=func.now()
    await db.execute(
        update(Conversation)
        .where(Conversation.id == conversation_id)
        .values(summary=summary)
    )

    # Il commit è gestito automaticamente da get_db() in database.py


# =============================================================================
# 4. LOAD FROM DB — ricostruzione della history
# =============================================================================

async def load_from_db(
    db             : AsyncSession,
    conversation_id: uuid.UUID,
) -> dict:
    """
    Carica la history e il summary dal DB e ricostruisce lo state LangGraph.

    Punto di ingresso per le conversazioni esistenti:
      - Nuova conversazione → restituisce state vuoto (messages=[], summary="")
      - Conversazione esistente → ricostruisce messages + summary

    Il summary viene incluso nel system prompt degli agenti come contesto
    storico — l'LLM "ricorda" la conversazione anche dopo il restart.

    Returns:
        dict con chiavi "messages" (lista LangChain messages) e "summary" (str)
    """
    # ── Carica la Conversation (per avere il summary) ─────────────────────────
    conv = await db.get(Conversation, conversation_id)

    if conv is None:
        # Conversazione non trovata — state vuoto per nuova conversazione
        return {"messages": [], "summary": ""}

    # ── Carica tutti i messaggi in ordine cronologico ─────────────────────────
    result   = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
    )
    db_messages = result.scalars().all()

    # ── Ricostruisce i LangChain messages ─────────────────────────────────────
    # LangGraph lavora con oggetti HumanMessage e AIMessage — non dict semplici
    lc_messages = []
    for msg in db_messages:
        if msg.role == "human":
            lc_messages.append(HumanMessage(content=msg.content))
        elif msg.role == "ai":
            lc_messages.append(AIMessage(content=msg.content))
        # Altri ruoli (tool, system) vengono ignorati — non servono per la history

    print(f"[Memory] Caricati {len(lc_messages)} messaggi + summary ({len(conv.summary)} chars)")

    return {
        "messages": lc_messages,
        "summary" : conv.summary or "",
    }
