# =============================================================================
# app/agents.py — Due sotto-agenti documentali LangGraph autonomi
# =============================================================================
# Ogni agente è un grafo LangGraph compilato con ciclo ReAct:
#   START → llm_node → tools_condition → tools → llm_node → ... → END
#
# Entrambi gli agenti usano build_rag_prompt() — il punto esatto dove la
# Prompt Repetition viene applicata o disattivata in base al flag
# settings.use_prompt_repetition. Zero modifiche al codice per l'A/B test.
#
# Agenti:
#   build_contrattuali_agent() → namespace capitolati + listini (in parallelo)
#   build_tecnici_agent()      → schede_tecniche + procedure + non_conformita
# =============================================================================

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # accumula invece di sovrascrivere
    summary : str

from app.config import get_settings, get_llm
from app.tools import (
    contrattuali_tools,
    tecnici_tools,
    NESSUN_DOCUMENTO_RILEVANTE,
    MESSAGGIO_RIFIUTO,
)
from app.prompts import (
    build_rag_prompt,
    CONTRATTUALI_SYSTEM_PROMPT,
    TECNICI_SYSTEM_PROMPT,
)

settings = get_settings()


# =============================================================================
# HELPER — estrae query e context dallo state
# =============================================================================

def _get_last_human_query(state: dict) -> str:
    """
    Estrae l'ultima domanda dell'utente dalla history dei messaggi.
    È la query che il Supervisor ha riformulato e passato all'agente.
    """
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def _build_system_with_summary(base_system: str, summary: str) -> str:
    """
    Arricchisce il system prompt con il riassunto della conversazione
    se presente. Garantisce continuità contestuale tra sessioni.
    """
    if not summary:
        return base_system
    return (
        f"{base_system}\n\n"
        f"--- Contesto della conversazione precedente ---\n"
        f"{summary}\n"
        f"--- Fine contesto ---"
    )


# =============================================================================
# FACTORY — costruisce un agente documentale RAG
# =============================================================================

def _build_document_agent(tools: list, system_prompt: str):
    """
    Costruisce un sotto-agente RAG con ciclo ReAct.

    I due agenti documentali differiscono solo per i tool che possono chiamare
    e per il system prompt: la meccanica del grafo è identica, quindi vive qui
    una volta sola.

    Args:
        tools:         lista dei tool esposti all'agente
        system_prompt: system prompt specifico del dominio documentale
    """
    llm = get_llm(temperature=0).bind_tools(tools)

    def llm_node(state: dict) -> dict:
        """
        Nodo di generazione.

        Costruisce il prompt con build_rag_prompt() che applica
        automaticamente la Prompt Repetition in base al flag Settings.
        Il context viene estratto dai ToolMessage precedenti se disponibili,
        altrimenti l'LLM userà i tool per recuperarlo.
        """
        messages = state.get("messages", [])
        summary  = state.get("summary", "")
        query    = _get_last_human_query(state)

        # Estrae il contesto dai ToolMessage precedenti (risultati del retrieval)
        context = _extract_tool_context(messages)

        # RIFIUTO SENZA LLM.
        # Il retrieval non ha prodotto alcun brano sopra la soglia di rilevanza.
        # Invocare comunque il modello significherebbe chiedergli di rispondere
        # senza fonti: e' esattamente la condizione in cui inventa. Si esce qui,
        # con un messaggio fisso, senza citazione e senza chiamata all'API.
        if context.strip() == NESSUN_DOCUMENTO_RILEVANTE:
            return {"messages": [AIMessage(content=MESSAGGIO_RIFIUTO)]}

        # System prompt arricchito con il summary della conversazione
        system = _build_system_with_summary(system_prompt, summary)

        if context:
            # Abbiamo già i documenti recuperati — costruisce il prompt completo
            # Qui si applica la Prompt Repetition (o meno) in base al flag
            final_prompt = build_rag_prompt(query, context, system)
            response = llm.invoke([HumanMessage(content=final_prompt)])
        else:
            # Prima iterazione — nessun documento ancora recuperato
            # L'LLM decide di chiamare il tool di ricerca (ciclo ReAct)
            system_msg = SystemMessage(content=system)
            response   = llm.invoke([system_msg] + messages)

        return {"messages": [response]}

    # Costruisce il grafo ReAct
    graph = StateGraph(AgentState)
    graph.add_node("llm",   llm_node)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "llm")
    graph.add_conditional_edges("llm", tools_condition)
    graph.add_edge("tools", "llm")

    return graph.compile(checkpointer=MemorySaver())


# =============================================================================
# I DUE AGENTI DOCUMENTALI
# =============================================================================

def build_contrattuali_agent():
    """
    Sotto-agente sui documenti contrattuali e commerciali.
    Usa search_documenti_contrattuali → namespace capitolati + listini,
    interrogati in parallelo con asyncio.gather.
    """
    return _build_document_agent(contrattuali_tools, CONTRATTUALI_SYSTEM_PROMPT)


def build_tecnici_agent():
    """
    Sotto-agente su schede tecniche e documentazione di sistema qualità.
    Usa search_documenti_tecnici_e_sistema → namespace schede_tecniche,
    procedure e non_conformita, interrogati in parallelo con asyncio.gather.
    """
    return _build_document_agent(tecnici_tools, TECNICI_SYSTEM_PROMPT)


# =============================================================================
# HELPER — estrae contesto dai ToolMessage
# =============================================================================

def _extract_tool_context(messages: list) -> str:
    """
    Estrae il contenuto dei ToolMessage dalla history dell'agente.
    Questi messaggi contengono i risultati dei tool (chunk Pinecone reranked).
    Vengono usati come 'context' in build_rag_prompt().

    Se non ci sono ToolMessage (prima iterazione del ciclo ReAct),
    restituisce stringa vuota — l'LLM chiamerà il tool per recuperarli.
    """
    from langchain_core.messages import ToolMessage

    tool_contents = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_contents.append(msg.content)

    return "\n\n---\n\n".join(tool_contents) if tool_contents else ""