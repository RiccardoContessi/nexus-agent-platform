# =============================================================================
# app/agents.py — Quattro sotto-agenti LangGraph autonomi
# =============================================================================
# Ogni agente è un grafo LangGraph compilato con ciclo ReAct:
#   START → llm_node → tools_condition → tools → llm_node → ... → END
#
# Tutti e tre gli agenti RAG usano build_rag_prompt() — il punto esatto
# dove la Prompt Repetition viene applicata o disattivata in base al flag
# settings.use_prompt_repetition. Zero modifiche al codice per l'A/B test.
#
# Agenti:
#   build_hr_agent()       → 3 namespace HR in parallelo
#   build_ml_agent()       → namespace ml_docs
#   build_report_agent()   → HR + ML, output Markdown strutturato
#   build_calendar_agent() → Google Calendar con HITL obbligatorio
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
from app.tools import hr_tools, ml_tools, report_tools, calendar_tools
from app.prompts import (
    build_rag_prompt,
    HR_SYSTEM_PROMPT,
    ML_SYSTEM_PROMPT,
    REPORT_SYSTEM_PROMPT,
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
# HR AGENT
# =============================================================================

def build_hr_agent():
    """
    Sotto-agente specializzato in documenti HR.
    Usa search_hr_documents che interroga 3 namespace Pinecone in parallelo.
    Applica build_rag_prompt() con Prompt Repetition se abilitata.
    """
    llm = get_llm(temperature=0).bind_tools(hr_tools)

    def hr_llm_node(state: dict) -> dict:
        """
        Nodo di generazione dell'HR Agent.

        Costruisce il prompt con build_rag_prompt() che applica
        automaticamente la Prompt Repetition in base al flag Settings.
        Il context viene estratto dai ToolMessage precedenti se disponibili,
        altrimenti l'LLM userà i tool per recuperarlo.
        """
        messages = state.get("messages", [])
        summary  = state.get("summary", "")
        query    = _get_last_human_query(state)

        # Estrae il contesto dai ToolMessage precedenti (risultati di search_hr_documents)
        context = _extract_tool_context(messages)

        # System prompt arricchito con il summary della conversazione
        system = _build_system_with_summary(HR_SYSTEM_PROMPT, summary)

        if context:
            # Abbiamo già i documenti recuperati — costruisce il prompt completo
            # Qui si applica la Prompt Repetition (o meno) in base al flag
            final_prompt = build_rag_prompt(query, context, system)
            response = llm.invoke([HumanMessage(content=final_prompt)])
        else:
            # Prima iterazione — nessun documento ancora recuperato
            # L'LLM decide di chiamare search_hr_documents (ciclo ReAct)
            system_msg = SystemMessage(content=system)
            response   = llm.invoke([system_msg] + messages)

        return {"messages": [response]}

    # Costruisce il grafo ReAct
    graph = StateGraph(AgentState)
    graph.add_node("llm",   hr_llm_node)
    graph.add_node("tools", ToolNode(hr_tools))

    graph.add_edge(START, "llm")
    graph.add_conditional_edges("llm", tools_condition)
    graph.add_edge("tools", "llm")

    return graph.compile(checkpointer=MemorySaver())


# =============================================================================
# ML AGENT
# =============================================================================

def build_ml_agent():
    """
    Sotto-agente specializzato in documenti Machine Learning.
    Stesso pattern dell'HR Agent — usa search_ml_documents.
    """
    llm = get_llm(temperature=0).bind_tools(ml_tools)

    def ml_llm_node(state: dict) -> dict:
        messages = state.get("messages", [])
        summary  = state.get("summary", "")
        query    = _get_last_human_query(state)
        context  = _extract_tool_context(messages)
        system   = _build_system_with_summary(ML_SYSTEM_PROMPT, summary)

        if context:
            # Documenti recuperati — applica Prompt Repetition
            final_prompt = build_rag_prompt(query, context, system)
            response = llm.invoke([HumanMessage(content=final_prompt)])
        else:
            system_msg = SystemMessage(content=system)
            response   = llm.invoke([system_msg] + messages)

        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("llm",   ml_llm_node)
    graph.add_node("tools", ToolNode(ml_tools))

    graph.add_edge(START, "llm")
    graph.add_conditional_edges("llm", tools_condition)
    graph.add_edge("tools", "llm")

    return graph.compile(checkpointer=MemorySaver())


# =============================================================================
# REPORT AGENT
# =============================================================================

def build_report_agent():
    """
    Sotto-agente per report strutturati in Markdown.
    Usa generate_report che incrocia namespace HR e ML.
    Output sempre in formato Markdown con titolo, sezioni e conclusioni.
    """
    llm = get_llm(temperature=0).bind_tools(report_tools)

    def report_llm_node(state: dict) -> dict:
        messages = state.get("messages", [])
        summary  = state.get("summary", "")
        query    = _get_last_human_query(state)
        context  = _extract_tool_context(messages)
        system   = _build_system_with_summary(REPORT_SYSTEM_PROMPT, summary)

        if context:
            # Il tool generate_report restituisce già Markdown strutturato.
            # build_rag_prompt lo usa come context — l'LLM può raffinarlo
            # o restituirlo direttamente se già completo.
            final_prompt = build_rag_prompt(query, context, system)
            response = llm.invoke([HumanMessage(content=final_prompt)])
        else:
            system_msg = SystemMessage(content=system)
            response   = llm.invoke([system_msg] + messages)

        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("llm",   report_llm_node)
    graph.add_node("tools", ToolNode(report_tools))

    graph.add_edge(START, "llm")
    graph.add_conditional_edges("llm", tools_condition)
    graph.add_edge("tools", "llm")

    return graph.compile(checkpointer=MemorySaver())


# =============================================================================
# CALENDAR AGENT
# =============================================================================

def build_calendar_agent():
    """
    Sotto-agente per la gestione del Google Calendar.

    Due differenze fondamentali rispetto agli agenti RAG:
      1. NON usa build_rag_prompt() — non fa retrieval, non ha context
      2. interrupt_before=["tools"] — si ferma SEMPRE prima di chiamare
         create_calendar_event, aspettando approvazione esplicita via HITL

    Il HITL funziona così:
      - LLM decide di chiamare create_calendar_event
      - LangGraph si ferma prima dell'esecuzione (interrupt)
      - main.py rileva lo stato sospeso → risponde con pending_approval
      - Utente approva o rifiuta via POST /v1/approve
      - Se approvato: update_state → invoke(None) → tool eseguito → evento creato
    """
    CALENDAR_SYSTEM = """Sei un assistente per la gestione del calendario aziendale.

Quando l'utente chiede di creare un evento:
1. Estrai: titolo, data (YYYY-MM-DD), ora inizio (HH:MM), ora fine (HH:MM)
2. Se mancano informazioni, chiedi SOLO quelle mancanti
3. Appena hai TUTTI i dati, chiama IMMEDIATAMENTE create_calendar_event
   senza chiedere ulteriori conferme — la conferma avviene dopo tramite il sistema

Formato data: YYYY-MM-DD | Formato ora: HH:MM"""

    llm = get_llm(temperature=0).bind_tools(calendar_tools)

    def calendar_llm_node(state: dict) -> dict:
        """
        Nodo del Calendar Agent.
        Non usa build_rag_prompt — prompt diretto senza context RAG.
        """
        messages   = state.get("messages", [])
        system_msg = SystemMessage(content=CALENDAR_SYSTEM)
        response   = llm.invoke([system_msg] + messages)
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("llm",   calendar_llm_node)
    graph.add_node("tools", ToolNode(calendar_tools))

    graph.add_edge(START, "llm")
    graph.add_conditional_edges("llm", tools_condition)
    graph.add_edge("tools", "llm")

    # interrupt_before=["tools"]: si ferma PRIMA di eseguire ToolNode
    # Garantisce che create_calendar_event non venga mai chiamato senza approvazione
    return graph.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["tools"],
    )


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