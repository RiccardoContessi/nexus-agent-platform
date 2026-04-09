# =============================================================================
# app/supervisor.py — Supervisor e grafo principale
# =============================================================================
# Il Supervisor è il grafo master che coordina i quattro sotto-agenti.
#
# Flusso:
#   START → supervisor_node → route_to_agent → (hr|ml|report|calendar)
#         → (summarize_node?) → END
#
# Responsabilità:
#   supervisor_node  → legge la richiesta, produce RoutingDecision strutturata
#   route_to_agent() → arco condizionale che smista verso il sotto-agente giusto
#   *_node           → invoca il sotto-agente, estrae risposta e tool usati
#   summarize_node   → comprime la history se supera la soglia (da memory.py)
#
# Decisioni architetturali chiave:
#   - with_structured_output(RoutingDecision): routing deterministico via Pydantic
#   - Thread ID composto: ogni sotto-agente ha memoria isolata per utente/conversazione
#   - Calendar node rileva interrupt e popola pending_calendar_event
# =============================================================================

import json
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from app.config import get_settings, get_llm
from app.models import RoutingDecision
from app.prompts import SUPERVISOR_PROMPT
from app.agents import (
    build_hr_agent,
    build_ml_agent,
    build_report_agent,
    build_calendar_agent,
)
from app.memory import should_summarize, summarize_node
import logging
logger = logging.getLogger(__name__)

settings = get_settings()


# =============================================================================
# SUPERVISOR STATE
# =============================================================================

class SupervisorState(TypedDict):
    """
    State condiviso del grafo Supervisor.

    messages:               history completa — add_messages accumula invece di sovrascrivere
    summary:                riassunto cumulativo della conversazione (da memory.py)
    routing:                decisione del Supervisor (agente + motivazione + query riformulata)
    agente_usato:           nome del sotto-agente che ha risposto (per ChatResponse)
    tools_usati:            lista dei tool chiamati durante la risposta
    pending_calendar_event: dettagli evento da approvare via HITL (None se non calendario)
    user_id:                UUID utente — usato per costruire il thread_id dei sotto-agenti
    conversation_id:        UUID conversazione — idem
    """
    messages               : Annotated[list, add_messages]
    summary                : str
    routing                : RoutingDecision | None
    agente_usato           : str
    tools_usati            : list[str]
    pending_calendar_event : dict | None
    user_id                : str
    conversation_id        : str


# =============================================================================
# NODO SUPERVISOR
# =============================================================================

def supervisor_node(state: SupervisorState) -> dict:
    """
    Analizza la richiesta e produce una RoutingDecision strutturata.

    Usa with_structured_output(RoutingDecision) invece di JsonOutputParser:
    → Pydantic valida l'output dell'LLM prima che entri nello state
    → Se l'LLM produce JSON malformato, lancia eccezione esplicita
    → Il campo 'agente' è un Literal — nessun valore inatteso può passare

    Include il summary come contesto storico nel prompt — il Supervisor
    conosce la storia della conversazione anche dopo la compressione.
    """
    llm      = get_llm(temperature=0)
    messages = state.get("messages", [])
    summary  = state.get("summary", "")

    # Arricchisce il prompt del Supervisor con il contesto storico
    system_content = SUPERVISOR_PROMPT
    if summary:
        system_content += (
            f"\n\n--- Contesto conversazione precedente ---\n"
            f"{summary}\n"
            f"--- Fine contesto ---"
        )

    # with_structured_output forza l'LLM a produrre un RoutingDecision Pydantic valido
    structured_llm = llm.with_structured_output(RoutingDecision)
    routing        = structured_llm.invoke(
        [SystemMessage(content=system_content)] + messages
    )

    logger.info(f"[Supervisor] Routing → {routing.agente} | {routing.motivazione}")

    return {"routing": routing}


# =============================================================================
# ROUTING CONDIZIONALE
# =============================================================================

def route_to_agent(state: SupervisorState) -> Literal[
    "hr_node", "ml_node", "report_node", "calendar_node"
]:
    """
    Arco condizionale che legge state["routing"].agente e smista.

    LangGraph richiede una funzione separata per gli archi condizionali —
    non può usare direttamente supervisor_node come arco.
    """
    agente = state["routing"].agente
    mapping = {
        "hr_agent"      : "hr_node",
        "ml_agent"      : "ml_node",
        "report_agent"  : "report_node",
        "calendar_agent": "calendar_node",
    }
    return mapping[agente]


# =============================================================================
# NODI AGENTE — wrapper che invocano i sotto-agenti
# =============================================================================

# Gli agenti vengono costruiti una volta sola — il Supervisor li riusa
# Il build avviene dentro build_supervisor() per evitare import circolari
_hr_agent       = None
_ml_agent       = None
_report_agent   = None
_calendar_agent = None


def _make_thread_id(state: SupervisorState, agent_name: str) -> str:
    """
    Costruisce un thread_id univoco per ogni sotto-agente per ogni conversazione.

    Formato: {user_id}_{conversation_id}_{agent_name}
    Es: "uuid-utente_uuid-conv_hr_agent"

    Questo garantisce che ogni sotto-agente mantenga memoria isolata —
    l'HR Agent di utente A non condivide stato con quello di utente B,
    e nemmeno con l'HR Agent della stessa conversazione se interrogato due volte.
    """
    return f"{state['user_id']}_{state['conversation_id']}_{agent_name}"


def _extract_final_response(agent_result: dict) -> str:
    """
    Estrae l'ultimo AIMessage dalla history del sotto-agente.
    È la risposta finale da inserire nella history del Supervisor.
    """
    messages = agent_result.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return "Nessuna risposta generata."


def _extract_tools_used(agent_result: dict) -> list[str]:
    """
    Estrae i nomi dei tool chiamati dai ToolMessage nella history del sotto-agente.
    Usati per popolare tools_usati nello state e nella ChatResponse.
    """
    messages   = agent_result.get("messages", [])
    tools_used = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            # Il nome del tool è nell'attributo name del ToolMessage
            tool_name = getattr(msg, "name", None)
            if tool_name and tool_name not in tools_used:
                tools_used.append(tool_name)
    return tools_used


def hr_node(state: SupervisorState) -> dict:
    """
    Invoca l'HR Agent con la query riformulata dal Supervisor.
    Estrae risposta finale e tool usati.
    """
    routing   = state["routing"]
    thread_id = _make_thread_id(state, "hr_agent")
    config    = {"configurable": {"thread_id": thread_id}}

    result = _hr_agent.invoke(
        {"messages": [HumanMessage(content=routing.query_riformulata)],
         "summary" : state.get("summary", "")},
        config=config,
    )

    risposta   = _extract_final_response(result)
    tools_used = _extract_tools_used(result)

    logger.info(f"[HR Node] Risposta generata | Tool usati: {tools_used}")

    return {
        "messages"    : [AIMessage(content=risposta)],
        "agente_usato": "hr_agent",
        "tools_usati" : tools_used,
    }


def ml_node(state: SupervisorState) -> dict:
    """Invoca l'ML Agent."""
    routing   = state["routing"]
    thread_id = _make_thread_id(state, "ml_agent")
    config    = {"configurable": {"thread_id": thread_id}}

    result = _ml_agent.invoke(
        {"messages": [HumanMessage(content=routing.query_riformulata)],
         "summary" : state.get("summary", "")},
        config=config,
    )

    risposta   = _extract_final_response(result)
    tools_used = _extract_tools_used(result)

    logger.info(f"[ML Node] Risposta generata | Tool usati: {tools_used}")

    return {
        "messages"    : [AIMessage(content=risposta)],
        "agente_usato": "ml_agent",
        "tools_usati" : tools_used,
    }


def report_node(state: SupervisorState) -> dict:
    """Invoca il Report Agent."""
    routing   = state["routing"]
    thread_id = _make_thread_id(state, "report_agent")
    config    = {"configurable": {"thread_id": thread_id}}

    result = _report_agent.invoke(
        {"messages": [HumanMessage(content=routing.query_riformulata)],
         "summary" : state.get("summary", "")},
        config=config,
    )

    risposta   = _extract_final_response(result)
    tools_used = _extract_tools_used(result)

    logger.info(f"[Report Node] Report generato | Tool usati: {tools_used}")

    return {
        "messages"    : [AIMessage(content=risposta)],
        "agente_usato": "report_agent",
        "tools_usati" : tools_used,
    }


def calendar_node(state: SupervisorState) -> dict:
    """
    Invoca il Calendar Agent — gestisce il caso di interrupt HITL.

    Due scenari possibili:
      A) LLM chiede chiarimenti (es. manca l'ora) → risponde normalmente
      B) LLM chiama create_calendar_event → grafo si sospende su interrupt_before

    Nel caso B, get_state(config).next non è vuoto — il grafo è sospeso.
    Il nodo legge i dettagli dell'evento dall'ultimo AIMessage e popola
    pending_calendar_event nello state del Supervisor.
    main.py rileverà pending_calendar_event e risponderà con status=pending_approval.
    """
    routing   = state["routing"]
    thread_id = _make_thread_id(state, "calendar_agent")
    config    = {"configurable": {"thread_id": thread_id}}

    # Invoca il Calendar Agent — si ferma sull'interrupt se decide di creare l'evento
    _calendar_agent.invoke(
        {"messages": [HumanMessage(content=routing.query_riformulata)]},
        config=config,
    )

    # Controlla se il grafo è sospeso sull'interrupt
    agent_state = _calendar_agent.get_state(config)
    is_suspended = bool(agent_state.next)  # next non vuoto = grafo sospeso

    if is_suspended:
        # Il Calendar Agent vuole chiamare create_calendar_event — aspetta approvazione
        # Estrae i dettagli dell'evento dall'ultimo AIMessage o ToolCall
        event_details = _extract_pending_event(agent_state)

        logger.info(f"[Calendar Node] HITL attivato — evento in attesa di approvazione")
        logger.info(f"[Calendar Node] Dettagli: {event_details}")

        return {
            "agente_usato"           : "calendar_agent",
            "tools_usati"            : ["create_calendar_event"],
            "pending_calendar_event" : event_details,
            # Nessun AIMessage — main.py risponde con pending_approval
        }
    else:
        # Il Calendar Agent ha risposto normalmente (chiarimenti, info, ecc.)
        messages   = agent_state.values.get("messages", [])
        risposta   = _extract_final_response({"messages": messages})
        tools_used = _extract_tools_used({"messages": messages})

        logger.info(f"[Calendar Node] Risposta diretta (no interrupt)")

        return {
            "messages"               : [AIMessage(content=risposta)],
            "agente_usato"           : "calendar_agent",
            "tools_usati"            : tools_used,
            "pending_calendar_event" : None,
        }


def _extract_pending_event(agent_state) -> dict:
    """
    Estrae i dettagli dell'evento dal tool_call presente nell'ultimo AIMessage
    del Calendar Agent quando il grafo è sospeso su interrupt_before=["tools"].

    Quando LangGraph si ferma su interrupt_before=["tools"], l'ultimo messaggio
    nella history è un AIMessage con tool_calls — contiene i parametri che
    l'LLM aveva preparato per create_calendar_event.
    """
    messages = agent_state.values.get("messages", [])

    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                if tool_call.get("name") == "create_calendar_event":
                    args = tool_call.get("args", {})
                    return {
                        "titolo"     : args.get("titolo", "Evento senza titolo"),
                        "data"       : args.get("data", ""),
                        "ora_inizio" : args.get("ora_inizio", ""),
                        "ora_fine"   : args.get("ora_fine", ""),
                        "descrizione": args.get("descrizione", ""),
                    }

    # Fallback — non dovrebbe succedere se il grafo è sospeso correttamente
    return {"titolo": "", "data": "", "ora_inizio": "", "ora_fine": "", "descrizione": ""}


# =============================================================================
# ARCO POST-AGENTE — decide se comprimere la history
# =============================================================================

def should_summarize_edge(state: SupervisorState) -> Literal["summarize", END]:
    """
    Arco condizionale dopo ogni sotto-agente.
    Se la history ha superato summary_threshold → comprime.
    Altrimenti → END direttamente.
    """
    if should_summarize(state):
        logger.info(f"[Supervisor] Soglia summary raggiunta ({len(state['messages'])} msg) → compressione")
        return "summarize"
    return END


# =============================================================================
# BUILD SUPERVISOR — costruisce e compila il grafo
# =============================================================================

def build_supervisor():
    """
    Costruisce il grafo Supervisor completo.

    Chiamato una sola volta nel lifespan FastAPI e salvato in app.state.supervisor.
    I sotto-agenti vengono inizializzati qui — costoso da costruire, riusato ad ogni richiesta.

    Returns:
        Grafo LangGraph compilato pronto per .invoke() e .astream_events()
    """
    global _hr_agent, _ml_agent, _report_agent, _calendar_agent

    # Inizializza i sotto-agenti una volta sola
    logger.info("[Supervisor] Inizializzazione sotto-agenti...")
    _hr_agent       = build_hr_agent()
    _ml_agent       = build_ml_agent()
    _report_agent   = build_report_agent()
    _calendar_agent = build_calendar_agent()
    logger.info("[Supervisor] Sotto-agenti pronti")

    # Costruisce il grafo del Supervisor
    graph = StateGraph(SupervisorState)

    # Aggiunge i nodi
    graph.add_node("supervisor",    supervisor_node)
    graph.add_node("hr_node",       hr_node)
    graph.add_node("ml_node",       ml_node)
    graph.add_node("report_node",   report_node)
    graph.add_node("calendar_node", calendar_node)
    graph.add_node("summarize",     summarize_node)

    # Archi fissi
    graph.add_edge(START, "supervisor")

    # Arco condizionale: supervisor → route_to_agent → nodo corretto
    graph.add_conditional_edges("supervisor", route_to_agent)

    # Arco post-agente: ogni nodo → summarize o END
    for node in ["hr_node", "ml_node", "report_node", "calendar_node"]:
        graph.add_conditional_edges(node, should_summarize_edge, {
            "summarize": "summarize",
            END        : END,
        })

    graph.add_edge("summarize", END)

    return graph.compile()