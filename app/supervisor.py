# =============================================================================
# app/supervisor.py — Supervisor e grafo principale
# =============================================================================
# Il Supervisor è il grafo master che coordina i due sotto-agenti documentali.
#
# Flusso:
#   START → supervisor_node → route_to_agent
#         → (documenti_contrattuali | documenti_tecnici_e_sistema)
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
# =============================================================================

from typing import Annotated, Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from app.config import get_settings, get_llm
from app.models import RoutingDecision
from app.prompts import SUPERVISOR_PROMPT
from app.agents import build_contrattuali_agent, build_tecnici_agent
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

    messages:        history completa — add_messages accumula invece di sovrascrivere
    summary:         riassunto cumulativo della conversazione (da memory.py)
    routing:         decisione del Supervisor (agente + motivazione + query riformulata)
    agente_usato:    nome del sotto-agente che ha risposto (per ChatResponse)
    tools_usati:     lista dei tool chiamati durante la risposta
    user_id:         UUID utente — usato per costruire il thread_id dei sotto-agenti
    conversation_id: UUID conversazione — idem
    """
    messages        : Annotated[list, add_messages]
    summary         : str
    routing         : RoutingDecision | None
    agente_usato    : str
    tools_usati     : list[str]
    user_id         : str
    conversation_id : str


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
    "contrattuali_node", "tecnici_node"
]:
    """
    Arco condizionale che legge state["routing"].agente e smista.

    LangGraph richiede una funzione separata per gli archi condizionali —
    non può usare direttamente supervisor_node come arco.

    Nessun ramo di default: RoutingDecision.agente è un Literal sulle due
    rotte, quindi un nome inatteso fallisce già in validazione Pydantic
    dentro supervisor_node. Un KeyError qui significherebbe che il Literal
    e questa mappa sono andati fuori sincrono — meglio rumoroso che silenzioso.
    """
    mapping = {
        "documenti_contrattuali"      : "contrattuali_node",
        "documenti_tecnici_e_sistema" : "tecnici_node",
    }
    return mapping[state["routing"].agente]


# =============================================================================
# NODI AGENTE — wrapper che invocano i sotto-agenti
# =============================================================================

# Gli agenti vengono costruiti una volta sola — il Supervisor li riusa
# Il build avviene dentro build_supervisor() per evitare import circolari
_contrattuali_agent = None
_tecnici_agent      = None


def _make_thread_id(state: SupervisorState, agent_name: str) -> str:
    """
    Costruisce un thread_id univoco per ogni sotto-agente per ogni conversazione.

    Formato: {user_id}_{conversation_id}_{agent_name}
    Es: "uuid-utente_uuid-conv_documenti_contrattuali"

    Questo garantisce che ogni sotto-agente mantenga memoria isolata —
    l'agente contrattuale di utente A non condivide stato con quello di
    utente B, né con lo stesso agente in un'altra conversazione.
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


def _run_document_agent(
    state      : SupervisorState,
    agent,
    agent_name : str,
    label      : str,
) -> dict:
    """
    Invoca un sotto-agente documentale con la query riformulata dal Supervisor
    ed estrae risposta finale e tool usati.

    I due nodi differiscono solo per l'agente invocato e per il nome
    registrato in agente_usato: la meccanica sta qui una volta sola.
    """
    routing   = state["routing"]
    thread_id = _make_thread_id(state, agent_name)
    config    = {"configurable": {"thread_id": thread_id}}

    result = agent.invoke(
        {"messages": [HumanMessage(content=routing.query_riformulata)],
         "summary" : state.get("summary", "")},
        config=config,
    )

    risposta   = _extract_final_response(result)
    tools_used = _extract_tools_used(result)

    logger.info(f"[{label}] Risposta generata | Tool usati: {tools_used}")

    return {
        "messages"    : [AIMessage(content=risposta)],
        "agente_usato": agent_name,
        "tools_usati" : tools_used,
    }


def contrattuali_node(state: SupervisorState) -> dict:
    """Invoca l'agente sui documenti contrattuali (capitolati, listini)."""
    return _run_document_agent(
        state, _contrattuali_agent, "documenti_contrattuali", "Contrattuali Node"
    )


def tecnici_node(state: SupervisorState) -> dict:
    """Invoca l'agente su schede tecniche, procedure HACCP e non conformita."""
    return _run_document_agent(
        state, _tecnici_agent, "documenti_tecnici_e_sistema", "Tecnici Node"
    )


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
    global _contrattuali_agent, _tecnici_agent

    # Inizializza i sotto-agenti una volta sola
    logger.info("[Supervisor] Inizializzazione sotto-agenti...")
    _contrattuali_agent = build_contrattuali_agent()
    _tecnici_agent      = build_tecnici_agent()
    logger.info("[Supervisor] Sotto-agenti pronti")

    # Costruisce il grafo del Supervisor
    graph = StateGraph(SupervisorState)

    # Aggiunge i nodi
    graph.add_node("supervisor",        supervisor_node)
    graph.add_node("contrattuali_node", contrattuali_node)
    graph.add_node("tecnici_node",      tecnici_node)
    graph.add_node("summarize",         summarize_node)

    # Archi fissi
    graph.add_edge(START, "supervisor")

    # Arco condizionale: supervisor → route_to_agent → nodo corretto
    graph.add_conditional_edges("supervisor", route_to_agent)

    # Arco post-agente: ogni nodo → summarize o END
    for node in ["contrattuali_node", "tecnici_node"]:
        graph.add_conditional_edges(node, should_summarize_edge, {
            "summarize": "summarize",
            END        : END,
        })

    graph.add_edge("summarize", END)

    return graph.compile()