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

import logging
logger = logging.getLogger(__name__)

settings = get_settings()


# Numero massimo di ricerche che un agente puo' fare per una singola domanda.
# 3 e' scelto sui dati della batteria: ogni domanda che ha ricevuto risposta
# corretta ha usato UNA sola ricerca. Le iterazioni successive non hanno mai
# aggiunto informazione — il tool e' deterministico a parita' di query, quindi
# ricercare due volte la stessa cosa restituisce gli stessi brani. 3 lascia
# margine a una riformulazione utile e taglia il ciclo patologico.
MAX_TOOL_ITERATIONS = 3


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

    # LLM SENZA tool, per la sola risposta finale al raggiungimento del tetto.
    # Non e' una variante di configurazione: e' cio' che rende il tetto
    # STRUTTURALE invece che una richiesta cortese al modello. Senza tool nello
    # schema, una quarta ricerca non e' improbabile — e' impossibile.
    llm_finale = get_llm(temperature=0)

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

        # Risultati delle ricerche gia' effettuate, separati fra utili e a vuoto.
        risultati = _tool_results(messages)
        utili     = [r for r in risultati if r.strip() != NESSUN_DOCUMENTO_RILEVANTE]
        context   = "\n\n---\n\n".join(utili)

        # System prompt arricchito con il summary della conversazione
        system = _build_system_with_summary(system_prompt, summary)

        # RIFIUTO SENZA LLM.
        # Il retrieval non ha prodotto alcun brano sopra la soglia di rilevanza.
        # Invocare comunque il modello significherebbe chiedergli di rispondere
        # senza fonti: e' esattamente la condizione in cui inventa. Si esce qui,
        # con un messaggio fisso, senza citazione e senza chiamata all'API.
        #
        # La condizione e' "ci sono state ricerche e NESSUNA ha prodotto nulla",
        # non "l'unico risultato e' il sentinel". La differenza conta: con due
        # ricerche a vuoto il contesto concatenato diventava
        # "__NESSUN_DOCUMENTO_RILEVANTE__\n\n---\n\n__NESSUN_DOCUMENTO_RILEVANTE__",
        # che non e' uguale al sentinel — il confronto falliva, il rifiuto non
        # scattava e il sentinel finiva all'LLM come se fosse un documento.
        if risultati and not utili:
            return {"messages": [AIMessage(content=MESSAGGIO_RIFIUTO)]}

        # TETTO ALLE ITERAZIONI DI RICERCA.
        # Senza questo il ciclo ReAct non ha un limite superiore: se i brani
        # recuperati non rispondono alla domanda, il modello richiama il tool,
        # riceve gli stessi brani e richiama ancora. Misurato in batteria: la
        # domanda "chi effettua il controllo ufficiale" ha fatto 17 ricerche
        # identiche in 75.4s. In una demo di tre minuti non e' imprecisione,
        # e' una schermata bloccata.
        #
        # Al tetto NON si cerca piu'. Se qualcosa e' stato recuperato si
        # risponde da quello, con l'LLM privo di tool; se non c'e' nulla si
        # rifiuta. Rispondere dal contesto gia' in mano, invece di scartarlo,
        # evita di trasformare in rifiuto una domanda cui il primo retrieval
        # aveva gia' risposto: il tetto serve a limitare il tempo, non a
        # buttare via un recupero riuscito.
        if len(risultati) >= MAX_TOOL_ITERATIONS:
            logger.warning(
                f"[Agent] Tetto di {MAX_TOOL_ITERATIONS} ricerche raggiunto "
                f"per '{query}' — risposta forzata dal contesto disponibile "
                f"({len(utili)} risultati utili), nessuna nuova ricerca."
            )
            if not utili:
                return {"messages": [AIMessage(content=MESSAGGIO_RIFIUTO)]}
            final_prompt = build_rag_prompt(query, context, system)
            return {"messages": [llm_finale.invoke([HumanMessage(content=final_prompt)])]}

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

def _tool_results(messages: list) -> list[str]:
    """
    Restituisce il contenuto di ogni ToolMessage nella history dell'agente,
    uno per elemento — cioe' il risultato di ogni ricerca gia' effettuata.

    Restituisce una LISTA e non la stringa concatenata perche' al chiamante
    servono due cose che la concatenazione distrugge: QUANTE ricerche sono
    state fatte (per il tetto sulle iterazioni) e QUALI sono andate a vuoto
    (per distinguere "nessun documento rilevante" da "documenti trovati").

    Lista vuota alla prima iterazione del ciclo ReAct: nessuna ricerca ancora,
    l'LLM chiamera' il tool per farla.
    """
    from langchain_core.messages import ToolMessage

    return [msg.content for msg in messages if isinstance(msg, ToolMessage)]