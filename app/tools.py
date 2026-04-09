# =============================================================================
# app/tools.py — Definizione dei tool per gli agenti LangGraph
# =============================================================================
# Ogni tool è una funzione Python decorata con @tool.
# L'LLM legge la docstring per decidere quando e come usarlo.
# Il tool riceve parametri semplici e restituisce SEMPRE una stringa.
#
# Regola fondamentale: i tool non restituiscono MAI HumanMessage/AIMessage.
# Restituiscono str — LangGraph la wrappa automaticamente in ToolMessage.
#
# Tool esposti:
#   hr_tools       → [search_hr_documents]
#   ml_tools       → [search_ml_documents]
#   report_tools   → [generate_report]
#   calendar_tools → [create_calendar_event]
# =============================================================================

import asyncio
import json
import time

from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from flashrank import Ranker, RerankRequest

from app.config import get_settings, get_llm
import logging
logger = logging.getLogger(__name__)

settings = get_settings()


embeddings = OpenAIEmbeddings(
    model=settings.embedding_model,
    api_key=settings.openai_api_key,
)

# Ranker FlashrankRerank — modello leggero per reranking locale (no API esterne)
ranker = Ranker()


def _get_vectorstore(namespace: str) -> PineconeVectorStore:
    """
    Restituisce un PineconeVectorStore per il namespace specificato.
    Funzione helper interna — non esposta come tool.
    """
    return PineconeVectorStore(
        index_name=settings.pinecone_index,
        embedding=embeddings,
        namespace=namespace,
        pinecone_api_key=settings.pinecone_api_key, 
    )


def _retrieve(namespace: str, query: str, k: int = None) -> list[Document]:
    """
    Esegue una similarity search sincrona su un namespace Pinecone.
    Usata dentro asyncio.to_thread per non bloccare l'event loop.
    """
    k = k or settings.retriever_k
    vs = _get_vectorstore(namespace)
    return vs.similarity_search(query, k=k)


def _rerank(query: str, docs: list[Document], top_n: int = 5) -> list[Document]:
    """
    Applica FlashrankRerank sull'insieme di documenti recuperati.
    Restituisce i top_n documenti riordinati per rilevanza.

    FlashrankRerank usa un cross-encoder leggero che valuta la coppia
    (query, documento) invece dei soli vettori — più preciso del solo embedding.
    """
    if not docs:
        return []

    passages = [
        {"id": i, "text": doc.page_content}
        for i, doc in enumerate(docs)
    ]
    request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(request)[:top_n]

    # Ricostruisce i Document preservando i metadata originali
    return [
        Document(
            page_content=r["text"],
            metadata=docs[r["id"]].metadata,
        )
        for r in results
    ]


def _docs_to_string(docs: list[Document]) -> str:
    """
    Converte una lista di Document in una stringa leggibile dall'LLM.
    Include source e namespace dai metadata per citabilità.
    """
    if not docs:
        return "Nessun documento trovato."

    chunks = []
    for i, doc in enumerate(docs, 1):
        source    = doc.metadata.get("source", "sconosciuta")
        namespace = doc.metadata.get("namespace", "")
        topic     = doc.metadata.get("topic", "")
        header    = f"[Documento {i} | {namespace} | {source}"
        if topic:
            header += f" | {topic}"
        header += "]"
        chunks.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(chunks)


# =============================================================================
# TOOL 1 — HR Documents (3 namespace in parallelo)
# =============================================================================

@tool
def search_hr_documents(query: str) -> str:
    """
    Cerca informazioni nei documenti HR aziendali.
    Usa questo tool per domande su: ferie, permessi, ROL, congedo parentale,
    orario di lavoro, smart working, rimborsi spese, benefit aziendali,
    polizza sanitaria, bonus MBO, contratti di lavoro, TFR, livelli di
    inquadramento, procedure HR, formazione, salute e sicurezza sul lavoro.
    """
    HR_NAMESPACES = ["hr_policy", "hr_faq", "hr_contracts"]

    # Misura il tempo parallelo vs sequenziale
    t_start = time.perf_counter()

    async def _parallel_retrieve():
        """Interroga i 3 namespace in parallelo con asyncio.gather."""
        tasks = [
            asyncio.to_thread(_retrieve, ns, query, settings.retriever_k)
            for ns in HR_NAMESPACES
        ]
        # gather lancia tutti i task simultaneamente e aspetta che finiscano tutti
        results = await asyncio.gather(*tasks)
        return results

    # Esegue la coroutine nel thread corrente (i tool sono chiamati in contesti sync)
    results_per_ns = asyncio.run(_parallel_retrieve())

    t_parallel = time.perf_counter() - t_start

    # Merge di tutti i risultati dai 3 namespace
    all_docs = []
    for ns_docs in results_per_ns:   
        for doc in ns_docs:          
            all_docs.append(doc)

    logger.info(f"[HR Tool] Recuperati {len(all_docs)} docs da {len(HR_NAMESPACES)} namespace "
            f"in {t_parallel:.2f}s (parallelo)")
    
    # Reranking sull'insieme combinato → top 5
    reranked = _rerank(query, all_docs, top_n=5)

    return _docs_to_string(reranked)


# =============================================================================
# TOOL 2 — ML Documents
# =============================================================================

@tool
def search_ml_documents(query: str) -> str:
    """
    Cerca informazioni nei documenti tecnici di Machine Learning.
    Usa questo tool per domande su: machine learning, algoritmi di classificazione
    e regressione, reti neurali, deep learning, backpropagation, overfitting,
    regularizzazione, metriche di valutazione, pipeline ML, LLM, RAG,
    embedding, transformer, tecniche AI avanzate, framework (PyTorch, scikit-learn,
    LangChain, LangGraph).
    """
    t_start = time.perf_counter()

    # Chiamata sincrona Pinecone wrappata per compatibilità
    docs = _retrieve("ml_docs", query, k=settings.retriever_k)

    t_retrieval = time.perf_counter() - t_start
    logger.info(f"[ML Tool] Recuperati {len(docs)} docs in {t_retrieval:.2f}s")

    reranked = _rerank(query, docs, top_n=5)

    return _docs_to_string(reranked)


# =============================================================================
# TOOL 3 — Report Generator
# =============================================================================

@tool
def generate_report(topic: str, include_hr: bool = True, include_ml: bool = True) -> str:
    """
    Genera un report strutturato in Markdown incrociando le knowledge base HR e ML.
    Usa questo tool per: report aziendali, riassunti strutturati, analisi comparative,
    documenti che richiedono informazioni da fonti diverse (policy HR + contenuti tecnici ML),
    sintesi di argomenti complessi in formato professionale con sezioni e conclusioni.
    """
    t_start = time.perf_counter()

    # ── 1. RETRIEVAL ─────────────────────────────────────────────────────────
    K_PER_NS = 2
    all_docs = []

    try:
        namespaces = []
        if include_hr:
            namespaces.extend(["hr_policy", "hr_faq", "hr_contracts"])
        if include_ml:
            namespaces.append("ml_docs")

        if not namespaces:
            return "Nessun namespace selezionato per il report."

        # Retrieval sequenziale ma con per-namespace try/except: se un namespace
        # fallisce gli altri proseguono comunque.
        for ns in namespaces:
            try:
                docs = _retrieve(ns, topic, k=K_PER_NS)
                all_docs.extend(docs)
            except Exception as e:
                logger.warning(f"[Report Tool] Retrieval namespace '{ns}' fallito: {e}")
                continue

        logger.info(
            f"[Report Tool] Recuperati {len(all_docs)} docs da {len(namespaces)} namespace "
            f"in {time.perf_counter() - t_start:.2f}s"
        )
    except Exception as e:
        logger.error(f"[Report Tool] Errore durante il retrieval: {e}")
        return f"Errore durante il recupero dei documenti per il report: {e}"

    if not all_docs:
        return "Nessun documento trovato per generare il report."

    # ── 2. RERANKING ─────────────────────────────────────────────────────────
    try:
        t_rerank = time.perf_counter()
        reranked = _rerank(topic, all_docs, top_n=5)
        context  = _docs_to_string(reranked)
        logger.info(
            f"[Report Tool] Rerank di {len(all_docs)} docs → top {len(reranked)} "
            f"in {time.perf_counter() - t_rerank:.2f}s"
        )
    except Exception as e:
        logger.warning(f"[Report Tool] Rerank fallito, uso docs grezzi: {e}")
        # Fallback: salta il rerank, usa i primi 5 documenti grezzi
        context = _docs_to_string(all_docs[:5])

    # ── 3. GENERAZIONE LLM ───────────────────────────────────────────────────
    # Wrappato in try/except: in caso di errore o timeout restituisce comunque
    # una stringa invece di rimanere appeso indefinitamente.
    prompt = f"""Sei un assistente specializzato nella generazione di report aziendali.

Genera un report professionale e strutturato in Markdown sul seguente argomento: {topic}

Usa ESCLUSIVAMENTE le informazioni nei documenti seguenti:

{context}

Il report deve seguire questa struttura:
# [Titolo del Report]

## Sommario Esecutivo
[2-3 righe che riassumono il contenuto]

## [Sezione 1 — primo tema principale]
[Contenuto dettagliato con bullet point dove appropriato]

## [Sezione 2 — secondo tema principale]
[Contenuto dettagliato]

## Conclusioni e Raccomandazioni
[Punti chiave actionable]

Sii completo, preciso e cita le fonti quando possibile."""

    try:
        t_llm = time.perf_counter()
        llm = get_llm(temperature=0)
        # request_timeout su ChatOpenAI: 25s evita blocchi infiniti del Report Agent
        response = llm.invoke(prompt, config={"timeout": 25})
        logger.info(
            f"[Report Tool] LLM completato in {time.perf_counter() - t_llm:.2f}s "
            f"(totale: {time.perf_counter() - t_start:.2f}s)"
        )
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        logger.error(f"[Report Tool] LLM invoke fallito: {e}")
        # Fallback: restituisce un report minimale costruito dal context grezzo,
        # così l'agente termina invece di rimanere appeso.
        return (
            f"# Report su: {topic}\n\n"
            f"## Sommario\n"
            f"Generazione automatica fallita ({e}). "
            f"Di seguito i documenti rilevanti recuperati dalle knowledge base.\n\n"
            f"## Documenti di riferimento\n\n{context}"
        )


# =============================================================================
# TOOL 4 — Calendar Event (HITL — non scrive direttamente)
# =============================================================================

@tool
def create_calendar_event(
    titolo: str,
    data: str,
    ora_inizio: str,
    ora_fine: str,
    descrizione: str = "",
) -> str:
    """
    Prepara la creazione di un evento sul Google Calendar aziendale.
    Usa questo tool quando l'utente vuole: aggiungere un evento, creare una riunione,
    fissare un appuntamento, schedulare una call, o qualsiasi azione che richieda
    di scrivere sul calendario. Formato data: YYYY-MM-DD. Formato ora: HH:MM.
    IMPORTANTE: questo tool NON scrive direttamente sul calendario —
    richiede approvazione esplicita dell'utente prima di procedere.
    """
    # NON chiama Google Calendar API.
    # Restituisce una stringa strutturata JSON che il nodo calendar_node
    # in supervisor.py intercetta per popolare pending_calendar_event nello State.
    # Il grafo si è già fermato su interrupt_before=["tools"] prima di arrivare qui.
    # Questo return viene letto dopo l'approvazione in POST /v1/approve.

    event_details = {
        "titolo"     : titolo,
        "data"       : data,
        "ora_inizio" : ora_inizio,
        "ora_fine"   : ora_fine,
        "descrizione": descrizione,
    }

    return json.dumps(event_details, ensure_ascii=False)


# =============================================================================
# ESPOSIZIONE DEI TOOL PER AGENTE
# =============================================================================

hr_tools       = [search_hr_documents]
ml_tools       = [search_ml_documents]
report_tools   = [generate_report]
calendar_tools = [create_calendar_event]