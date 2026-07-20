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
#   contrattuali_tools → [search_documenti_contrattuali]
#   tecnici_tools      → [search_documenti_tecnici_e_sistema]
#
# Entrambi interrogano più namespace Pinecone in parallelo con asyncio.gather.
# =============================================================================

import asyncio
import time

from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from flashrank import Ranker, RerankRequest

from app.config import get_settings, get_llm, ensure_flashrank_model
import logging
logger = logging.getLogger(__name__)

settings = get_settings()


embeddings = OpenAIEmbeddings(
    model=settings.embedding_model,
    api_key=settings.openai_api_key,
)

# Ranker FlashrankRerank — modello leggero per reranking locale (no API esterne).
# cache_dir è esplicito: il default della libreria è "/tmp" (→ C:\tmp su Windows).
# ensure_flashrank_model() solleva RuntimeError se il modello non è già su disco,
# così l'assenza è un errore leggibile e non un download silenzioso all'import.
_flashrank_cache_dir = ensure_flashrank_model()
ranker = Ranker(
    model_name=settings.flashrank_model,
    cache_dir=str(_flashrank_cache_dir),
)
logger.info(
    f"[Flashrank] Modello '{settings.flashrank_model}' caricato da {_flashrank_cache_dir}"
)


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


def _search_namespaces(query: str, namespaces: list[str], label: str) -> str:
    """
    Interroga più namespace Pinecone IN PARALLELO con asyncio.gather,
    unisce i risultati, li rerankizza e li serializza per l'LLM.

    È il cuore condiviso dei due tool documentali: il pattern di retrieval
    parallelo è identico, cambiano solo i namespace interrogati.
    """
    t_start = time.perf_counter()

    async def _parallel_retrieve():
        """Lancia una similarity_search per namespace, tutte simultanee."""
        tasks = [
            asyncio.to_thread(_retrieve, ns, query, settings.retriever_k)
            for ns in namespaces
        ]
        # gather lancia tutti i task simultaneamente e aspetta che finiscano tutti
        return await asyncio.gather(*tasks)

    # Esegue la coroutine nel thread corrente (i tool sono chiamati in contesti sync)
    results_per_ns = asyncio.run(_parallel_retrieve())

    t_parallel = time.perf_counter() - t_start

    # Merge di tutti i risultati dai namespace interrogati
    all_docs = []
    for ns_docs in results_per_ns:
        all_docs.extend(ns_docs)

    logger.info(
        f"[{label}] Recuperati {len(all_docs)} docs da {len(namespaces)} namespace "
        f"in {t_parallel:.2f}s (parallelo)"
    )

    # Reranking sull'insieme combinato → top 5
    reranked = _rerank(query, all_docs, top_n=5)

    return _docs_to_string(reranked)


# =============================================================================
# TOOL 1 — Documenti contrattuali (capitolati, listini)
# =============================================================================

CONTRATTUALI_NAMESPACES = ["capitolati", "listini"]


@tool
def search_documenti_contrattuali(query: str) -> str:
    """
    Cerca nei documenti contrattuali e commerciali.
    Usa questo tool per domande su: capitolati di fornitura delle catene
    distributive, requisiti contrattuali richiesti dal cliente, obblighi di
    fornitura, penali, listini prezzi, codici articolo, pezzature e formati.
    """
    return _search_namespaces(query, CONTRATTUALI_NAMESPACES, "Contrattuali Tool")


# =============================================================================
# TOOL 2 — Documenti tecnici e di sistema
#          (schede tecniche, procedure HACCP, non conformità)
# =============================================================================

TECNICI_NAMESPACES = ["schede_tecniche", "procedure", "non_conformita"]


@tool
def search_documenti_tecnici_e_sistema(query: str) -> str:
    """
    Cerca nelle schede tecniche di prodotto e nella documentazione di sistema qualità.
    Usa questo tool per domande su: schede tecniche di prodotto, ingredienti,
    allergeni, valori nutrizionali, shelf life, conservazione, procedure HACCP,
    temperature di cella, controlli di processo, verbali di non conformità
    e relative azioni correttive.
    """
    return _search_namespaces(query, TECNICI_NAMESPACES, "Tecnici Tool")


# =============================================================================
# ESPOSIZIONE DEI TOOL PER AGENTE
# =============================================================================

contrattuali_tools = [search_documenti_contrattuali]
tecnici_tools      = [search_documenti_tecnici_e_sistema]
