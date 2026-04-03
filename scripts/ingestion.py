# =============================================================================
# scripts/ingestion.py — Script standalone di indicizzazione
# =============================================================================
# Eseguire UNA VOLTA per caricare i documenti su Pinecone.
# Rieseguibile senza duplicati grazie agli ID MD5 idempotenti.
# =============================================================================

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Aggiunge la root del progetto al path per importare app/config.py
sys.path.append(str(Path(__file__).parent.parent))

from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langdetect import detect
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CONFIGURAZIONE
# =============================================================================

# Mappa namespace → lista di PDF da indicizzare
# Ogni namespace è una "sezione" logica dentro l'unico indice Pinecone
DOCUMENTS = {
    "hr_policy"   : ["docs/hr/policy.pdf"],
    "hr_faq"      : ["docs/hr/faq.pdf"],
    "hr_contracts": ["docs/hr/contracts.pdf"],
    "ml_docs"     : ["docs/ml/intro.pdf", "docs/ml/advanced.pdf"],
}

PINECONE_INDEX  = os.getenv("PINECONE_INDEX", "exerag")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY= os.getenv("PINECONE_API_KEY")
BATCH_SIZE      = 10   # quanti chunk elaborare per volta con l'LLM

# =============================================================================
# INIZIALIZZAZIONE CLIENT
# =============================================================================

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY,
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
    temperature=0,
)

# SemanticChunker taglia il testo dove il significato cambia in modo significativo.
# breakpoint_threshold_type="percentile": taglia solo nei punti dove il salto
# semantico è nel top 10% — chunk semanticamente omogenei, dimensione variabile.
splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=90,
)

pc = Pinecone(api_key=PINECONE_API_KEY)

# =============================================================================
# FUNZIONI
# =============================================================================

def make_id(namespace: str, source: str, content: str) -> str:
    """
    Genera un ID univoco e deterministico per ogni chunk.
    MD5 di namespace + source + contenuto → sempre lo stesso ID per lo stesso chunk.
    Idempotente: rieseguire lo script non crea duplicati su Pinecone.
    """
    raw = f"{namespace}::{source}::{content}"
    return hashlib.md5(raw.encode()).hexdigest()


def detect_language(text: str) -> str:
    """Rileva la lingua del testo con langdetect. Fallback 'unknown' se fallisce."""
    try:
        return detect(text)
    except Exception:
        return "unknown"


def generate_metadata_batch(chunks: list) -> list[dict]:
    """
    Invoca l'LLM una volta per batch di chunk (max BATCH_SIZE).
    Chiede di generare topic e document_type per ognuno.
    Restituisce una lista di dict con i metadata generati.
    """
    # Costruisce il prompt con tutti i chunk del batch numerati
    chunks_text = ""
    for i, chunk in enumerate(chunks):
        # Tronca a 500 caratteri per non sprecare token
        preview = chunk.page_content[:500].replace("\n", " ")
        chunks_text += f"\nChunk {i}:\n{preview}\n"

    prompt = f"""Analizza i seguenti chunk di testo e per ognuno restituisci un JSON.
        {chunks_text}
        Restituisci SOLO un array JSON valido, senza backtick, senza spiegazioni:
        [
        {{
            "topic": "argomento principale in 3-5 parole",
            "document_type": "uno tra: policy, faq, contract, tutorial, advanced"
        }},
        ...
        ]
        Un oggetto per ogni chunk, nello stesso ordine."""

    try:
        response = llm.invoke(prompt).content.strip()
        # Rimuove eventuali backtick markdown che l'LLM aggiunge
        response = response.replace("```json", "").replace("```", "").strip()
        return json.loads(response)
    except Exception as e:
        print(f"Errore generazione metadata LLM: {e} — uso defaults")
        # Fallback: metadata neutri per non bloccare l'ingestion
        return [{"topic": "non classificato", "document_type": "policy"}] * len(chunks)


def process_namespace(namespace: str, pdf_paths: list[str]) -> int:
    """
    Processa tutti i PDF di un namespace:
    1. Carica i PDF
    2. Chunking semantico
    3. Genera metadata in batch
    4. Indicizza su Pinecone con ID idempotenti

    Restituisce il numero di chunk indicizzati.
    """
    print(f"\n{'='*60}")
    print(f"  Namespace: {namespace}")
    print(f"{'='*60}")

    all_chunks = []

    # ── Step 1: Carica e splitta ogni PDF ────────────────────────────────────
    for pdf_path in pdf_paths:
        full_path = Path(__file__).parent.parent / pdf_path

        if not full_path.exists():
            print(f"File non trovato: {full_path} — skip")
            continue

        print(f"Carico: {pdf_path}")
        loader = PyPDFLoader(str(full_path))
        pages  = loader.load()  # ogni pagina è un Document con metadata source e page
        print(f"     {len(pages)} pagine caricate")

        # SemanticChunker taglia in base al significato, non alla lunghezza
        chunks = splitter.split_documents(pages)

        # Filtra chunk troppo corti (rumore, intestazioni, numeri di pagina)
        chunks = [c for c in chunks if len(c.page_content.strip()) > 100]
        print(f"     {len(chunks)} chunk semantici generati")

        all_chunks.extend(chunks)

    if not all_chunks:
        print(f"Nessun chunk — namespace saltato")
        return 0

    # ── Step 2: Genera metadata in batch ─────────────────────────────────────
    print(f"\n Generazione metadata LLM ({len(all_chunks)} chunk, batch da {BATCH_SIZE})...")

    llm_metadata = []
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i:i + BATCH_SIZE]
        batch_meta = generate_metadata_batch(batch)
        llm_metadata.extend(batch_meta)
        print(f"     Batch {i//BATCH_SIZE + 1}/{(len(all_chunks)-1)//BATCH_SIZE + 1} completato")

    # ── Step 3: Arricchisce i chunk con tutti i metadata ─────────────────────
    ids = []
    for idx, (chunk, meta) in enumerate(zip(all_chunks, llm_metadata)):
        source = chunk.metadata.get("source", "unknown")

        # Metadata statici — sempre presenti
        chunk.metadata.update({
            "namespace"   : namespace,
            "source"      : source,
            "chunk_index" : idx,
            "indexed_at"  : datetime.now(timezone.utc).isoformat(),
            "language"    : detect_language(chunk.page_content),
        })

        # Metadata dinamici generati dall'LLM
        chunk.metadata.update({
            "topic"        : meta.get("topic", "non classificato"),
            "document_type": meta.get("document_type", "policy"),
        })

        # ID deterministico per idempotenza
        chunk_id = make_id(namespace, source, chunk.page_content)
        ids.append(chunk_id)

    # ── Step 4: Indicizza su Pinecone ─────────────────────────────────────────
    print(f"\nInvio a Pinecone (namespace: {namespace})...")

    PineconeVectorStore.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX,
        namespace=namespace,
        ids=ids,      # ID deterministici — upsert, non duplica
    )

    print(f"{len(all_chunks)} chunk indicizzati")
    return len(all_chunks)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n🚀 Enterprise Agent Platform — Ingestion Script")
    print(f"   Index: {PINECONE_INDEX}")
    print(f"   Namespace da processare: {list(DOCUMENTS.keys())}")

    riepilogo = {}
    totale    = 0

    for namespace, pdf_paths in DOCUMENTS.items():
        n_chunks = process_namespace(namespace, pdf_paths)
        riepilogo[namespace] = n_chunks
        totale += n_chunks

    # ── Riepilogo finale ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  RIEPILOGO INDICIZZAZIONE")
    print(f"{'='*60}")
    for namespace, n in riepilogo.items():
        stato = "OK" if n > 0 else "ERROR"
        print(f"  {stato}  {namespace:<20} {n:>4} chunk")

    print(f"  {'─'*40}")
    print(f"Totale chunk indicizzati: {totale}")

    # Conta i vettori totali nell'index via API Pinecone
    try:
        index = pc.Index(PINECONE_INDEX)
        stats = index.describe_index_stats()
        print(f"Vettori totali nell'index: {stats['total_vector_count']}")
    except Exception as e:
        print(f"Impossibile leggere stats index: {e}")

    print(f"\nIngestion completata\n")


if __name__ == "__main__":
    main()