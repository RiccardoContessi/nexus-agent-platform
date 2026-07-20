# =============================================================================
# scripts/ingestion.py — Script standalone di indicizzazione
# =============================================================================
# Eseguire UNA VOLTA per caricare i documenti su Pinecone.
# Rieseguibile senza duplicati grazie agli ID MD5 idempotenti.
# =============================================================================

import hashlib
import os
import re
import sys
from pathlib import Path

# La console Windows usa cp1252, che non sa codificare né le frecce né i
# trattini lunghi. Senza questa riga lo script MUORE a metà ingestion su un
# print — non su un errore vero — e i titoli in DISPLAY_TITLES contengono un
# trattino lungo, quindi il rischio non si elimina scrivendo i literal in ASCII:
# passa dai dati. errors="replace" degrada il carattere, non l'esecuzione.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Aggiunge la root del progetto al path per importare app/config.py
sys.path.append(str(Path(__file__).parent.parent))

from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# TITOLI DA MOSTRARE IN CITAZIONE  ← modificabili a mano
# =============================================================================
# Mappa nome-file → titolo leggibile. Questo, e non il path del file, è ciò che
# finisce nel metadato `source` e quindi nella citazione mostrata al cliente.
# Il path completo non deve mai comparire: è illeggibile in una citazione e
# proietterebbe la struttura di cartelle della macchina sullo schermo.

DISPLAY_TITLES = {
    "Allegato L7_capitolato-tecnico-gara-carne-ed-affettati 2023_2026_DEF.pdf":
        "Capitolato tecnico — Fornitura carni fresche, mense comunali 2023/2026",
    "1ALLEGATO_1_Manuale_Macelli_carni_rosse_agg._15.04.2019.pdf":
        "Manuale controllo ufficiale — Sezionamento ungulati domestici",
    "1225-hamburgher-ba.pdf":
        "Scheda tecnica — Hamburger B.A.",
}

# =============================================================================
# REVISIONE PER DOCUMENTO  ← modificabile a mano
# =============================================================================
# Mappa nome-file → stringa di revisione da mostrare in citazione.
# È una costante per documento, non un dato per pagina: il manuale dichiara
# "ED. 1 / Rev 1 / LUGLIO 2019" nell'intestazione di ogni pagina, sempre uguale.
# Una tabella scritta a mano è più solida di una regex sull'intestazione, e
# questi valori cambiano una volta ogni edizione — non ogni ingestion.
# Un file assente da questa mappa non è un errore: `revisione` resta None.

DOCUMENT_REVISIONS = {
    "1ALLEGATO_1_Manuale_Macelli_carni_rosse_agg._15.04.2019.pdf": "Rev 1 — 2019",
}

# =============================================================================
# INTESTAZIONI RIPETUTE DA RIMUOVERE PRIMA DEL CHUNKING
# =============================================================================
# Due documenti ripetono un blocco di intestazione su OGNI pagina. Finisce in
# ogni chunk e costa due volte:
#   1. diluisce l'embedding e l'input del reranker su tutti i chunk
#   2. il numero di pagina STAMPATO nel corpo è un numero che il modello può
#      citare al posto di quello nei metadata — esattamente l'errore che la
#      regola di citazione dello step 4 esiste per impedire
#
# Entrambe le regole sono verificate deterministiche sull'intero documento
# (27/27 pagine per il manuale, 12/12 per il capitolato) e falliscono APERTE:
# una pagina che non corrisponde esattamente viene lasciata intatta, mai
# tagliata a indovinare. `strip_stats` conta le pagine non toccate così che
# un mancato match sia visibile e non silenzioso.

# Blocco costante in testa a ogni pagina del manuale di sezionamento.
MANUALE_HEADER_LINES = [
    "MANUALE PER IL CONTROLLO UFFICIALE PRESSO OPERATORI DEL SETTORE",
    "ALIMENTARE PRESSO OPERATORI DEL SETTORE ALIMENTARE",
    "MACELLO/SEZIONAMENTO DI UNGULATI DOMESTICI",
    "ED. 1",
    "Rev 1",
    "LUGLIO 2019",
]

# Riga 0 di ogni pagina del capitolato: "Pagina n. 5 di pagine n.12".
# Le altre righe di intestazione del capitolato NON sono costanti (spariscono
# dopo pagina 4), quindi si toglie solo questa — il resto sarebbe una regex
# fragile su testo variabile.
CAPITOLATO_PAGENUM_RE = re.compile(r"Pagina n\. (\d+) di pagine n\.(\d+)")


def _strip_manuale_header(text: str, page_label) -> tuple[str, bool]:
    """
    Rimuove dal manuale di sezionamento il blocco di intestazione ripetuto e
    il numero di pagina stampato che lo segue.

    Restituisce (testo, applicata). Se le prime 6 righe non corrispondono
    ESATTAMENTE alla costante, il testo torna immutato e applicata=False.
    """
    lines = text.split("\n")
    if [l.strip() for l in lines[:6]] != MANUALE_HEADER_LINES:
        return text, False

    rest = lines[6:]

    # Salta le righe vuote fra l'intestazione e il numero di pagina stampato
    i = 0
    while i < len(rest) and not rest[i].strip():
        i += 1

    # Il numero stampato sul foglio coincide con page_label su tutte le pagine.
    # Si toglie solo se coincide: un numero diverso sarebbe contenuto, non
    # intestazione.
    if i < len(rest) and rest[i].strip() == str(page_label):
        i += 1

    return "\n".join(rest[i:]), True


def _strip_capitolato_pagenum(text: str, page_label) -> tuple[str, bool]:
    """
    Rimuove dal capitolato la riga 0 "Pagina n. N di pagine n.M".

    Si toglie solo se N coincide con page_label: è la prova che la riga è
    l'intestazione di pagina e non testo che parla di un'altra pagina.
    """
    lines = text.split("\n")
    if not lines:
        return text, False

    m = CAPITOLATO_PAGENUM_RE.fullmatch(lines[0].strip())
    if not m or m.group(1) != str(page_label):
        return text, False

    return "\n".join(lines[1:]), True


# Mappa nome-file → funzione di pulizia. Un file non presente non viene toccato.
HEADER_STRIPPERS = {
    "1ALLEGATO_1_Manuale_Macelli_carni_rosse_agg._15.04.2019.pdf":
        _strip_manuale_header,
    "Allegato L7_capitolato-tecnico-gara-carne-ed-affettati 2023_2026_DEF.pdf":
        _strip_capitolato_pagenum,
}


# =============================================================================
# CHUNKING A CAMPO PER LE SCHEDE TECNICHE
# =============================================================================
# Perché esiste, misurato e non supposto.
#
# La scheda tecnica è UNA pagina che il SemanticChunker produce come un blob
# unico di ~1500 caratteri: è una tabella etichettata, non prosa, quindi non
# offre il salto semantico su cui quel chunker taglia. Il risultato è che il
# fatto puntuale sparisce nella media del vettore. Misura diretta, cosine
# contro text-embedding-3-small:
#
#     domanda                     blob intero      chunk di campo
#     "che pezzatura ha ..."         0.4205            0.5949
#     "che allergeni dichiara ..."   0.5598            0.7753
#
# Il massimo delle domande FUORI corpus è 0.4827: col blob la pezzatura
# (0.4205) sta SOTTO una domanda a cui il sistema deve rifiutare di rispondere,
# e nessuna soglia può separarle. A campo, entrambe lo superano.
#
# Il taglio è per etichetta, da una lista scritta a mano e non da una regex
# sulle maiuscole: le etichette di questo documento sono un insieme chiuso e
# noto, e una lista esplicita fallisce RUMOROSAMENTE se il documento cambia,
# mentre una regex si adatterebbe in silenzio producendo campi sbagliati.
#
# Riguarda SOLO il namespace schede_tecniche. `procedure` e `capitolati`
# restano sul SemanticChunker verificato allo step 0 — inclusa la garanzia
# che i chunk non attraversino i confini di pagina.

SCHEDA_LABELS = [
    "PRODOTTO",
    "DEFINIZIONE",
    "USO",
    "CONFEZIONAMENTO",
    "PEZZATURA",
    "CARATTERISTICHE ORGANOLETTICHE",
    "COLORE",
    "ODORE",
    "SAPORE",
    "SHELF-LIFE",
    "CONSERVAZIONE",
    "INGREDIENTI",
    "ALLERGENI",
    "ORGANISMI GENETICAMENTE MODIFICATI",
    "CARATTERISTICHE MICROBIOLOGICHE, CHIMICHE E FISICHE",
    "VALORI NUTRIZIONALI MEDI x 100 G DI PRODOTTO",
]

# Le prime righe sono l'intestazione aziendale: chi produce. È informazione
# legittima e citabile, ma non è un campo della scheda — diventa un chunk suo.
SCHEDA_INTESTAZIONE_FINE = "SCHEDA TECNICA DEL PRODOTTO"


def _split_scheda_tecnica(text: str, titolo: str) -> list[str]:
    """
    Taglia una scheda tecnica in un chunk per campo.

    Ogni chunk viene prefissato col titolo del documento e con l'etichetta del
    campo: un chunk che dice solo "Circa 200 g" è inutile sia da recuperare sia
    da leggere. Il prefisso è ciò che porta i cosine misurati sopra.

    Fallisce con ValueError se un'etichetta attesa non compare: un documento
    diverso da quello previsto va visto adesso, non in riunione.
    """
    # Whitespace normalizzato: le etichette su più righe nel PDF
    # ("ORGANISMI / GENETICAMENTE / MODIFICATI") devono corrispondere alla
    # forma unica dichiarata in SCHEDA_LABELS.
    flat = " ".join(text.split())

    chunks = []

    # ── Intestazione aziendale ────────────────────────────────────────────────
    taglio = flat.find(SCHEDA_INTESTAZIONE_FINE)
    if taglio > 0:
        chunks.append(f"{titolo} — Produttore: {flat[:taglio].strip()}")
        flat = flat[taglio + len(SCHEDA_INTESTAZIONE_FINE):]

    # ── Posizione di ogni etichetta, nell'ordine del documento ────────────────
    posizioni = []
    cursore   = 0
    for label in SCHEDA_LABELS:
        idx = flat.find(label, cursore)
        if idx == -1:
            raise ValueError(
                f"Campo '{label}' non trovato in '{titolo}'. La scheda tecnica "
                f"non ha la struttura attesa: aggiorna SCHEDA_LABELS in "
                f"scripts/ingestion.py invece di indicizzare campi sbagliati."
            )
        posizioni.append((idx, label))
        cursore = idx + len(label)

    # ── Un chunk per campo, dal proprio inizio all'inizio del successivo ──────
    for i, (idx, label) in enumerate(posizioni):
        fine   = posizioni[i + 1][0] if i + 1 < len(posizioni) else len(flat)
        valore = flat[idx + len(label):fine].strip(" :—-")

        # Un'etichetta di sola sezione (es. CARATTERISTICHE ORGANOLETTICHE, che
        # è seguita subito da COLORE) non ha valore proprio: si salta invece di
        # indicizzare un chunk vuoto.
        if not valore:
            continue

        chunks.append(f"{titolo} — {label}: {valore}")

    return chunks


# =============================================================================
# CONFIGURAZIONE
# =============================================================================

# Mappa namespace → lista di PDF da indicizzare
# Ogni namespace è una "sezione" logica dentro l'unico indice Pinecone
DOCUMENTS = {
    "capitolati": [
        "docs/capitolati/Allegato L7_capitolato-tecnico-gara-carne-ed-affettati 2023_2026_DEF.pdf",
    ],
    "procedure": [
        "docs/procedure/1ALLEGATO_1_Manuale_Macelli_carni_rosse_agg._15.04.2019.pdf",
    ],
    "schede_tecniche": [
        "docs/schede_tecniche/1225-hamburgher-ba.pdf",
    ],
}

PINECONE_INDEX  = os.getenv("PINECONE_INDEX", "ambrosini-demo")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY= os.getenv("PINECONE_API_KEY")

# =============================================================================
# INIZIALIZZAZIONE CLIENT
# =============================================================================

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY,
)

# SemanticChunker taglia il testo dove il significato cambia in modo significativo.
# breakpoint_threshold_type="percentile": taglia solo nei punti dove il salto
# semantico è nel top 10% — chunk semanticamente omogenei, dimensione variabile.
#
# NON modificare questi parametri: lo step 0 ha verificato che con questa
# configurazione i chunk restano dentro i confini di pagina (32/32), che è la
# condizione perché `page` sia un dato corretto e non approssimato.
splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=90,
)

pc = Pinecone(api_key=PINECONE_API_KEY)

# =============================================================================
# FUNZIONI
# =============================================================================

def make_id(namespace: str, filename: str, page, content: str) -> str:
    """
    Genera un ID univoco e deterministico per ogni chunk.
    Idempotente: rieseguire lo script non crea duplicati su Pinecone.

    Si basa sul NOME DEL FILE, non sul titolo visualizzato: così correggere a
    mano un titolo in DISPLAY_TITLES riscrive gli stessi vettori invece di
    crearne una seconda copia con ID diversi.
    La pagina entra nell'ID per evitare collisioni fra chunk di testo identico
    su pagine diverse (intestazioni e piè di pagina ripetuti).
    """
    raw = f"{namespace}::{filename}::{page}::{content}"
    return hashlib.md5(raw.encode()).hexdigest()


def display_title(filename: str) -> str:
    """
    Titolo leggibile per la citazione. Se un file non è mappato è un errore
    esplicito e non un fallback silenzioso: un titolo mancante si vedrebbe
    solo in riunione, dentro una citazione.
    """
    if filename not in DISPLAY_TITLES:
        raise KeyError(
            f"Nessun titolo in DISPLAY_TITLES per '{filename}'. "
            f"Aggiungilo in cima a questo script prima di indicizzare."
        )
    return DISPLAY_TITLES[filename]


def process_namespace(namespace: str, pdf_paths: list[str]) -> int:
    """
    Processa tutti i PDF di un namespace:
    1. Carica i PDF
    2. Chunking semantico
    3. Applica i metadata di citazione
    4. Indicizza su Pinecone con ID idempotenti

    Restituisce il numero di chunk indicizzati.
    """
    print(f"\n{'='*60}")
    print(f"  Namespace: {namespace}")
    print(f"{'='*60}")

    all_chunks = []
    ids        = []

    # ── Step 1: Carica, splitta e assegna i metadata, un PDF per volta ───────
    # I metadata si applicano dentro questo ciclo: fuori non si saprebbe più da
    # quale file provenga ogni chunk.
    for pdf_path in pdf_paths:
        full_path = Path(__file__).parent.parent / pdf_path

        if not full_path.exists():
            print(f"File non trovato: {full_path} - skip")
            continue

        filename = full_path.name
        titolo   = display_title(filename)

        print(f"Carico: {pdf_path}")
        loader = PyPDFLoader(str(full_path))
        pages  = loader.load()  # ogni pagina è un Document con metadata source e page
        print(f"     {len(pages)} pagine caricate  ->  \"{titolo}\"")

        # ── Pulizia intestazioni ripetute, PRIMA del chunking ────────────────
        # Va fatta qui e non dopo: se il blocco entra nel chunk, entra anche
        # nell'embedding, e a quel punto toglierlo dal testo non ripulisce più
        # il vettore.
        stripper = HEADER_STRIPPERS.get(filename)
        if stripper:
            applicate = 0
            for page_doc in pages:
                nuovo, ok = stripper(
                    page_doc.page_content,
                    page_doc.metadata.get("page_label"),
                )
                page_doc.page_content = nuovo
                applicate += ok
            saltate = len(pages) - applicate
            print(f"     Intestazione rimossa da {applicate}/{len(pages)} pagine")
            if saltate:
                # Rumoroso di proposito: una regola che smette di corrispondere
                # è un cambio di documento, e va visto adesso e non in riunione.
                print(f"     ATTENZIONE: {saltate} pagine non corrispondono alla "
                      f"regola e sono state lasciate intatte")

        if namespace == "schede_tecniche":
            # ── Chunking a campo ─────────────────────────────────────────────
            # Il SemanticChunker non sa tagliare una tabella etichettata: vedi
            # il blocco di commento su _split_scheda_tecnica. Qui si taglia per
            # campo, e il filtro sulla lunghezza NON si applica — "PEZZATURA:
            # circa 200 g" sta in 40 caratteri ed è esattamente il chunk che
            # serve recuperare. Quel filtro esiste per il rumore di impaginazione
            # dei documenti lunghi, che qui non c'è.
            from langchain_core.documents import Document as _Doc

            chunks = []
            for page_doc in pages:
                for testo in _split_scheda_tecnica(page_doc.page_content, titolo):
                    chunks.append(_Doc(page_content=testo, metadata=dict(page_doc.metadata)))
            print(f"     {len(chunks)} chunk a campo generati")
        else:
            # SemanticChunker taglia in base al significato, non alla lunghezza
            chunks = splitter.split_documents(pages)

            # Filtra chunk troppo corti (rumore, intestazioni, numeri di pagina)
            chunks = [c for c in chunks if len(c.page_content.strip()) > 100]
            print(f"     {len(chunks)} chunk semantici generati")

        for chunk in chunks:
            page       = chunk.metadata.get("page")
            page_label = chunk.metadata.get("page_label")

            # I metadata si RICOSTRUISCONO, non si aggiornano: PyPDFLoader
            # popola anche source (path completo della macchina), producer,
            # creator, creationdate, moddate, total_pages. Nessuno di questi
            # serve alla citazione e `source` andrebbe attivamente nascosto.
            chunk.metadata = {
                "source"     : titolo,       # titolo leggibile, non il path
                "page"       : page,         # indice PDF, 0-based
                "page_label" : page_label,   # numero stampato sul foglio
                "namespace"  : namespace,
            }

            # `revisione` è una costante del documento: si allega a ogni chunk
            # solo se il documento ne dichiara una. Pinecone non accetta None
            # come valore di metadato, quindi la chiave si omette del tutto
            # invece di scriverla vuota.
            revisione = DOCUMENT_REVISIONS.get(filename)
            if revisione:
                chunk.metadata["revisione"] = revisione

            all_chunks.append(chunk)
            ids.append(make_id(namespace, filename, page, chunk.page_content))

    if not all_chunks:
        print(f"Nessun chunk - namespace saltato")
        return 0

    # ── Step 2: Svuota il namespace prima di reindicizzare ───────────────────
    # Gli ID sono MD5 del contenuto: se il testo cambia — ed è esattamente ciò
    # che fa la rimozione delle intestazioni — cambia anche l'ID. L'upsert
    # scriverebbe i chunk nuovi ACCANTO ai vecchi invece che sopra, lasciando
    # in indice le versioni con l'intestazione ancora attaccata. Sarebbero
    # recuperabili, e citabili, senza che nulla lo segnali.
    # L'idempotenza degli ID protegge dal rieseguire lo script a parità di
    # testo; non protegge da un cambio di testo. Qui serve svuotare.
    try:
        index = pc.Index(PINECONE_INDEX)
        stats = index.describe_index_stats()
        esistenti = stats["namespaces"].get(namespace, {}).get("vector_count", 0)
        if esistenti:
            print(f"\nSvuoto il namespace '{namespace}' ({esistenti} vettori esistenti)...")
            index.delete(delete_all=True, namespace=namespace)
        else:
            print(f"\nNamespace '{namespace}' già vuoto")
    except Exception as e:
        print(f"Impossibile svuotare il namespace '{namespace}': {e}")
        raise

    # ── Step 3: Indicizza su Pinecone ────────────────────────────────────────
    print(f"Invio a Pinecone (namespace: {namespace})...")

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
    print("\nEnterprise Agent Platform - Ingestion Script")
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

    print(f"  {'-'*40}")
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
