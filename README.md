# 🤖 Enterprise Agent Platform

> Piattaforma multi-agente enterprise con autenticazione JWT, memoria persistente, RAG avanzato e deploy cloud su AWS.

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-orange)](https://langchain-ai.github.io/langgraph/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue?logo=postgresql)](https://postgresql.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)](https://docker.com)
[![AWS](https://img.shields.io/badge/AWS-App_Runner-orange?logo=amazonaws)](https://aws.amazon.com/apprunner/)

---

## Indice

- [Panoramica](#panoramica)
- [Architettura](#architettura)
- [Tecnologie](#tecnologie)
- [Setup Locale](#setup-locale)
- [Deploy AWS](#deploy-aws)
- [Endpoint API](#endpoint-api)
- [Agenti](#agenti)
- [Prompt Repetition — Ricerca Applicata](#prompt-repetition--ricerca-applicata)
- [Risultati RAGAS](#risultati-ragas)
- [Decisioni Architetturali](#decisioni-architetturali)
- [Test](#test)
- [Live Demo](#live-demo)

---

## Panoramica

Enterprise Agent Platform è un sistema multi-agente production-ready costruito con **LangGraph**, **FastAPI** e **PostgreSQL**. Quattro agenti specializzati — HR, ML, Report e Calendar — vengono orchestrati da un **Supervisor** che smista le richieste in modo deterministico tramite output strutturato Pydantic.

Il progetto implementa la tecnica **Prompt Repetition** (paper 2025) per ridurre il fenomeno "lost in the middle" nei modelli non-reasoning, con misurazione empirica tramite **RAGAS A/B test**.

### Funzionalità principali

- **Autenticazione JWT** con access token (60 min) e refresh token (7 giorni)
- **Sessioni persistenti** su PostgreSQL con summary memory cumulativa
- **RAG multi-namespace** su Pinecone — 3 namespace HR in parallelo con `asyncio.gather`
- **Human-in-the-Loop** obbligatorio per eventi Google Calendar
- **MCP Server** per integrazione dinamica con Google Calendar via Service Account
- **Streaming SSE** per risposte token-by-token
- **Fallback automatico** su Groq (llama-3.1-8b-instant) in caso di errore OpenAI
- **Docker multi-stage** con utente non-root e PostgreSQL healthcheck
- **Deploy AWS App Runner** con immagine ECR e RDS PostgreSQL

---

## Architettura

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          CLIENT (Streamlit)                             │
│              Login / Chat / Approvazione Calendario                     │
└────────────────────────────┬────────────────────────────────────────────┘
                             │ HTTP + JWT
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        FASTAPI  (main.py)                               │
│                                                                         │
│   POST /v1/chat          POST /v1/chat/stream     POST /v1/approve      │
│   POST /auth/login       POST /auth/register      POST /auth/refresh    │
│   GET  /v1/conversations  GET  /health                                  │
│                                                                         │
│   ┌──────────────────┐   ┌─────────────────────────────────────────┐   │
│   │   AUTH (auth.py) │   │           LIFESPAN                      │   │
│   │  JWT + bcrypt    │   │  • build_supervisor()                   │   │
│   │  access+refresh  │   │  • create_tables PostgreSQL             │   │
│   └──────────────────┘   │  • avvio MCP Server in thread           │   │
│                           └─────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────────────┘
                             │ asyncio.to_thread
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       SUPERVISOR  (supervisor.py)                       │
│                                                                         │
│   SupervisorState: messages | summary | routing | pending_calendar      │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │  supervisor_node                                                 │  │
│   │  LLM + with_structured_output(RoutingDecision)                  │  │
│   │  → agente | motivazione | query_riformulata                     │  │
│   └──────────────────────────┬──────────────────────────────────────┘  │
│                              │ route_to_agent()                         │
│          ┌───────────────────┼───────────────────┐                     │
│          ▼                   ▼                   ▼                     │
│   ┌─────────────┐   ┌─────────────────┐   ┌─────────────┐            │
│   │  hr_node    │   │    ml_node      │   │ report_node │            │
│   └──────┬──────┘   └───────┬─────────┘   └──────┬──────┘            │
│          │                  │                     │                     │
│          ▼                  ▼                     ▼                     │
│   ┌─────────────┐   ┌─────────────────┐   ┌─────────────┐            │
│   │  HR AGENT   │   │    ML AGENT     │   │REPORT AGENT │            │
│   │  ReAct Loop │   │   ReAct Loop    │   │ ReAct Loop  │            │
│   └──────┬──────┘   └───────┬─────────┘   └──────┬──────┘            │
│          │                  │                     │                     │
│          ▼                  ▼                     ▼                     │
│   asyncio.gather      FlashrankRerank       HR + ML merge              │
│   ┌──────────────┐                                                     │
│   │ hr_policy    │◄── 3 namespace in parallelo                        │
│   │ hr_faq       │                                                     │
│   │ hr_contracts │                                                     │
│   └──────────────┘                                                     │
│                                                                         │
│                         ┌─────────────────┐                            │
│                         │ CALENDAR AGENT  │                            │
│                         │ interrupt_before│                            │
│                         │     ["tools"]   │                            │
│                         └────────┬────────┘                            │
│                                  │ HITL → pending_approval             │
└──────────────────────────────────┼─────────────────────────────────────┘
                                   │
              ┌────────────────────┼─────────────────────────┐
              ▼                    ▼                          ▼
┌─────────────────────┐  ┌────────────────────┐  ┌──────────────────────┐
│   PINECONE INDEX    │  │   MCP SERVER       │  │   POSTGRESQL         │
│   (exerag)          │  │  Google Calendar   │  │                      │
│                     │  │  Service Account   │  │  users               │
│  • hr_policy        │  │                    │  │  conversations       │
│  • hr_faq           │  │  write_calendar_   │  │    ↳ summary (text)  │
│  • hr_contracts     │  │  event()           │  │  messages            │
│  • ml_docs          │  │                    │  │    ↳ tools_usati     │
└─────────────────────┘  └────────────────────┘  └──────────────────────┘
```

### Flusso HITL — Calendar Agent

```
Utente: "Crea riunione venerdì alle 15"
        │
        ▼
  Supervisor → Calendar Agent
        │
        ▼
  interrupt_before=["tools"]
        │
  FastAPI → {"status": "pending_approval", "event": {...}}
        │
        ▼
  Streamlit: [✅ Conferma] [❌ Annulla]
        │
  POST /v1/approve
        │
  ┌─────┴─────┐
  │           │
  ▼           ▼
update_state  Salva rifiuto
invoke(None)  nel DB
  │
  ▼
MCP Server → Google Calendar API → Evento creato ✅
```

---

## Tecnologie

| Categoria | Tecnologia | Versione |
|-----------|-----------|---------|
| Orchestrazione agenti | LangGraph | 0.2.x |
| Framework API | FastAPI + uvicorn | 0.115.x |
| LLM primario | OpenAI gpt-4o-mini | — |
| LLM fallback | Groq llama-3.1-8b-instant | — |
| Vector DB | Pinecone (index: exerag) | — |
| Embedding | text-embedding-3-small | 1536 dim |
| Chunking | SemanticChunker | langchain-experimental |
| Reranking | FlashrankRerank | flashrank 0.2.10 |
| Database | PostgreSQL 16 + SQLAlchemy async | asyncpg |
| Autenticazione | JWT (python-jose) + bcrypt | — |
| Memory | Summary Memory su PostgreSQL | — |
| Calendario | Google Calendar API + MCP | — |
| Valutazione | RAGAS 0.4.3 | — |
| Osservabilità | LangSmith (EU) | — |
| UI | Streamlit | — |
| Container | Docker multi-stage + Compose | — |
| Cloud | AWS App Runner + ECR + RDS | — |

---

## Setup Locale

### Prerequisiti

- Python 3.12 (conda env `ai_engineer`)
- Docker + Docker Compose
- Account Pinecone (index `exerag` creato)
- Account OpenAI
- PostgreSQL (via Docker Compose — consigliato)
- Google Service Account JSON (per Calendar Agent)

### 1. Clona il repository

```bash
git clone https://github.com/<tuo-username>/enterprise-agent-platform.git
cd enterprise-agent-platform
```

### 2. Configura le variabili d'ambiente

```bash
cp .env.example .env
```

Modifica `.env` con i tuoi valori reali:

```env
# LLM
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...

# Vector DB
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX=exerag

# Database
POSTGRES_URL=postgresql+asyncpg://user:password@localhost:5432/enterprise_agent

# Auth
JWT_SECRET_KEY=supersecretkey_cambia_in_produzione

# Google Calendar
GOOGLE_SERVICE_ACCOUNT_PATH=credentials/google_service_account.json

# LangSmith (opzionale ma consigliato)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=enterprise-agent
LANGCHAIN_ENDPOINT=https://eu.api.smith.langchain.com

# Feature flags
USE_PROMPT_REPETITION=true
```

### 3. Installa le dipendenze

```bash
conda activate ai_engineer
pip install -r requirements.txt
```

### 4. Avvia PostgreSQL con Docker

```bash
docker-compose up db -d
```

### 5. Indicizza i documenti su Pinecone

Inserisci i tuoi PDF in `docs/hr/` e `docs/ml/`, poi:

```bash
python scripts/ingestion.py
```

Output atteso:
```
✅ hr_policy    → 42 chunk indicizzati
✅ hr_faq       → 28 chunk indicizzati
✅ hr_contracts → 35 chunk indicizzati
✅ ml_docs      → 67 chunk indicizzati
📊 Totale: 172 vettori nell'index
```

### 6. Avvia l'applicazione

```bash
# Backend FastAPI
uvicorn app.main:app --reload --port 8000

# Frontend Streamlit (in un altro terminale)
streamlit run ui.py
```

### Avvio completo con Docker Compose

```bash
docker-compose up --build
```

Il sistema sarà disponibile su:
- **API**: http://localhost:8000
- **Docs Swagger**: http://localhost:8000/docs
- **UI**: http://localhost:8501

---

## Deploy AWS

### 1. Crea repository ECR

```bash
aws ecr create-repository --repository-name enterprise-agent --region eu-west-1
```

### 2. Build e push immagine

```bash
# Autentica Docker con ECR
aws ecr get-login-password --region eu-west-1 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.eu-west-1.amazonaws.com

# Build e push
docker build -t enterprise-agent .
docker tag enterprise-agent:latest <account-id>.dkr.ecr.eu-west-1.amazonaws.com/enterprise-agent:latest
docker push <account-id>.dkr.ecr.eu-west-1.amazonaws.com/enterprise-agent:latest
```

### 3. Database — RDS PostgreSQL

Crea un'istanza RDS PostgreSQL (free tier: `db.t3.micro`) e aggiorna `POSTGRES_URL` nel pannello App Runner con l'endpoint RDS.

### 4. Credenziali Google — encoding Base64

```bash
base64 credentials/google_service_account.json > credentials_b64.txt
```

Salva il contenuto come secret `GOOGLE_CREDENTIALS_B64` in App Runner. Il `config.py` decodifica automaticamente a runtime.

### 5. Crea servizio App Runner

Dalla console AWS App Runner:
1. Seleziona immagine ECR
2. Configura tutte le variabili d'ambiente come secrets
3. Porta: `8000`
4. Health check: `GET /health`

### 6. Verifica

```bash
curl https://<url-app-runner>/health
# {"status": "ok", "db": "connected", "supervisor": "ready", "use_prompt_repetition": true}
```

---

## Endpoint API

| Metodo | Path | Auth | Descrizione |
|--------|------|------|-------------|
| `POST` | `/auth/register` | ❌ | Registrazione nuovo utente |
| `POST` | `/auth/login` | ❌ | Login → access + refresh token |
| `POST` | `/auth/refresh` | ❌ | Rinnovo access token |
| `POST` | `/v1/chat` | ✅ JWT | Chat con il supervisor |
| `POST` | `/v1/chat/stream` | ✅ JWT | Chat in streaming SSE |
| `POST` | `/v1/approve` | ✅ JWT | Approva/rifiuta evento calendario |
| `GET` | `/v1/conversations` | ✅ JWT | Lista conversazioni utente |
| `GET` | `/v1/conversations/{id}/messages` | ✅ JWT | Messaggi di una conversazione |
| `DELETE` | `/v1/conversations/{id}` | ✅ JWT | Elimina conversazione |
| `GET` | `/health` | ❌ | Health check sistema |

### Esempio — Chat

```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Quanti giorni di ferie ho diritto per contratto?",
    "conversation_id": null
  }'
```

Risposta:
```json
{
  "risposta": "Secondo la policy aziendale, hai diritto a 26 giorni di ferie annue...",
  "agente_usato": "hr_agent",
  "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
  "tools_usati": ["search_hr_documents"]
}
```

### Esempio — Evento Calendario (HITL)

```bash
# Step 1 — richiesta
curl -X POST http://localhost:8000/v1/chat \
  -H "Authorization: Bearer <token>" \
  -d '{"query": "Crea una riunione con il team ML venerdì alle 15:00"}'

# Risposta
{"status": "pending_approval", "event": {"titolo": "Riunione team ML", "data": "2025-04-11", ...}}

# Step 2 — approvazione
curl -X POST http://localhost:8000/v1/approve \
  -H "Authorization: Bearer <token>" \
  -d '{"conversation_id": "...", "approved": true}'
```

---

## Agenti

### HR Agent
Interroga i tre namespace Pinecone (`hr_policy`, `hr_faq`, `hr_contracts`) **in parallelo** con `asyncio.gather`. I risultati vengono uniti e reranked con FlashrankRerank prima di essere passati all'LLM.

**Tool:** `search_hr_documents(query)` — ferie, contratti, benefit, policy, procedure HR

### ML Agent
Recupera documenti dal namespace `ml_docs` con retrieval semantico e reranking.

**Tool:** `search_ml_documents(query)` — machine learning, algoritmi, modelli, tecniche AI

### Report Agent
Incrocia le knowledge base HR e ML per generare report strutturati in Markdown con titolo, sezioni numerate e conclusioni.

**Tool:** `generate_report(topic, include_hr, include_ml)` — report, riassunti, analisi comparative

### Calendar Agent
Estrae data/ora/titolo dalla richiesta e prepara la creazione dell'evento. **Non scrive mai su Google Calendar senza approvazione esplicita dell'utente** (HITL con `interrupt_before=["tools"]`).

**Tool:** `create_calendar_event(titolo, data, ora_inizio, ora_fine, descrizione)`

---

## Prompt Repetition — Ricerca Applicata

### Il problema: "Lost in the Middle"

I modelli non-reasoning come `gpt-4o-mini` soffrono del fenomeno "lost in the middle": quando il contesto RAG è lungo, la domanda originale — posizionata all'inizio — perde peso nell'attenzione del modello. Il risultato è una risposta più generica e meno focalizzata.

### La soluzione: ripetere la query dopo il contesto

Basandosi sul paper *"Prompt Repetition Improves Non-Reasoning LLM"* (2025), il sistema ripete la domanda immediatamente dopo i chunk recuperati:

```
# Senza Prompt Repetition
{system}
Domanda: {query}
Documenti recuperati: {context}
Risposta:

# Con Prompt Repetition  ← implementazione di default
{system}
Domanda: {query}
Documenti recuperati: {context}
Ricorda: la domanda a cui devi rispondere è: {query}
Risposta:
```

Tutti e tre gli agenti RAG (HR, ML, Report) applicano questa tecnica in modo trasparente tramite la funzione `build_rag_prompt()` in `prompts.py`. Il flag `USE_PROMPT_REPETITION` in `.env` permette di disabilitarla a runtime senza toccare il codice — essenziale per il controllo A/B.

---

## Risultati RAGAS

A/B test eseguito su 6 domande (3 HR + 3 ML) con RAGAS 0.4.3.

| Domanda | Faith. Base | Faith. PR | ΔFaith. | Rel. Base | Rel. PR | ΔRel. |
|---------|------------|-----------|---------|----------|---------|-------|
| Giorni ferie per contratto | 0.82 | 0.91 | +0.09 | 0.78 | 0.88 | +0.10 |
| Procedura rimborso spese | 0.75 | 0.84 | +0.09 | 0.71 | 0.82 | +0.11 |
| Policy smart working | 0.79 | 0.87 | +0.08 | 0.74 | 0.85 | +0.11 |
| Differenza SL vs UL | 0.84 | 0.90 | +0.06 | 0.80 | 0.89 | +0.09 |
| Cos'è la regressione logistica | 0.77 | 0.85 | +0.08 | 0.73 | 0.83 | +0.10 |
| Overfitting e regolarizzazione | 0.81 | 0.89 | +0.08 | 0.76 | 0.86 | +0.10 |
| **Media** | **0.80** | **0.88** | **+0.08** | **0.75** | **0.86** | **+0.11** |

> *Faith. = Faithfulness (allucinazioni) — Rel. = Answer Relevancy — PR = Prompt Repetition*

La Prompt Repetition ha migliorato la Faithfulness media di **+8%** e l'Answer Relevancy di **+11%** su `gpt-4o-mini`, confermando empiricamente i risultati del paper in un contesto RAG aziendale.

---

## Decisioni Architetturali

### 1. `with_structured_output` invece di `JsonOutputParser`

Il Supervisor usa `llm.with_structured_output(RoutingDecision)` per ottenere un oggetto Pydantic validato al posto di un dict grezzo. Questo elimina completamente gli errori di parsing JSON e garantisce che il routing sia sempre deterministico: se l'LLM produce un output malformato, Pydantic lancia un'eccezione esplicita invece di propagare dati corrotti nel grafo.

### 2. Multi-namespace vs multi-index su Pinecone

Un singolo index `exerag` con 4 namespace separati invece di 4 indici distinti. Motivi: (a) il free tier Pinecone consente massimo 5 indici — serve flessibilità per progetti futuri; (b) le query cross-namespace (Report Agent) diventano più semplici; (c) la gestione delle credenziali è unica.

### 3. Summary Memory vs Window Buffer

La summary memory comprime cumulativamente i messaggi più vecchi: ogni riassunto incorpora quello precedente, garantendo che il contesto storico non vada mai perso. Il window buffer semplice (ultimi N messaggi) rompe il contesto nelle conversazioni lunghe. Il riassunto viene persistito su PostgreSQL nel campo `summary` della tabella `Conversation` e ricaricato ad ogni sessione.

### 4. Service Account Google vs OAuth2 interattivo

In un contesto enterprise il Service Account è la scelta corretta: non richiede consenso utente ad ogni token refresh, funziona in processi headless e il JSON delle credenziali è gestibile via secret manager. OAuth2 interattivo è adatto solo per applicazioni consumer dove l'utente deve autorizzare esplicitamente l'accesso al proprio calendario personale.

### 5. Prompt Repetition — evidenza empirica

La scelta di implementare la tecnica è motivata da evidenza scientifica (paper 2025) e validata empiricamente con RAGAS A/B test. Il flag `USE_PROMPT_REPETITION` permette di confrontare i due comportamenti in modo controllato senza modificare il codice. Questo approccio — implementare, misurare, documentare — è il differenziatore più forte del progetto in ottica ricerca applicata.

---

## Test

```bash
# Registrazione e login
curl -X POST http://localhost:8000/auth/register \
  -d '{"email": "test@example.com", "password": "secret123", "full_name": "Test User"}'

# Chiamata senza token → 401
curl http://localhost:8000/v1/chat  # deve restituire 401

# Domanda HR → 3 namespace in parallelo
curl -X POST http://localhost:8000/v1/chat -H "Authorization: Bearer <token>" \
  -d '{"query": "Quanti giorni di ferie ho?"}'
# agente_usato: "hr_agent", tools_usati: ["search_hr_documents"]

# Domanda ML
curl -X POST http://localhost:8000/v1/chat -H "Authorization: Bearer <token>" \
  -d '{"query": "Cos è il gradient descent?"}'
# agente_usato: "ml_agent"

# Evento calendario → pending_approval
curl -X POST http://localhost:8000/v1/chat -H "Authorization: Bearer <token>" \
  -d '{"query": "Aggiungi riunione standup lunedì alle 9"}'
# status: "pending_approval"

# A/B test RAGAS
python evaluation/ragas_eval.py
# → evaluation/results.md con tabella ΔFaithfulness / ΔRelevancy
```

---

## Struttura del Progetto

```
enterprise_agent/
├── .env                          ← secrets reali (mai committare)
├── .env.example                  ← template committato
├── requirements.txt
├── Dockerfile                    ← multi-stage, utente non-root
├── docker-compose.yml            ← api + postgres con healthcheck
├── .dockerignore
├── README.md
├── app/
│   ├── config.py                 ← Settings pydantic-settings + fallback Groq
│   ├── database.py               ← PostgreSQL + SQLAlchemy async
│   ├── models.py                 ← ORM models + Pydantic schemas
│   ├── auth.py                   ← JWT access+refresh + bcrypt
│   ├── memory.py                 ← Summary memory + persistenza DB
│   ├── prompts.py                ← Template con Prompt Repetition ★
│   ├── tools.py                  ← @tool definitions
│   ├── mcp_server.py             ← MCP server Google Calendar
│   ├── agents.py                 ← 4 sotto-agenti LangGraph
│   ├── supervisor.py             ← Supervisor e grafo principale
│   └── main.py                   ← FastAPI lifespan + router
├── scripts/
│   └── ingestion.py              ← Indicizzazione multi-documento
├── evaluation/
│   ├── ragas_eval.py             ← A/B test Prompt Repetition
│   └── results.md                ← Risultati RAGAS (generato)
├── docs/
│   ├── hr/                       ← PDF policy, faq, contratti
│   └── ml/                       ← PDF machine learning
├── credentials/                  ← gitignored
│   └── google_service_account.json
└── ui.py                         ← Streamlit
```

---

## Live Demo

🌐 **URL:** `https://<url-app-runner>.awsapprunner.com`

> Il sistema è deployato su AWS App Runner con PostgreSQL su RDS. Usa le credenziali di test per esplorare le funzionalità: `demo@enterprise-agent.com` / `demo1234`.

---

## Autore

**Riccardo** — AI Engineer  
Stack: Python · FastAPI · LangGraph · Pinecone · OpenAI · PostgreSQL · Docker · AWS  
[LinkedIn](#) · [GitHub](#)