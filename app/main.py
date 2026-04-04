# =============================================================================
# app/main.py — FastAPI application principale
# =============================================================================
# Espone tutti gli endpoint REST del sistema multi-agente.
#
# Struttura:
#   - Lifespan: inizializza DB, Supervisor, MCP Server
#   - /auth/*:  register, login, refresh (no JWT richiesto)
#   - /v1/*:    tutti gli endpoint protetti da JWT
#   - /health:  health check pubblico
#
# Pattern chiave:
#   - asyncio.to_thread per chiamate sincrone (Supervisor, Pinecone)
#   - Depends(get_current_user) su tutti gli endpoint /v1/
#   - pending_calendar_event nello state → risposta pending_approval
# =============================================================================

import asyncio
import threading
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings, get_llm
from app.database import async_engine, Base, get_db
from app.models import (
    User, Conversation, Message,
    UserCreate, UserResponse, TokenResponse,
    ChatRequest, ChatResponse,
    ApproveRequest,
)
from app.auth import (
    hash_password, verify_password,
    create_access_token, create_refresh_token, decode_token,
    get_current_user,
)
from app.memory import save_to_db, load_from_db
from app.mcp_server import start_mcp_server, write_event_direct
from app.supervisor import build_supervisor
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

settings = get_settings()


# =============================================================================
# LIFESPAN — avvio e shutdown dell'applicazione
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Eseguito una volta all'avvio e una volta allo shutdown.
    Tutto prima del yield = startup. Tutto dopo = shutdown.
    """
    print(f"\n{'='*55}")
    print(f"  {settings.app_name} v{settings.app_version}")
    print(f"{'='*55}")

    # 1. Crea le tabelle PostgreSQL se non esistono
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("  ✅ PostgreSQL — tabelle pronte")

    # 2. Inizializza il Supervisor (costruisce i 4 sotto-agenti)
    app.state.supervisor = build_supervisor()
    print("  ✅ Supervisor — agenti inizializzati")

    # 3. Avvia il MCP Server in un thread daemon separato
    mcp_thread = threading.Thread(target=start_mcp_server, daemon=True)
    mcp_thread.start()
    print("  ✅ MCP Server — avviato in thread separato")

    print(f"  📋 Prompt Repetition: {'ABILITATA' if settings.use_prompt_repetition else 'DISABILITATA'}")
    print(f"  🗄️  Database: {settings.postgres_url.split('@')[-1]}")  # stampa solo host/db
    print(f"{'='*55}\n")

    yield  # app in esecuzione

    # Shutdown: chiude il pool di connessioni PostgreSQL
    await async_engine.dispose()
    print("\n[Shutdown] Pool PostgreSQL chiuso")


# =============================================================================
# APP E ROUTER
# =============================================================================

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

# Router per gli endpoint versionati
router = APIRouter(prefix="/v1")

# Router separato per auth (no prefix /v1)
auth_router = APIRouter(prefix="/auth", tags=["Auth"])


# =============================================================================
# HELPER — genera titolo conversazione con LLM
# =============================================================================

async def _generate_title(query: str) -> str:
    """
    Genera un titolo breve per la conversazione dal primo messaggio.
    Usato quando conversation_id è None (nuova conversazione).
    """
    llm    = get_llm(temperature=0)
    prompt = (
        f"Genera un titolo breve (max 6 parole) per una conversazione "
        f"che inizia con questa domanda: '{query}'. "
        f"Rispondi SOLO con il titolo, nessuna spiegazione."
    )
    response = await asyncio.to_thread(llm.invoke, prompt)
    return response.content.strip().strip('"')


def _extract_new_messages(
    state_before : list,
    state_after  : list,
    agente_usato : str,
    tools_usati  : list[str],
) -> list[dict]:
    """
    Estrae i messaggi nuovi confrontando la history prima e dopo l'invocazione.
    Restituisce solo i messaggi aggiunti durante questa invocazione del Supervisor.
    """
    n_before   = len(state_before)
    new_msgs   = state_after[n_before:]
    result     = []

    for msg in new_msgs:
        if isinstance(msg, HumanMessage):
            result.append({
                "role"        : "human",
                "content"     : msg.content,
                "agente_usato": None,
                "tools_usati" : None,
            })
        elif isinstance(msg, AIMessage) and msg.content:
            result.append({
                "role"        : "ai",
                "content"     : msg.content,
                "agente_usato": agente_usato,
                "tools_usati" : tools_usati if tools_usati else None,
            })
    return result


# =============================================================================
# AUTH ENDPOINTS
# =============================================================================

@auth_router.post("/register", response_model=UserResponse, status_code=201)
async def register(
    user_data: UserCreate,
    db       : AsyncSession = Depends(get_db),
):
    """Registra un nuovo utente. Verifica che l'email non sia già in uso."""
    # Controlla duplicati
    result = await db.execute(select(User).where(User.email == user_data.email))
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email già registrata",
        )

    new_user = User(
        email           =user_data.email,
        hashed_password =hash_password(user_data.password),
        full_name       =user_data.full_name,
    )
    db.add(new_user)
    await db.flush()  # genera l'UUID prima del commit

    return UserResponse.model_validate(new_user)


@auth_router.post("/login", response_model=TokenResponse)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db       : AsyncSession              = Depends(get_db),
):
    """
    Login con email e password.
    Usa OAuth2PasswordRequestForm — compatibile con il tasto Authorize di Swagger.
    Il campo 'username' di OAuth2 corrisponde all'email nel nostro sistema.
    """
    result = await db.execute(select(User).where(User.email == form_data.username))
    user   = result.scalar_one_or_none()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenziali non valide",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account disattivato")

    return TokenResponse(
        access_token =create_access_token(str(user.id), user.email),
        refresh_token=create_refresh_token(str(user.id)),
    )


@auth_router.post("/refresh", response_model=TokenResponse)
async def refresh(
    refresh_token: str,
    db           : AsyncSession = Depends(get_db),
):
    """
    Rinnova l'access token usando il refresh token.
    Verifica che il token sia di tipo 'refresh' (non un access token riusato).
    """
    payload = decode_token(refresh_token)

    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Token non è un refresh token")

    user_id = payload["sub"]
    user    = await db.get(User, uuid.UUID(user_id))

    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="Utente non trovato o disattivato")

    return TokenResponse(
        access_token =create_access_token(str(user.id), user.email),
        refresh_token=create_refresh_token(str(user.id)),  # rinnova anche il refresh
    )


# =============================================================================
# POST /v1/chat — endpoint principale
# =============================================================================

@router.post("/chat", response_model=None)
async def chat(
    request     : ChatRequest,
    req         : Request,
    current_user: User          = Depends(get_current_user),
    db          : AsyncSession  = Depends(get_db),
):
    """
    Endpoint principale di chat. Flusso:
      1. Crea conversazione se nuova (conversation_id=None)
      2. Carica history + summary dal DB
      3. Invoca il Supervisor (in thread separato — è sincrono)
      4. Salva nuovi messaggi e summary nel DB
      5. Se pending_calendar_event → restituisce pending_approval
         Altrimenti → restituisce ChatResponse
    """
    supervisor = req.app.state.supervisor

    # ── Step 1: crea nuova conversazione se necessario ────────────────────────
    if request.conversation_id is None:
        title = await _generate_title(request.query)
        conv  = Conversation(
            user_id=current_user.id,
            title  =title,
            summary="",
        )
        db.add(conv)
        await db.flush()
        conversation_id = conv.id
    else:
        conversation_id = request.conversation_id
        # Verifica che la conversazione appartenga all'utente corrente
        conv = await db.get(Conversation, conversation_id)
        if not conv or conv.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Conversazione non trovata")

    # ── Step 2: carica history dal DB ─────────────────────────────────────────
    history = await load_from_db(db, conversation_id)

    # ── Step 3: costruisce lo state e invoca il Supervisor ────────────────────
    # Aggiunge il nuovo messaggio dell'utente alla history caricata
    messages_before = history["messages"].copy()
    history["messages"].append(HumanMessage(content=request.query))

    initial_state = {
        **history,
        "routing"                : None,
        "agente_usato"           : "",
        "tools_usati"            : [],
        "pending_calendar_event" : None,
        "user_id"                : str(current_user.id),
        "conversation_id"        : str(conversation_id),
    }

    # asyncio.to_thread: il Supervisor è sincrono — lo esegue in thread separato
    # senza bloccare l'event loop di FastAPI
    result = await asyncio.to_thread(
        supervisor.invoke,
        initial_state,
        {"configurable": {"thread_id": str(conversation_id)}},
    )

    # ── Step 4: salva nel DB ──────────────────────────────────────────────────
    agente_usato = result.get("agente_usato", "")
    tools_usati  = result.get("tools_usati", [])
    summary      = result.get("summary", history["summary"])

    new_messages = _extract_new_messages(
        messages_before,
        result.get("messages", []),
        agente_usato,
        tools_usati,
    )

    await save_to_db(db, conversation_id, new_messages, summary)

    # ── Step 5: risposta ──────────────────────────────────────────────────────
    pending = result.get("pending_calendar_event")
    if pending:
        # Il Calendar Agent è sospeso — chiede approvazione all'utente
        return {
            "status"         : "pending_approval",
            "event"          : pending,
            "conversation_id": str(conversation_id),
        }

    # Estrae la risposta finale dall'ultimo AIMessage
    risposta = ""
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            risposta = msg.content
            break

    return ChatResponse(
        risposta       =risposta,
        agente_usato   =agente_usato,
        conversation_id=conversation_id,
        tools_usati    =tools_usati,
    )


# =============================================================================
# POST /v1/chat/stream — streaming SSE
# =============================================================================

@router.post("/chat/stream")
async def chat_stream(
    request     : ChatRequest,
    req         : Request,
    current_user: User         = Depends(get_current_user),
    db          : AsyncSession = Depends(get_db),
):
    """
    Streaming della risposta token per token via SSE (Server-Sent Events).
    Usa astream_events del Supervisor — filtra solo on_chat_model_stream.

    Il client legge lo stream con:
      fetch('/v1/chat/stream', {method: 'POST', ...})
      reader = response.body.getReader()
    """
    supervisor = req.app.state.supervisor

    # Stessa logica di /chat per caricare/creare la conversazione
    if request.conversation_id is None:
        title = await _generate_title(request.query)
        conv  = Conversation(user_id=current_user.id, title=title, summary="")
        db.add(conv)
        await db.flush()
        conversation_id = conv.id
    else:
        conversation_id = request.conversation_id
        conv = await db.get(Conversation, conversation_id)
        if not conv or conv.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Conversazione non trovata")

    history = await load_from_db(db, conversation_id)
    history["messages"].append(HumanMessage(content=request.query))

    initial_state = {
        **history,
        "routing"                : None,
        "agente_usato"           : "",
        "tools_usati"            : [],
        "pending_calendar_event" : None,
        "user_id"                : str(current_user.id),
        "conversation_id"        : str(conversation_id),
    }

    async def event_generator():
        """
        Generatore asincrono che produce eventi SSE.
        Formato SSE: "data: <contenuto>\n\n"
        """
        async for event in supervisor.astream_events(
            initial_state,
            {"configurable": {"thread_id": str(conversation_id)}},
            version="v2",
        ):
            # Filtra solo gli eventi di streaming del modello chat
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    # Formato SSE standard
                    yield f"data: {chunk.content}\n\n"

        # Segnale di fine stream
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control"    : "no-cache",
            "X-Accel-Buffering": "no",   # disabilita buffering nginx
            "Connection"       : "keep-alive",
        },
    )


# =============================================================================
# POST /v1/approve — approvazione HITL calendario
# =============================================================================

@router.post("/approve")
async def approve_calendar(
    request     : ApproveRequest,
    req         : Request,
    current_user: User         = Depends(get_current_user),
    db          : AsyncSession = Depends(get_db),
):
    """
    Approva o rifiuta la creazione di un evento sul calendario.

    Se approved=True:
      1. update_state sul Calendar Agent (sblocca l'interrupt)
      2. invoke(None) — riprende il grafo dal punto di sospensione
      3. Scrive l'evento su Google Calendar via MCP
      4. Salva il messaggio di conferma nel DB

    Se approved=False:
      Salva un messaggio di rifiuto e restituisce conferma annullamento.
    """
    from app.supervisor import _calendar_agent

    supervisor = req.app.state.supervisor

    # Carica la conversazione per verificare ownership e leggere pending_event
    conv = await db.get(Conversation, request.conversation_id)
    if not conv or conv.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Conversazione non trovata")

    # Ricostruisce il thread_id del Calendar Agent per questa conversazione
    thread_id = f"{current_user.id}_{request.conversation_id}_calendar_agent"
    config    = {"configurable": {"thread_id": thread_id}}

    if request.approved:
        # ── Approvazione ──────────────────────────────────────────────────────
        # Riprende il Calendar Agent dal punto di sospensione
        # invoke(None) = riprendi senza aggiungere nuovi messaggi
        calendar_result = await asyncio.to_thread(
            _calendar_agent.invoke,
            None,
            config,
        )

        # Estrae i dettagli dell'evento dall'ultimo ToolMessage (create_calendar_event)
        event_details = _extract_event_from_result(calendar_result)

        # Scrive su Google Calendar via MCP
        try:
            calendar_result_mcp = await asyncio.to_thread(
                write_event_direct,
                event_details,
            )
            event_id   = calendar_result_mcp.get("event_id", "")
            event_link = calendar_result_mcp.get("link", "")
            risposta   = (
                f"✅ Evento creato con successo!\n"
                f"📅 {event_details.get('titolo')} — {event_details.get('data')} "
                f"{event_details.get('ora_inizio')}-{event_details.get('ora_fine')}\n"
                f"🔗 {event_link}"
            )
        except Exception as e:
            risposta = f"⚠️ Errore nella creazione dell'evento: {str(e)}"

        # Salva la risposta nel DB
        await save_to_db(
            db,
            request.conversation_id,
            [{"role": "ai", "content": risposta, "agente_usato": "calendar_agent", "tools_usati": ["create_calendar_event"]}],
            summary=conv.summary,
        )

        return {"status": "approved", "message": risposta}

    else:
        # ── Rifiuto ───────────────────────────────────────────────────────────
        risposta = "❌ Creazione evento annullata su richiesta dell'utente."

        await save_to_db(
            db,
            request.conversation_id,
            [{"role": "ai", "content": risposta, "agente_usato": "calendar_agent", "tools_usati": []}],
            summary=conv.summary,
        )

        return {"status": "rejected", "message": risposta}


def _extract_event_from_result(result: dict) -> dict:
    """Estrae i dettagli dell'evento dal ToolMessage di create_calendar_event."""
    import json as _json
    messages = result.get("messages", [])
    for msg in messages:
        if isinstance(msg, ToolMessage):
            try:
                return _json.loads(msg.content)
            except Exception:
                pass
    return {}


# =============================================================================
# GET /v1/conversations — lista conversazioni
# =============================================================================

@router.get("/conversations")
async def list_conversations(
    current_user: User         = Depends(get_current_user),
    db          : AsyncSession = Depends(get_db),
):
    """Lista delle conversazioni dell'utente corrente con conteggio messaggi."""
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == current_user.id)
        .order_by(Conversation.updated_at.desc())
    )
    conversations = result.scalars().all()

    output = []
    for conv in conversations:
        # Conta i messaggi di questa conversazione
        count_result = await db.execute(
            select(func.count(Message.id))
            .where(Message.conversation_id == conv.id)
        )
        n_messages = count_result.scalar() or 0

        output.append({
            "id"          : str(conv.id),
            "title"       : conv.title,
            "created_at"  : conv.created_at.isoformat(),
            "updated_at"  : conv.updated_at.isoformat(),
            "n_messages"  : n_messages,
        })

    return {"conversations": output}


# =============================================================================
# GET /v1/conversations/{id}/messages — messaggi di una conversazione
# =============================================================================

@router.get("/conversations/{conversation_id}/messages")
async def get_messages(
    conversation_id: uuid.UUID,
    current_user   : User         = Depends(get_current_user),
    db             : AsyncSession = Depends(get_db),
):
    """Restituisce tutti i messaggi di una conversazione. Verifica ownership."""
    conv = await db.get(Conversation, conversation_id)
    if not conv or conv.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Conversazione non trovata")

    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
    )
    messages = result.scalars().all()

    return {
        "conversation_id": str(conversation_id),
        "title"          : conv.title,
        "summary"        : conv.summary,
        "messages"       : [
            {
                "id"          : str(msg.id),
                "role"        : msg.role,
                "content"     : msg.content,
                "agente_usato": msg.agente_usato,
                "tools_usati" : msg.tools_usati,
                "created_at"  : msg.created_at.isoformat(),
            }
            for msg in messages
        ],
    }


# =============================================================================
# DELETE /v1/conversations/{id} — elimina conversazione
# =============================================================================

@router.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: uuid.UUID,
    current_user   : User         = Depends(get_current_user),
    db             : AsyncSession = Depends(get_db),
):
    """
    Elimina una conversazione e tutti i suoi messaggi.
    cascade="all, delete-orphan" nel modello ORM gestisce i Message automaticamente.
    """
    conv = await db.get(Conversation, conversation_id)
    if not conv or conv.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Conversazione non trovata")

    await db.delete(conv)
    # Nessun return — status 204 No Content


# =============================================================================
# GET /health — health check pubblico
# =============================================================================

@app.get("/health", include_in_schema=False)
async def health(req: Request, db: AsyncSession = Depends(get_db)):
    """
    Health check pubblico — non richiede JWT, non appare nella docs Swagger.
    Verificato da AWS App Runner ogni 30s per determinare se il servizio è sano.
    """
    # Verifica connessione DB con una query leggera
    db_ok = True
    try:
        await db.execute(select(func.now()))
    except Exception:
        db_ok = False

    supervisor_ready = hasattr(req.app.state, "supervisor")

    return {
        "status"              : "ok" if db_ok else "degraded",
        "db"                  : "connected" if db_ok else "error",
        "supervisor"          : "ready" if supervisor_ready else "not_initialized",
        "use_prompt_repetition": settings.use_prompt_repetition,
        "version"             : settings.app_version,
    }


# =============================================================================
# MONTAGGIO ROUTER
# =============================================================================

app.include_router(auth_router)
app.include_router(router)
# ⚠️ include_router è OBBLIGATORIO — senza questa riga tutti gli endpoint
#    del router restituiscono 404, anche se il codice è corretto