# =============================================================================
# app/mcp_server.py — MCP Server per Google Calendar
# =============================================================================
# MCP (Model Context Protocol) è un protocollo standard Anthropic che permette
# agli LLM di scoprire e usare tool esposti da server esterni in modo dinamico.
#
# Differenza con API diretta:
#   API diretta → tu scrivi il codice che chiama Google Calendar, l'LLM non sa
#                 che esiste fino a quando non glielo dici esplicitamente.
#   MCP         → il server espone i tool con schema, l'LLM li scopre da solo
#                 interrogando il server — aggiungere tool non richiede
#                 modifiche al codice dell'agente.
#
# Flusso in questo progetto:
#   1. Il lifespan FastAPI avvia il MCP Server in un thread separato
#   2. Il Calendar Agent, dopo l'approvazione HITL, chiama write_calendar_event
#      tramite il server MCP invece che direttamente
#   3. Il server MCP autentica con Google tramite Service Account e crea l'evento
#
# Installazione:
#   pip install mcp google-auth google-auth-httplib2 google-api-python-client
# =============================================================================

import json
import threading
from datetime import datetime
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

import google.auth
from google.oauth2 import service_account
from googleapiclient.discovery import build

from app.config import get_settings

settings = get_settings()

# =============================================================================
# AUTENTICAZIONE GOOGLE — Service Account
# =============================================================================
# Il Service Account è un account non-umano usato dalle applicazioni per
# autenticarsi con Google senza intervento dell'utente.
# Non richiede OAuth2 interattivo — funziona in modo headless (senza browser).
# Le credenziali sono nel file JSON referenziato in settings.

SCOPES = ["https://www.googleapis.com/auth/calendar"]


def get_google_service():
    """
    Crea e restituisce il client autenticato per Google Calendar API.

    Usa Service Account JSON per autenticazione headless — appropriato
    per applicazioni server-side che girano senza interazione utente.
    """
    credentials_path = Path(settings.google_service_account_path)

    if not credentials_path.exists():
        raise FileNotFoundError(
            f"Service Account JSON non trovato: {credentials_path}\n"
            "Assicurati di avere il file in credentials/google_service_account.json"
        )

    # Carica le credenziali dal file JSON
    credentials = service_account.Credentials.from_service_account_file(
        str(credentials_path),
        scopes=SCOPES,
    )

    # Costruisce il client Google Calendar API v3
    service = build("calendar", "v3", credentials=credentials)
    return service


# =============================================================================
# MCP SERVER — Definizione e tool
# =============================================================================

# Istanza del server MCP — il nome identifica il server per i client
mcp_server = Server("google-calendar")


@mcp_server.list_tools()
async def list_tools() -> list[types.Tool]:
    """
    Espone la lista dei tool disponibili sul server MCP.
    L'LLM chiama questo endpoint per scoprire dinamicamente
    quali tool può usare — senza che tu li hardcodi nell'agente.
    """
    return [
        types.Tool(
            name="write_calendar_event",
            description=(
                "Crea un evento su Google Calendar. "
                "Usa questo tool per scrivere riunioni, appuntamenti o promemoria "
                "sul calendario aziendale. Richiede data e orari nel formato specificato."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "calendar_id": {
                        "type": "string",
                        "description": "ID del calendario Google (es. 'primary' o email del calendario)",
                    },
                    "titolo": {
                        "type": "string",
                        "description": "Titolo dell'evento",
                    },
                    "data": {
                        "type": "string",
                        "description": "Data dell'evento in formato YYYY-MM-DD",
                    },
                    "ora_inizio": {
                        "type": "string",
                        "description": "Ora di inizio in formato HH:MM",
                    },
                    "ora_fine": {
                        "type": "string",
                        "description": "Ora di fine in formato HH:MM",
                    },
                    "descrizione": {
                        "type": "string",
                        "description": "Descrizione opzionale dell'evento",
                        "default": "",
                    },
                },
                "required": ["calendar_id", "titolo", "data", "ora_inizio", "ora_fine"],
            },
        )
    ]


@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """
    Esegue il tool richiesto dall'LLM.
    MCP instrada la chiamata qui dopo che l'LLM ha scelto il tool da list_tools().
    """
    if name != "write_calendar_event":
        return [types.TextContent(
            type="text",
            text=f"Tool '{name}' non riconosciuto."
        )]

    try:
        result = _write_calendar_event(**arguments)
        return [types.TextContent(type="text", text=result)]
    except Exception as e:
        return [types.TextContent(type="text", text=f"Errore creazione evento: {str(e)}")]


def _write_calendar_event(
    calendar_id: str,
    titolo: str,
    data: str,
    ora_inizio: str,
    ora_fine: str,
    descrizione: str = "",
) -> str:
    """
    Funzione interna che scrive effettivamente su Google Calendar.
    Separata dal tool MCP per poter essere chiamata anche direttamente
    da POST /v1/approve senza passare per il protocollo MCP.

    Args:
        calendar_id: 'primary' oppure l'email del calendario specifico
        titolo:      titolo dell'evento
        data:        YYYY-MM-DD
        ora_inizio:  HH:MM
        ora_fine:    HH:MM
        descrizione: testo opzionale

    Returns:
        JSON string con id evento e link di accesso
    """
    service = get_google_service()

    # Costruisce il datetime completo nel formato RFC3339 richiesto da Google
    # Es: "2025-04-11T15:00:00" (timezone Europe/Rome per utenti italiani)
    timezone = "Europe/Rome"
    start_dt = f"{data}T{ora_inizio}:00"
    end_dt   = f"{data}T{ora_fine}:00"

    # Corpo dell'evento nel formato Google Calendar API
    event_body = {
        "summary"    : titolo,
        "description": descrizione,
        "start": {
            "dateTime": start_dt,
            "timeZone": timezone,
        },
        "end": {
            "dateTime": end_dt,
            "timeZone": timezone,
        },
        "reminders": {
            "useDefault": True,  # usa i promemoria default del calendario
        },
    }

    # Chiama Google Calendar API — crea l'evento
    created_event = service.events().insert(
        calendarId=calendar_id,
        body=event_body,
    ).execute()

    result = {
        "status"  : "creato",
        "event_id": created_event.get("id"),
        "link"    : created_event.get("htmlLink"),
        "titolo"  : titolo,
        "data"    : data,
        "inizio"  : ora_inizio,
        "fine"    : ora_fine,
    }

    return json.dumps(result, ensure_ascii=False)


# =============================================================================
# FUNZIONE PUBBLICA — chiamata da POST /v1/approve
# =============================================================================

def write_event_direct(event_details: dict) -> dict:
    """
    Scrive l'evento su Google Calendar direttamente, senza passare per MCP.
    Chiamata da main.py nel handler di POST /v1/approve dopo l'approvazione HITL.

    Args:
        event_details: dizionario con titolo, data, ora_inizio, ora_fine, descrizione
                       (lo stesso formato di CalendarEventRequest in models.py)

    Returns:
        dict con event_id e link dell'evento creato
    """
    calendar_id = "primary"  # default — usa il calendario principale dell'utente

    result_str = _write_calendar_event(
        calendar_id  = calendar_id,
        titolo       = event_details["titolo"],
        data         = event_details["data"],
        ora_inizio   = event_details["ora_inizio"],
        ora_fine     = event_details["ora_fine"],
        descrizione  = event_details.get("descrizione", ""),
    )

    return json.loads(result_str)


# =============================================================================
# AVVIO DEL SERVER MCP — eseguito nel lifespan FastAPI in thread separato
# =============================================================================

def start_mcp_server():
    """
    Avvia il MCP Server in modalità stdio.
    Da chiamare in un thread daemon nel lifespan FastAPI:

        import threading
        from app.mcp_server import start_mcp_server

        thread = threading.Thread(target=start_mcp_server, daemon=True)
        thread.start()

    daemon=True garantisce che il thread venga terminato automaticamente
    quando il processo principale (FastAPI) si chiude.

    Nota: il server MCP in modalità stdio comunica su stdin/stdout.
    In produzione si può configurare in modalità SSE per comunicazione HTTP.
    """
    import asyncio

    async def _run():
        async with stdio_server() as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )

    # Crea un nuovo event loop nel thread separato
    # (non può usare quello di FastAPI che gira nel thread principale)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_run())
    except Exception as e:
        print(f"[MCP Server] Errore: {e}")
    finally:
        loop.close()


# =============================================================================
# MAPPA DELLE CHIAMATE
# =============================================================================
#
#  FLUSSO NORMALE (dopo approvazione HITL):
#  ─────────────────────────────────────────
#  POST /v1/approve {approved: true}
#      │
#      ▼
#  main.py — endpoint approve
#      │  legge pending_calendar_event dallo State
#      │  chiama write_event_direct(event_details)   ← mcp_server.py
#      │      └→ _write_calendar_event()
#      │              └→ service.events().insert()   ← Google Calendar API
#      │
#      └→ risposta con event_id e link
#
#  FLUSSO MCP (scoperta dinamica del tool):
#  ─────────────────────────────────────────
#  LLM chiede lista tool al server MCP
#      │
#      ▼
#  list_tools() → restituisce schema di write_calendar_event
#      │
#  LLM decide di usare write_calendar_event
#      │
#      ▼
#  call_tool("write_calendar_event", {...})
#      │
#      └→ _write_calendar_event() → Google Calendar API
#
#  AVVIO:
#  ──────
#  FastAPI lifespan
#      │
#      ▼
#  threading.Thread(target=start_mcp_server, daemon=True).start()
#      │
#      ▼
#  MCP Server in ascolto su stdio in thread separato
#  FastAPI continua il suo avvio normalmente