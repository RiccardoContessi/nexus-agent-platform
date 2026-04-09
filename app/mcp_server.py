# =============================================================================
# app/mcp_server.py — MCP Server per Google Calendar
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

from app.config import get_settings, get_google_credentials_path
import logging

logger = logging.getLogger(__name__)
settings = get_settings()

# =============================================================================
# AUTENTICAZIONE GOOGLE — Service Account
# =============================================================================
SCOPES = ["https://www.googleapis.com/auth/calendar"]


def get_google_service():
    """
    Crea e restituisce il client autenticato per Google Calendar API.

    Usa Service Account JSON per autenticazione headless — appropriato
    per applicazioni server-side che girano senza interazione utente.

    Il path delle credenziali è risolto da get_google_credentials_path():
      - in locale/Docker → file montato come volume
      - su Railway/cloud → file temporaneo materializzato da GOOGLE_CREDENTIALS_B64
    """
    credentials_path = Path(get_google_credentials_path())

    if not credentials_path.exists():
        raise FileNotFoundError(
            f"Service Account JSON non trovato: {credentials_path}\n"
            "Verifica GOOGLE_SERVICE_ACCOUNT_PATH (locale) o "
            "GOOGLE_CREDENTIALS_B64 (cloud) nelle variabili d'ambiente."
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
    try:
        service = get_google_service()
        timezone = "Europe/Rome"
        start_dt = f"{data}T{ora_inizio}:00"
        end_dt   = f"{data}T{ora_fine}:00"
        event_body = {
            "summary"    : titolo,
            "description": descrizione,
            "start": {"dateTime": start_dt, "timeZone": timezone},
            "end"  : {"dateTime": end_dt,   "timeZone": timezone},
            "reminders": {"useDefault": True},
        }
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
        logger.info(f"[MCP] Evento creato: {result['link']}")
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        logger.error(f"[MCP] Errore scrittura calendario: {type(e).__name__}: {e}")
        raise

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
    calendar_id = settings.google_calendar_id
    
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