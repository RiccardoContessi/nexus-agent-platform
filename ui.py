# =============================================================================
# ui.py — Streamlit Frontend
# =============================================================================
# Interfaccia utente per Enterprise Agent Platform.
#
# Flusso:
#   1. Se no token → form login/register
#   2. Se token → layout principale con sidebar e chat
#
# Session state keys:
#   access_token, refresh_token, user_email
#   conversation_id, conversations, messages
#   last_agent, streaming_enabled
#   pending_event (per HITL calendario)
# =============================================================================

import json
import requests
import streamlit as st
import os
API_BASE = os.getenv("API_BASE", "http://localhost:8000")


# =============================================================================
# HELPERS API
# =============================================================================

def _headers() -> dict:
    """Header JWT per ogni chiamata API autenticata."""
    return {"Authorization": f"Bearer {st.session_state.access_token}"}


def _api_call(method: str, path: str, **kwargs) -> requests.Response | None:
    """
    Wrapper per tutte le chiamate API.
    Gestisce automaticamente il rinnovo del token se scaduto (401).
    Se il refresh fallisce, resetta la sessione e torna al login.
    """
    url      = f"{API_BASE}{path}"
    response = getattr(requests, method)(url, headers=_headers(), **kwargs)

    if response.status_code == 401:
        # Access token scaduto — tenta il rinnovo con il refresh token
        refreshed = _refresh_token()
        if refreshed:
            # Riprova la chiamata con il nuovo access token
            response = getattr(requests, method)(url, headers=_headers(), **kwargs)
        else:
            # Refresh fallito — torna al login
            _logout()
            return None

    return response


def _refresh_token() -> bool:
    """
    Rinnova l'access token usando il refresh token.
    Restituisce True se il rinnovo ha successo, False altrimenti.
    """
    try:
        resp = requests.post(
            f"{API_BASE}/auth/refresh",
            params={"refresh_token": st.session_state.refresh_token},
        )
        if resp.status_code == 200:
            data = resp.json()
            st.session_state.access_token  = data["access_token"]
            st.session_state.refresh_token = data["refresh_token"]
            return True
    except Exception:
        pass
    return False


def _logout():
    """Resetta la sessione e forza il ritorno al login."""
    for key in ["access_token", "refresh_token", "user_email",
                "conversation_id", "conversations", "messages",
                "last_agent", "pending_event"]:
        st.session_state.pop(key, None)
    st.rerun()


# =============================================================================
# FORM LOGIN / REGISTER
# =============================================================================

def show_auth_form():
    """
    Mostra il form di autenticazione quando non c'è un token valido.
    Tabs separati per login e registrazione.
    """
    st.title("🤖 Enterprise Agent Platform")
    st.caption("Accedi per iniziare a chattare con gli agenti AI")
    st.divider()

    tab_login, tab_register = st.tabs(["Accedi", "Registrati"])

    with tab_login:
        with st.form("login_form"):
            email    = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit   = st.form_submit_button("Accedi", use_container_width=True)

        if submit:
            if not email or not password:
                st.error("Inserisci email e password")
                return

            resp = requests.post(
                f"{API_BASE}/auth/login",
                data={"username": email, "password": password},  # OAuth2 form
            )
            if resp.status_code == 200:
                data = resp.json()
                st.session_state.access_token  = data["access_token"]
                st.session_state.refresh_token = data["refresh_token"]
                st.session_state.user_email    = email
                st.session_state.messages      = []
                st.session_state.conversation_id = None
                st.rerun()
            else:
                st.error("Credenziali non valide")

    with tab_register:
        with st.form("register_form"):
            full_name = st.text_input("Nome completo")
            email_r   = st.text_input("Email", key="reg_email")
            password_r= st.text_input("Password (min 8 caratteri)", type="password", key="reg_pass")
            submit_r  = st.form_submit_button("Registrati", use_container_width=True)

        if submit_r:
            if not full_name or not email_r or not password_r:
                st.error("Compila tutti i campi")
                return
            if len(password_r) < 8:
                st.error("Password troppo corta (min 8 caratteri)")
                return

            resp = requests.post(f"{API_BASE}/auth/register", json={
                "email": email_r, "password": password_r, "full_name": full_name
            })
            if resp.status_code == 201:
                st.success("Registrazione completata! Ora accedi.")
            elif resp.status_code == 409:
                st.error("Email già registrata")
            else:
                st.error("Errore nella registrazione")


# =============================================================================
# SIDEBAR
# =============================================================================

def show_sidebar():
    """
    Sidebar con:
    - Info utente e logout
    - Toggle streaming e prompt repetition
    - Lista conversazioni cliccabili
    - Bottone nuova conversazione
    """
    with st.sidebar:
        st.title("💬 Enterprise Agent")
        st.caption(f"👤 {st.session_state.get('user_email', '')}")

        # ── Logout ────────────────────────────────────────────────────────────
        if st.button("🚪 Logout", use_container_width=True):
            _logout()

        st.divider()

        # ── Badge ultimo agente ───────────────────────────────────────────────
        last_agent = st.session_state.get("last_agent", "")
        if last_agent:
            agent_emoji = {
                "hr_agent"      : "👥",
                "ml_agent"      : "🤖",
                "report_agent"  : "📊",
                "calendar_agent": "📅",
            }.get(last_agent, "🔹")
            st.info(f"{agent_emoji} Ultimo agente: **{last_agent}**")

        st.divider()

        # ── Toggle streaming ──────────────────────────────────────────────────
        st.session_state.streaming_enabled = st.toggle(
            "⚡ Streaming",
            value=st.session_state.get("streaming_enabled", False),
            help="Mostra la risposta token per token in tempo reale",
        )

        # ── Toggle Prompt Repetition ──────────────────────────────────────────
        # Nota: il toggle aggiorna solo la visualizzazione locale.
        # In produzione chiamerebbe PATCH /v1/settings per aggiornare il flag
        # sul server. Per semplicità mostriamo solo il valore corrente.
        st.toggle(
            "🔬 Prompt Repetition",
            value=True,
            disabled=True,
            help="Tecnica attiva server-side (configurabile via .env)",
        )

        st.divider()

        # ── Nuova conversazione ───────────────────────────────────────────────
        if st.button("➕ Nuova conversazione", use_container_width=True):
            st.session_state.conversation_id = None
            st.session_state.messages        = []
            st.session_state.last_agent      = ""
            st.session_state.pending_event   = None
            st.rerun()

        st.divider()

        # ── Lista conversazioni ───────────────────────────────────────────────
        st.subheader("🗂️ Conversazioni")
        _load_conversations()

        conversations = st.session_state.get("conversations", [])
        if not conversations:
            st.caption("Nessuna conversazione ancora")
        else:
            for conv in conversations:
                label = f"{conv['title'][:30]}..." if len(conv['title']) > 30 else conv['title']

                col_open, col_del = st.columns([5, 1])
                with col_open:
                    if st.button(label, key=f"conv_{conv['id']}", use_container_width=True):
                        _load_conversation(conv["id"])
                        st.rerun()
                with col_del:
                    if st.button("🗑️", key=f"del_{conv['id']}", use_container_width=True):
                        _delete_conversation(conv["id"])
                        st.rerun()


def _load_conversations():
    """Carica la lista conversazioni dal DB."""
    resp = _api_call("get", "/v1/conversations")
    if resp and resp.status_code == 200:
        st.session_state.conversations = resp.json().get("conversations", [])


def _load_conversation(conversation_id: str):
    """Carica i messaggi di una conversazione esistente."""
    resp = _api_call("get", f"/v1/conversations/{conversation_id}/messages")
    if resp and resp.status_code == 200:
        data = resp.json()
        st.session_state.conversation_id = conversation_id
        st.session_state.messages        = data.get("messages", [])
        st.session_state.pending_event   = None


def _delete_conversation(conversation_id: str):
    """
    Cancella una conversazione via DELETE /v1/conversations/{id}.
    Se era quella attiva, resetta la chat corrente.
    Ricarica la lista conversazioni dopo la cancellazione.
    """
    resp = _api_call("delete", f"/v1/conversations/{conversation_id}")
    if not resp or resp.status_code not in (200, 204):
        st.error("Errore nella cancellazione della conversazione")
        return

    # Se la conversazione cancellata era quella attiva, resetta la chat
    if str(st.session_state.get("conversation_id")) == str(conversation_id):
        st.session_state.conversation_id = None
        st.session_state.messages        = []
        st.session_state.last_agent      = ""
        st.session_state.pending_event   = None

    _load_conversations()


# =============================================================================
# CHAT PRINCIPALE
# =============================================================================

def show_chat():
    """Area principale di chat con history e input utente."""
    st.title("🤖 Enterprise Agent Platform")

    conv_id = st.session_state.get("conversation_id")
    if conv_id:
        st.caption(f"Conversazione: `{str(conv_id)[:8]}...`")
    else:
        st.caption("Nuova conversazione")

    st.divider()

    # ── Mostra history ────────────────────────────────────────────────────────
    messages = st.session_state.get("messages", [])
    for msg in messages:
        role    = msg.get("role", "ai")
        content = msg.get("content", "")

        with st.chat_message("user" if role == "human" else "assistant"):
            st.markdown(content)

            # Caption con metadati agente e tool (solo per messaggi AI)
            if role == "ai":
                agente = msg.get("agente_usato", "")
                tools  = msg.get("tools_usati") or []
                if agente or tools:
                    caption = ""
                    if agente:
                        caption += f"🔹 {agente}"
                    if tools:
                        caption += f" | 🔧 {', '.join(tools)}"
                    st.caption(caption)

    # ── Evento calendario in attesa di approvazione ───────────────────────────
    pending = st.session_state.get("pending_event")
    if pending:
        show_pending_approval(pending)
        # Durante HITL il grafo è sospeso: blocca l'input chat e mostra solo
        # i bottoni Conferma/Annulla per evitare crash su nuove invocazioni.
        st.chat_input("Conferma o annulla l'evento per continuare...", disabled=True)
        return

    # ── Input utente ──────────────────────────────────────────────────────────
    if prompt := st.chat_input("Scrivi un messaggio..."):
        # Aggiunge subito il messaggio utente alla history visuale
        st.session_state.messages.append({"role": "human", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Invoca l'agente con o senza streaming
        if st.session_state.get("streaming_enabled", False):
            _invoke_streaming(prompt)
        else:
            _invoke_standard(prompt)


def _invoke_standard(prompt: str):
    """Invoca il Supervisor in modalità standard (risposta completa)."""
    with st.chat_message("assistant"):
        with st.spinner("⏳ Elaborazione in corso..."):
            resp = _api_call("post", "/v1/chat", json={
                "query"          : prompt,
                "conversation_id": st.session_state.get("conversation_id"),
            })

        if not resp or resp.status_code != 200:
            st.error("Errore nella comunicazione con il server")
            return

        data = resp.json()

        # Salva conversation_id se era una nuova conversazione
        if "conversation_id" in data:
            st.session_state.conversation_id = data["conversation_id"]

        # Gestisce pending_approval (Calendar Agent HITL)
        if data.get("status") == "pending_approval":
            st.session_state.pending_event = data.get("event", {})
            st.rerun()
            return

        # Risposta normale
        risposta     = data.get("risposta", "")
        agente_usato = data.get("agente_usato", "")
        tools_usati  = data.get("tools_usati", [])

        st.markdown(risposta)

        if agente_usato or tools_usati:
            caption = ""
            if agente_usato:
                caption += f"🔹 {agente_usato}"
            if tools_usati:
                caption += f" | 🔧 {', '.join(tools_usati)}"
            st.caption(caption)

        # Aggiorna session state
        st.session_state.messages.append({
            "role"        : "ai",
            "content"     : risposta,
            "agente_usato": agente_usato,
            "tools_usati" : tools_usati,
        })
        st.session_state.last_agent = agente_usato

        # Ricarica la lista conversazioni per aggiornare i contatori
        _load_conversations()


def _invoke_streaming(prompt: str):
    """
    Invoca il Supervisor in modalità streaming SSE.
    Aggiorna il placeholder token per token usando requests con stream=True.
    """
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            with requests.post(
                f"{API_BASE}/v1/chat/stream",
                headers=_headers(),
                json={
                    "query"          : prompt,
                    "conversation_id": st.session_state.get("conversation_id"),
                },
                stream=True,
                timeout=120,
            ) as response:

                if response.status_code == 401:
                    if _refresh_token():
                        st.rerun()
                    else:
                        _logout()
                    return

                for line in response.iter_lines():
                    if line:
                        line = line.decode("utf-8")
                        if line.startswith("data: "):
                            token = line[6:]  # rimuove il prefisso "data: "
                            if token == "[DONE]":
                                break
                            full_response += token
                            # Aggiorna il placeholder con il testo accumulato
                            placeholder.markdown(full_response + "▌")

        except requests.exceptions.Timeout:
            st.warning("Timeout — la risposta ha impiegato troppo tempo")
            return
        except Exception as e:
            st.error(f"Errore streaming: {e}")
            return

        # Mostra la risposta finale senza il cursore
        placeholder.markdown(full_response)

        # Aggiorna session state
        st.session_state.messages.append({
            "role"        : "ai",
            "content"     : full_response,
            "agente_usato": "",   # non disponibile in streaming
            "tools_usati" : [],
        })

        _load_conversations()


# =============================================================================
# PENDING APPROVAL — approvazione evento calendario
# =============================================================================

def show_pending_approval(event: dict):
    """
    Mostra i dettagli dell'evento calendario in attesa di approvazione.
    Bottoni Conferma e Annulla che chiamano POST /v1/approve.
    """
    st.divider()
    st.subheader("📅 Evento in attesa di approvazione")

    with st.container(border=True):
        st.markdown(f"**📌 Titolo:** {event.get('titolo', 'N/D')}")
        st.markdown(f"**📆 Data:** {event.get('data', 'N/D')}")
        st.markdown(f"**🕐 Inizio:** {event.get('ora_inizio', 'N/D')}")
        st.markdown(f"**🕑 Fine:** {event.get('ora_fine', 'N/D')}")
        if event.get("descrizione"):
            st.markdown(f"**📝 Note:** {event.get('descrizione')}")

    col_ok, col_no = st.columns(2)

    with col_ok:
        if st.button("✅ Conferma evento", use_container_width=True, type="primary"):
            _send_approval(approved=True)

    with col_no:
        if st.button("❌ Annulla", use_container_width=True):
            _send_approval(approved=False)

    st.divider()


def _send_approval(approved: bool):
    """Invia la decisione di approvazione al backend."""
    conv_id = st.session_state.get("conversation_id")
    if not conv_id:
        st.error("ID conversazione mancante")
        return

    resp = _api_call("post", "/v1/approve", json={
        "conversation_id": str(conv_id),
        "approved"       : approved,
    })

    if not resp or resp.status_code != 200:
        st.error("Errore nell'invio dell'approvazione")
        return

    data     = resp.json()
    message  = data.get("message", "")
    status   = data.get("status", "")

    # Aggiorna la chat con la risposta
    st.session_state.messages.append({
        "role"        : "ai",
        "content"     : message,
        "agente_usato": "calendar_agent",
        "tools_usati" : ["create_calendar_event"] if approved else [],
    })

    # Pulisce il pending event
    st.session_state.pending_event = None
    st.session_state.last_agent    = "calendar_agent"

    _load_conversations()
    st.rerun()


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    st.set_page_config(
        page_title ="Enterprise Agent Platform",
        page_icon  ="🤖",
        layout     ="wide",
        initial_sidebar_state="expanded",
    )

    # Inizializza session state alla prima esecuzione
    defaults = {
        "access_token"     : None,
        "refresh_token"    : None,
        "user_email"       : None,
        "conversation_id"  : None,
        "conversations"    : [],
        "messages"         : [],
        "last_agent"       : "",
        "streaming_enabled": False,
        "pending_event"    : None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Routing principale: login o app
    if not st.session_state.access_token:
        show_auth_form()
    else:
        show_sidebar()
        show_chat()


if __name__ == "__main__":
    main()