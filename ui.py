# =============================================================================
# ui.py — Streamlit, schermata unica per la demo
# =============================================================================
# Una sola domanda, una sola risposta, le fonti sotto. Niente altro.
#
# Cosa è stato tolto rispetto alla versione precedente, e perché:
#   - sidebar, lista conversazioni, bottone "nuova conversazione"
#     → in riunione non si naviga la cronologia, si fa una domanda per volta
#   - badge dell'agente, toggle streaming, toggle prompt repetition
#     → sono dettagli di architettura: interessano chi ha scritto il sistema,
#       non chi lo sta valutando
#   - emoji ovunque
#     → il tema è la sicurezza alimentare, non un assistente giocattolo
#
# Cosa resta, e perché:
#   - il login, perché l'accesso è autenticato e va mostrato che lo è
#   - un pannello a scomparsa con i brani e i punteggi di rerank, CHIUSO di
#     default: si apre solo se qualcuno chiede "come fa a saperlo"
#
# Dimensioni tarate per proiezione a 1440px: il corpo della risposta è a 26px,
# le fonti a 19px. Le fonti sono la parte che il cliente deve poter leggere
# dal fondo della sala — è lì che si vede che il sistema non sta inventando.
# =============================================================================

import os

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

# Timeout generoso: la prima query dopo il boot paga l'inizializzazione del
# reranker e degli agenti. Un timeout stretto la farebbe fallire proprio nel
# momento peggiore, cioè alla prima domanda della riunione.
REQUEST_TIMEOUT = 120


# =============================================================================
# STILE
# =============================================================================

CSS = """
<style>
  /* Nasconde la cromatura Streamlit: menu, footer, header, deploy button. */
  #MainMenu, footer, header, [data-testid="stToolbar"],
  [data-testid="stDecoration"], [data-testid="stStatusWidget"] {
      display: none !important;
  }

  .block-container {
      max-width: 1100px;
      padding-top: 3rem;
      padding-bottom: 4rem;
  }

  /* ── Input della domanda ─────────────────────────────────────────────── */
  .stTextInput input {
      font-size: 24px !important;
      padding: 0.75em 0.9em !important;
      line-height: 1.4 !important;
  }

  /* ── Risposta ────────────────────────────────────────────────────────── */
  .risposta {
      font-size: 26px;
      line-height: 1.55;
      margin: 2rem 0 0 0;
      white-space: pre-wrap;
  }

  /* Il rifiuto è reso ESATTAMENTE come una risposta: stesso colore, stesso
     corpo, nessun bordo, nessuna icona. Non è un errore del sistema — è il
     sistema che funziona. Colorarlo di rosso insegnerebbe alla sala a
     leggerlo come un guasto. L'unica differenza è il corsivo. */
  .risposta.rifiuto {
      font-style: italic;
      opacity: 0.85;
  }

  /* ── Fonti ───────────────────────────────────────────────────────────── */
  /* Separate da un filetto sottile, non da un box: devono leggersi come una
     nota in calce al documento, non come un secondo blocco di contenuto. */
  .fonti {
      margin-top: 2.5rem;
      padding-top: 1.1rem;
      border-top: 1px solid rgba(128, 128, 128, 0.35);
      font-size: 19px;
      line-height: 1.5;
      opacity: 0.9;
  }
  .fonti .etichetta {
      font-size: 14px;
      letter-spacing: 0.09em;
      text-transform: uppercase;
      opacity: 0.55;
      display: block;
      margin-bottom: 0.4rem;
  }

  /* ── Pannello diagnostico ────────────────────────────────────────────── */
  .stExpander { margin-top: 2.5rem; }
  .chunk {
      font-family: ui-monospace, "Cascadia Mono", Consolas, monospace;
      font-size: 13px;
      line-height: 1.45;
      padding: 0.5rem 0;
      border-bottom: 1px dotted rgba(128, 128, 128, 0.3);
  }
</style>
"""


# =============================================================================
# API
# =============================================================================

def _login(email: str, password: str) -> bool:
    """Autentica e salva i token. /auth/login vuole un form OAuth2, non JSON."""
    try:
        resp = requests.post(
            f"{API_BASE}/auth/login",
            data={"username": email, "password": password},
            timeout=30,
        )
    except requests.RequestException:
        return False

    if resp.status_code != 200:
        return False

    dati = resp.json()
    st.session_state.access_token = dati["access_token"]
    return True


def _chiedi(domanda: str) -> dict | None:
    """
    Invia la domanda e restituisce la risposta completa.

    conversation_id resta None a ogni giro: ogni domanda è indipendente.
    In demo è la scelta giusta — una domanda non deve poter essere spiegata
    dal contesto della precedente, altrimenti non si vede quale documento
    ha risposto.
    """
    try:
        resp = requests.post(
            f"{API_BASE}/v1/chat",
            headers={"Authorization": f"Bearer {st.session_state.access_token}"},
            json={"query": domanda, "conversation_id": None},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.Timeout:
        st.session_state.errore = "La richiesta ha superato il tempo massimo."
        return None
    except requests.RequestException:
        st.session_state.errore = "Server non raggiungibile."
        return None

    if resp.status_code == 401:
        st.session_state.pop("access_token", None)
        st.session_state.errore = "Sessione scaduta: accedi di nuovo."
        return None

    if resp.status_code != 200:
        st.session_state.errore = f"Errore dal server ({resp.status_code})."
        return None

    return resp.json()


# =============================================================================
# RENDERING
# =============================================================================

def _separa_fonte(risposta: str) -> tuple[str, str]:
    """
    Divide il corpo della risposta dalla riga "Fonte: ...".

    Il modello chiude sempre con quella riga (vedi CITAZIONE_RULES in
    app/prompts.py). Qui viene staccata per poterla rendere sotto il filetto,
    con un corpo diverso. Se la riga non c'è — un rifiuto non ne ha — la
    risposta torna intera e le fonti restano vuote.
    """
    righe = risposta.rstrip().split("\n")
    for i in range(len(righe) - 1, -1, -1):
        if righe[i].strip().lower().startswith("fonte:"):
            corpo = "\n".join(righe[:i]).strip()
            fonte = righe[i].strip()[len("fonte:"):].strip()
            return corpo, fonte
    return risposta.strip(), ""


def _mostra_risposta(dati: dict) -> None:
    corpo, fonte = _separa_fonte(dati.get("risposta", ""))

    # Un rifiuto si riconosce dall'assenza di fonte: senza brani sopra soglia
    # il sistema non cita nulla. È l'unico segnale necessario e non richiede
    # di confrontare stringhe con il testo del messaggio di rifiuto.
    e_rifiuto = not fonte

    classe = "risposta rifiuto" if e_rifiuto else "risposta"
    st.markdown(f'<div class="{classe}">{corpo}</div>', unsafe_allow_html=True)

    if fonte:
        st.markdown(
            '<div class="fonti">'
            '<span class="etichetta">Fonte</span>'
            f"{fonte}"
            "</div>",
            unsafe_allow_html=True,
        )

    _mostra_diagnostica(dati)


def _mostra_diagnostica(dati: dict) -> None:
    """
    Pannello a scomparsa con i brani recuperati e il loro punteggio.

    Chiuso di default, sempre. Contiene anche i brani SCARTATI: su una domanda
    rifiutata sono l'unica cosa che spiega il rifiuto, e senza di essi il
    pannello sarebbe vuoto proprio quando serve.
    """
    chunks = dati.get("chunks") or []
    if not chunks:
        return

    soglia = dati.get("soglia", 0.0)
    ammessi = sum(1 for c in chunks if c.get("ammesso"))

    with st.expander(
        f"Brani recuperati: {ammessi} sopra soglia su {len(chunks)} "
        f"(soglia {soglia:.2f})",
        expanded=False,
    ):
        st.caption(f"Agente: {dati.get('agente_usato', '—')}")
        for c in sorted(chunks, key=lambda x: x.get("score", 0), reverse=True):
            esito  = "AMMESSO " if c.get("ammesso") else "SCARTATO"
            pagina = f" · pag. {c['pagina']}" if c.get("pagina") else ""
            testo  = " ".join(c.get("estratto", "").split())[:220]
            st.markdown(
                f'<div class="chunk"><b>{c.get("score", 0):.4f} · {esito}</b> — '
                f'{c.get("source", "")}{pagina}<br>{testo}</div>',
                unsafe_allow_html=True,
            )


# =============================================================================
# SCHERMATE
# =============================================================================

def schermata_login() -> None:
    st.markdown("### Accesso")
    with st.form("login"):
        email    = st.text_input("Email")
        password = st.text_input("Password", type="password")
        entra    = st.form_submit_button("Accedi", use_container_width=True)

    if entra:
        if _login(email, password):
            st.rerun()
        else:
            st.error("Credenziali non valide.")


def schermata_demo() -> None:
    domanda = st.text_input(
        "Domanda",
        key="domanda",
        placeholder="Fai una domanda sui documenti…",
        label_visibility="collapsed",
    )

    # Streamlit rieseque lo script a ogni interazione. Senza confrontare con
    # l'ultima domanda servita, la stessa domanda verrebbe rilanciata a ogni
    # rerun — una chiamata LLM per ogni click sulla pagina.
    if domanda and domanda != st.session_state.get("ultima_domanda"):
        st.session_state.ultima_domanda = domanda
        st.session_state.errore = ""
        with st.spinner(""):
            st.session_state.risultato = _chiedi(domanda)

    if st.session_state.get("errore"):
        st.warning(st.session_state.errore)

    if st.session_state.get("risultato"):
        _mostra_risposta(st.session_state.risultato)


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    st.set_page_config(
        page_title="Assistente documentale",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    for chiave, valore in {
        "access_token"   : None,
        "ultima_domanda" : "",
        "risultato"      : None,
        "errore"         : "",
    }.items():
        st.session_state.setdefault(chiave, valore)

    if not st.session_state.access_token:
        schermata_login()
    else:
        schermata_demo()


if __name__ == "__main__":
    main()
