# =============================================================================
# scripts/test_demo.py — batteria di prova, un comando
# =============================================================================
# Da eseguire la mattina della riunione, con l'API gia' avviata:
#
#     python scripts/test_demo.py
#
# Stampa una tabella: domanda, agente instradato, risposto o rifiutato, fonte
# citata con pagina, punteggio di rerank piu' alto, tempo di risposta.
# Chiude con quanti rifiuti sono corretti e con la latenza a freddo.
#
# -----------------------------------------------------------------------------
# NOTA SULLA BATTERIA A — non e' quella del piano, e il motivo conta
# -----------------------------------------------------------------------------
# La batteria A di PIANO_DEMO.md interroga documenti che nel corpus attuale NON
# ESISTONO PIU': capitolato "Gruppo Distributivo Nord", verbali di non
# conformita', hamburger "di scottona", temperature di cella. Quel corpus e'
# stato sostituito allo step 3. Eseguirla alla lettera produrrebbe cinque
# rifiuti e li conterebbe come fallimenti, quando invece sarebbero corretti.
#
# Le domande qui sotto interrogano i tre documenti realmente indicizzati:
#   - Capitolato tecnico, mense comunali 2023/2026
#   - Manuale controllo ufficiale, sezionamento ungulati domestici
#   - Scheda tecnica, Hamburger B.A.
# =============================================================================

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import requests

sys.path.append(str(Path(__file__).parent.parent))

# La console Windows e' cp1252 e non sa codificare i trattini lunghi presenti
# nei titoli dei documenti. Senza questo lo script muore su una print.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
EMAIL    = os.getenv("DEMO_EMAIL", "demo@ambrosini.it")
PASSWORD = os.getenv("DEMO_PASSWORD", "demo2026")

RISPOSTA = "risposta"
RIFIUTO  = "rifiuto"

# =============================================================================
# LA BATTERIA
# =============================================================================
# `atteso` e' cio' che il sistema DEVE fare, giudicato su cio' che i documenti
# contengono — non su cio' che sembra plausibile chiedere.

BATTERIA = [
    # ── A — deve rispondere, citando ──────────────────────────────────────
    ("A", RISPOSTA, "Quali allergeni sono dichiarati nella scheda dell'hamburger?"),
    ("A", RISPOSTA, "Che pezzatura ha l'hamburger?"),
    ("A", RISPOSTA, "Qual e' la shelf-life dell'hamburger e come va conservato?"),
    ("A", RISPOSTA, "A che temperatura deve avvenire il trasporto delle carni secondo il capitolato?"),
    ("A", RISPOSTA, "Quali requisiti pone il manuale sul magazzinaggio delle carni sezionate?"),

    # ── B — deve rifiutare ────────────────────────────────────────────────
    # Nessuna di queste e' nel corpus. Una sola risposta inventata qui
    # significa che la demo non e' pronta.
    ("B", RIFIUTO, "Qual e' il fatturato dell'azienda?"),
    ("B", RIFIUTO, "Quanti dipendenti lavorano nello stabilimento?"),
    ("B", RIFIUTO, "Che cosa prevede il contratto collettivo per le ferie?"),
    ("B", RIFIUTO, "Qual e' il prezzo di vendita al pubblico consigliato?"),
    ("B", RIFIUTO, "Come si prepara la picanha alla brace?"),

    # ── B2 — rifiuti VICINI al corpus ─────────────────────────────────────
    # Piu' insidiosi dei precedenti: un capitolato parla davvero di condizioni
    # commerciali, quindi la domanda "suona" documentata.
    #
    # 'penali' e 'certificazioni' sono verificati ASSENTI dal testo estratto
    # dei tre PDF (grep su tutte le pagine): restano rifiuti corretti.
    ("B2", RIFIUTO, "Quali fornitori sono accreditati?"),
    ("B2", RIFIUTO, "Quanto costa al chilo la carne fornita?"),
    ("B2", RIFIUTO, "Che certificazioni possiede lo stabilimento?"),
    ("B2", RIFIUTO, "Quali penali sono previste per ritardo nella consegna?"),
    # 'chi effettua il controllo ufficiale' e' l'unica riclassificata: il
    # manuale tratta diffusamente l'autorita' competente e il veterinario
    # ufficiale, quindi il corpus PUO' rispondere.
    ("B2", RISPOSTA, "Chi effettua il controllo ufficiale in questo stabilimento?"),

    # ── C — riformulazioni, come le scrive davvero una persona ────────────
    # Stesso contenuto della batteria A, sintassi colloquiale. E' qui che il
    # pavimento di rilevanza fa i falsi rifiuti.
    ("C", RISPOSTA, "a che temperatura va tenuta la carne quando la sezioniamo"),
    ("C", RISPOSTA, "il capitolato cosa dice sul trasporto"),
    ("C", RISPOSTA, "l'hamburger ha allergeni?"),
    ("C", RISPOSTA, "quanto pesa un hamburger"),
    ("C", RISPOSTA, "come si identificano le carcasse"),
]


# =============================================================================
# API
# =============================================================================

def login() -> str:
    resp = requests.post(
        f"{API_BASE}/auth/login",
        data={"username": EMAIL, "password": PASSWORD},
        timeout=30,
    )
    if resp.status_code != 200:
        print(f"ERRORE: login fallito ({resp.status_code}) per {EMAIL}.")
        print("Controlla che l'API sia avviata e le credenziali in RUN_DEMO.md.")
        sys.exit(1)
    return resp.json()["access_token"]


def chiedi(token: str, domanda: str) -> tuple[dict | None, float]:
    """Invia la domanda. Restituisce (payload, secondi)."""
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{API_BASE}/v1/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={"query": domanda, "conversation_id": None},
            timeout=180,
        )
    except requests.RequestException as e:
        return {"errore": str(e)}, time.perf_counter() - t0
    dt = time.perf_counter() - t0

    if resp.status_code != 200:
        return {"errore": f"HTTP {resp.status_code}"}, dt
    return resp.json(), dt


# =============================================================================
# VALUTAZIONE
# =============================================================================

def estrai_fonte(risposta: str) -> str:
    """Ultima riga che inizia con 'Fonte:'. Stringa vuota se assente."""
    for riga in reversed(risposta.rstrip().split("\n")):
        if riga.strip().lower().startswith("fonte:"):
            return riga.strip()[len("fonte:"):].strip()
    return ""


def valuta(dati: dict) -> tuple[str, str, float]:
    """
    Restituisce (esito, fonte, top_score).

    L'esito si legge dalla FONTE, non dal testo: senza brani sopra soglia il
    sistema non cita, e una citazione presente e' la prova che ha risposto da
    un documento. Confrontare il testo col messaggio di rifiuto sarebbe fragile
    a ogni riscrittura di quel messaggio.
    """
    if "errore" in dati:
        return "ERRORE", dati["errore"], 0.0

    chunks = dati.get("chunks") or []
    top    = max((c.get("score", 0.0) for c in chunks), default=0.0)
    fonte  = estrai_fonte(dati.get("risposta", ""))
    return (RISPOSTA if fonte else RIFIUTO), fonte, top


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Batteria di prova per la demo.")
    ap.add_argument("--json", metavar="FILE", help="salva i risultati grezzi")
    args = ap.parse_args()

    print(f"API: {API_BASE}   utente: {EMAIL}")
    token = login()

    # ── Latenza a freddo ──────────────────────────────────────────────────
    # Prima query dopo il boot: paga l'inizializzazione degli agenti e il
    # primo caricamento del reranker. E' il numero che si vede in sala, ed e'
    # l'unico che nessuno misura. Va misurato su un'API appena avviata: se
    # l'API gira da un po', questo numero e' gia' "caldo" e va ignorato.
    print("\nMisura della latenza a freddo (prima query dopo il boot)...")
    _, freddo = chiedi(token, "Quali allergeni sono dichiarati nella scheda dell'hamburger?")
    print(f"  latenza a freddo: {freddo:.1f}s")

    # ── Batteria ──────────────────────────────────────────────────────────
    intestazione = (
        f"{'#':>3} {'bat':<4} {'atteso':<9} {'esito':<9} {'ok':<3} "
        f"{'agente':<12} {'top':>7} {'sec':>6}  domanda / fonte"
    )
    print("\n" + intestazione)
    print("-" * len(intestazione))

    risultati = []
    for i, (batteria, atteso, domanda) in enumerate(BATTERIA, 1):
        dati, dt = chiedi(token, domanda)
        esito, fonte, top = valuta(dati)

        corretto = (esito == atteso)
        agente   = (dati.get("agente_usato") or "—")[:12] if "errore" not in dati else "—"

        print(
            f"{i:>3} {batteria:<4} {atteso:<9} {esito:<9} "
            f"{'si' if corretto else 'NO':<3} {agente:<12} {top:>7.4f} {dt:>6.1f}  "
            f"{domanda[:58]}"
        )
        if fonte:
            print(f"{'':>58}{fonte[:80]}")

        risultati.append({
            "n": i, "batteria": batteria, "domanda": domanda,
            "atteso": atteso, "esito": esito, "corretto": corretto,
            "agente": dati.get("agente_usato", ""), "fonte": fonte,
            "top_score": top, "secondi": round(dt, 2),
            "risposta": dati.get("risposta", ""),
        })

    # ── Riepilogo ─────────────────────────────────────────────────────────
    def conta(pred):
        return sum(1 for r in risultati if pred(r))

    rifiuti_attesi   = conta(lambda r: r["atteso"] == RIFIUTO)
    rifiuti_corretti = conta(lambda r: r["atteso"] == RIFIUTO and r["corretto"])
    falsi_rifiuti    = [r for r in risultati
                        if r["atteso"] == RISPOSTA and r["esito"] == RIFIUTO]
    inventate        = [r for r in risultati
                        if r["atteso"] == RIFIUTO and r["esito"] == RISPOSTA]
    tempi            = [r["secondi"] for r in risultati]

    print("\n" + "=" * len(intestazione))
    print(f"Rifiuti corretti:      {rifiuti_corretti}/{rifiuti_attesi}")
    print(f"Risposte corrette:     "
          f"{conta(lambda r: r['atteso'] == RISPOSTA and r['corretto'])}"
          f"/{conta(lambda r: r['atteso'] == RISPOSTA)}")
    print(f"Totale corretti:       {conta(lambda r: r['corretto'])}/{len(risultati)}")
    print(f"Latenza a freddo:      {freddo:.1f}s")
    print(f"Latenza a caldo:       mediana {statistics.median(tempi):.1f}s, "
          f"max {max(tempi):.1f}s")

    if inventate:
        print("\nRISPOSTE INVENTATE — la demo NON e' pronta:")
        for r in inventate:
            print(f"  [{r['n']}] {r['domanda']}")
            print(f"       {r['fonte']}")

    if falsi_rifiuti:
        print("\nFALSI RIFIUTI — domande legittime rifiutate "
              "(il punteggio non ha superato la soglia):")
        for r in falsi_rifiuti:
            print(f"  [{r['n']}] top={r['top_score']:.4f}  {r['domanda']}")

    if args.json:
        Path(args.json).write_text(
            json.dumps(risultati, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nRisultati grezzi in {args.json}")

    # Exit code diverso da zero se qualcosa ha inventato: rende lo script
    # utilizzabile come check automatico, non solo come tabella da leggere.
    return 1 if inventate else 0


if __name__ == "__main__":
    sys.exit(main())
