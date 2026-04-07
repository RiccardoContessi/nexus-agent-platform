# evaluation/ragas_eval.py
# ============================================================
# A/B Test — Prompt Repetition vs Prompt Standard
# Misura Faithfulness e ResponseRelevancy su 6 domande (3 HR + 3 ML)
# con RAGAS moderna (0.2.x)
# ============================================================

import os
import sys
import asyncio
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Aggiunge la root del progetto al path per importare i moduli app/
sys.path.append(str(Path(__file__).parent.parent))

load_dotenv()

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# RAGAS — API moderna 0.2.x
from ragas import evaluate, SingleTurnSample, EvaluationDataset
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.llms import LangchainLLMWrapper

# Agenti del progetto
from app.agents import build_hr_agent, build_ml_agent
from app.config import get_settings

settings = get_settings()

# ============================================================
# DATASET — 6 domande con ground truth
# ============================================================
DATASET = [
    # ── HR ───────────────────────────────────────────────────
    {
        "domanda"     : "Quanti giorni di ferie ho diritto per contratto?",
        "ground_truth": "I dipendenti hanno diritto a 26 giorni lavorativi di ferie annue.",
        "agente"      : "hr",
    },
    {
        "domanda"     : "Come si richiedono i permessi ROL?",
        "ground_truth": "I permessi ROL si richiedono tramite il portale HR con almeno 48 ore di anticipo.",
        "agente"      : "hr",
    },
    {
        "domanda"     : "Quali sono le condizioni per lo smart working?",
        "ground_truth": "Lo smart working è consentito fino a 3 giorni a settimana previa approvazione del manager.",
        "agente"      : "hr",
    },
    # ── ML ───────────────────────────────────────────────────
    {
        "domanda"     : "Qual è la differenza tra supervised e unsupervised learning?",
        "ground_truth": "Il supervised learning usa dati etichettati per mappare input a output. L'unsupervised learning trova pattern in dati non etichettati.",
        "agente"      : "ml",
    },
    {
        "domanda"     : "Cos'è il gradient descent e come funziona?",
        "ground_truth": "Il gradient descent è un algoritmo di ottimizzazione che minimizza la loss aggiornando i pesi nella direzione opposta al gradiente.",
        "agente"      : "ml",
    },
    {
        "domanda"     : "Cosa sono le reti neurali e come sono strutturate?",
        "ground_truth": "Le reti neurali sono modelli composti da layer di neuroni artificiali che applicano funzioni di attivazione per apprendere rappresentazioni dai dati.",
        "agente"      : "ml",
    },
]

# ============================================================
# INVOCAZIONE AGENTE — raccoglie risposta e chunk
# ============================================================
def invoke_agent(agente_compilato, query: str, thread_id: str) -> tuple[str, list[str]]:
    """
    Invoca un sotto-agente e restituisce (risposta, lista_chunk).
    Estrae i chunk dai ToolMessage nello state finale.
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = agente_compilato.invoke(
        {"messages": [HumanMessage(content=query)], "summary": ""},
        config=config,
    )

    # Estrae risposta finale — ultimo AIMessage con content
    risposta = ""
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and msg.content and not getattr(msg, "tool_calls", None):
            risposta = msg.content
            break

    # Estrae chunk dai ToolMessage
    chunks = []
    for msg in result["messages"]:
        from langchain_core.messages import ToolMessage
        if isinstance(msg, ToolMessage) and msg.content:
            chunks.append(msg.content)

    return risposta, chunks


# ============================================================
# VALUTAZIONE RAGAS
# ============================================================
def valuta_dataset(samples: list[dict]) -> dict:
    """
    Riceve lista di dict {domanda, risposta, chunks, ground_truth}.
    Restituisce dict {domanda: {faithfulness, response_relevancy}}.
    """
    # LLM per RAGAS — usa LangchainLLMWrapper
    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model="gpt-4o-mini",
            api_key=settings.openai_api_key,
            temperature=0,
        )
    )

    # Costruisce EvaluationDataset nel formato RAGAS
    ragas_samples = []
    for s in samples:
        ragas_samples.append(
            SingleTurnSample(
                user_input        =s["domanda"],
                response          =s["risposta"],
                retrieved_contexts=s["chunks"] if s["chunks"] else ["nessun contesto recuperato"],
                reference         =s["ground_truth"],
            )
        )

    dataset = EvaluationDataset(samples=ragas_samples)

    # Valuta con Faithfulness e ResponseRelevancy
    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(llm=evaluator_llm),
            ResponseRelevancy(llm=evaluator_llm),
        ],
    )

    df = result.to_pandas()

    # Mappa domanda → score
    scores = {}
    for i, s in enumerate(samples):
        scores[s["domanda"]] = {
            "faithfulness"      : round(df.iloc[i]["faithfulness"], 4),
            "response_relevancy": round(df.iloc[i]["response_relevancy"], 4),
        }

    return scores


# ============================================================
# MAIN — A/B Test completo
# ============================================================
def main():
    print("=" * 60)
    print("A/B Test — Prompt Repetition vs Prompt Standard")
    print("=" * 60)

    # ── Costruisce gli agenti ─────────────────────────────────
    # Run A: use_prompt_repetition = False
    os.environ["USE_PROMPT_REPETITION"] = "false"
    # Forza reload settings
    from app.config import get_settings as _gs
    import importlib, app.config as _cfg
    _cfg._settings = None  # resetta cache lru_cache

    hr_agent_a  = build_hr_agent()
    ml_agent_a  = build_ml_agent()

    # Run B: use_prompt_repetition = True
    os.environ["USE_PROMPT_REPETITION"] = "true"
    _cfg._settings = None

    hr_agent_b  = build_hr_agent()
    ml_agent_b  = build_ml_agent()

    # ── Raccoglie dati per Run A e Run B ──────────────────────
    samples_a = []
    samples_b = []

    for i, item in enumerate(DATASET):
        print(f"\n[{i+1}/6] {item['domanda'][:50]}...")

        agente_a = hr_agent_a if item["agente"] == "hr" else ml_agent_a
        agente_b = hr_agent_b if item["agente"] == "hr" else ml_agent_b

        # Run A — baseline
        print("  → Run A (baseline)...")
        risposta_a, chunks_a = invoke_agent(
            agente_a, item["domanda"],
            thread_id=f"eval_a_{i}"
        )

        # Run B — prompt repetition
        print("  → Run B (prompt repetition)...")
        risposta_b, chunks_b = invoke_agent(
            agente_b, item["domanda"],
            thread_id=f"eval_b_{i}"
        )

        samples_a.append({
            "domanda"     : item["domanda"],
            "risposta"    : risposta_a,
            "chunks"      : chunks_a,
            "ground_truth": item["ground_truth"],
        })
        samples_b.append({
            "domanda"     : item["domanda"],
            "risposta"    : risposta_b,
            "chunks"      : chunks_b,
            "ground_truth": item["ground_truth"],
        })

    # ── Valutazione RAGAS ─────────────────────────────────────
    print("\n\nValutazione RAGAS Run A (baseline)...")
    scores_a = valuta_dataset(samples_a)

    print("Valutazione RAGAS Run B (prompt repetition)...")
    scores_b = valuta_dataset(samples_b)

    # ── Report ────────────────────────────────────────────────
    righe = []
    sum_faith_a = sum_faith_b = 0
    sum_relev_a = sum_relev_b = 0

    for item in DATASET:
        d  = item["domanda"]
        fa = scores_a[d]["faithfulness"]
        fb = scores_b[d]["faithfulness"]
        ra = scores_a[d]["response_relevancy"]
        rb = scores_b[d]["response_relevancy"]

        sum_faith_a += fa; sum_faith_b += fb
        sum_relev_a += ra; sum_relev_b += rb

        righe.append({
            "domanda"     : d[:40] + "..." if len(d) > 40 else d,
            "faith_a"     : fa,
            "faith_b"     : fb,
            "delta_faith" : round(fb - fa, 4),
            "relev_a"     : ra,
            "relev_b"     : rb,
            "delta_relev" : round(rb - ra, 4),
        })

    n = len(DATASET)
    media_fa = round(sum_faith_a / n, 4)
    media_fb = round(sum_faith_b / n, 4)
    media_ra = round(sum_relev_a / n, 4)
    media_rb = round(sum_relev_b / n, 4)

    # ── Stampa tabella ────────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"{'Domanda':<42} | {'Faith Base':>10} | {'Faith PR':>8} | {'ΔFaith':>8} | {'Relev Base':>10} | {'Relev PR':>8} | {'ΔRelev':>8}")
    print("-" * 100)
    for r in righe:
        print(f"{r['domanda']:<42} | {r['faith_a']:>10.4f} | {r['faith_b']:>8.4f} | {r['delta_faith']:>+8.4f} | {r['relev_a']:>10.4f} | {r['relev_b']:>8.4f} | {r['delta_relev']:>+8.4f}")
    print("-" * 100)
    print(f"{'MEDIA':<42} | {media_fa:>10.4f} | {media_fb:>8.4f} | {round(media_fb-media_fa,4):>+8.4f} | {media_ra:>10.4f} | {media_rb:>8.4f} | {round(media_rb-media_ra,4):>+8.4f}")
    print("=" * 100)

    # ── Salva results.md ──────────────────────────────────────
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)
    results_path = output_dir / "results.md"

    with open(results_path, "w", encoding="utf-8") as f:
        f.write(f"# RAGAS A/B Test — Prompt Repetition\n\n")
        f.write(f"**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"**Modello:** {settings.llm_model}\n\n")
        f.write(f"**Metriche:** Faithfulness, ResponseRelevancy\n\n")
        f.write("## Risultati\n\n")
        f.write("| Domanda | Faith Base | Faith PR | ΔFaith | Relev Base | Relev PR | ΔRelev |\n")
        f.write("|---------|-----------|---------|-------|-----------|---------|-------|\n")
        for r in righe:
            f.write(f"| {r['domanda']} | {r['faith_a']} | {r['faith_b']} | {r['delta_faith']:+} | {r['relev_a']} | {r['relev_b']} | {r['delta_relev']:+} |\n")
        f.write(f"\n## Medie\n\n")
        f.write(f"| Metrica | Baseline | Prompt Repetition | Delta |\n")
        f.write(f"|---------|----------|------------------|-------|\n")
        f.write(f"| Faithfulness     | {media_fa} | {media_fb} | {round(media_fb-media_fa,4):+} |\n")
        f.write(f"| ResponseRelevancy | {media_ra} | {media_rb} | {round(media_rb-media_ra,4):+} |\n")
        f.write(f"\n## Conclusione\n\n")
        f.write(f"La Prompt Repetition ha prodotto un delta Faithfulness di **{round(media_fb-media_fa,4):+}** ")
        f.write(f"e un delta ResponseRelevancy di **{round(media_rb-media_ra,4):+}** su {n} domande.\n")

    print(f"\nReport salvato in: {results_path}")


if __name__ == "__main__":
    main()



# DATASET — 6 domande con ground truth hardcoded, 3 HR e 3 ML.
# invoke_agent — invoca un agente e separa risposta dai chunk estratti dai ToolMessage.
# valuta_dataset — costruisce EvaluationDataset con SingleTurnSample e chiama 
# evaluate() con Faithfulness e ResponseRelevancy. 
# Usa LangchainLLMWrapper che è il wrapper corretto per la RAGAS moderna.
# main — costruisce agenti con flag diverso per Run A e Run B, raccoglie i dati, 
# valuta, stampa la tabella e salva results.md.
# Nota importante: il reset del lru_cache per cambiare use_prompt_repetition a 
# runtime è necessario perché get_settings() è cachato. 
# Se nel tuo progetto il flag viene letto da settings che è già istanziato, 
# potresti dover adattare questo meccanismo.