# Piano di lavoro — demo RAG per Ambrosini Carni

Adattamento di `nexus-agent-platform` a demo documentale per il settore alimentare.
Ogni step è un prompt da dare a Claude Code, uno alla volta.

**Prima di tutto:**

```bash
git checkout -b ambrosini-demo
```

Non lavorare su `main`. Quel repo è materiale da portfolio.

---

## Riepilogo del piano

| Step | Cosa | Tempo | Blocca? |
|---|---|---|---|
| 0 | Ambiente e verifica empirica dei metadati | 1h | Sì, tutto |
| 1 | Potatura: via Calendar, ML, Report | 1h | No |
| 2 | Documenti finti | — | Sì, step 3 |
| 3 | Ingestion del nuovo corpus | 45 min | Sì, step 4-7 |
| 4 | Citazioni con numero di pagina | 1h | No |
| 5 | Meccanismo di rifiuto | 2h | No |
| 6 | Streamlit spogliato | 30 min | No |
| 7 | Protocollo di prova | 1h | No |
| 8 | Atto 3 — verifica di conformità | 3h | Saltabile |

Appendici: correzione RAGAS (dopo la riunione), igiene dei segreti (prima).

---

## STEP 0 — Ambiente e verifica empirica

Lo step più importante, perché verifica l'assunto su cui poggia tutta la demo:
che il numero di pagina arrivi davvero in Pinecone. È dedotto dalla lettura del
codice, non provato.

```markdown
# Step 0 — Get the app running locally, and prove the page metadata claim

Two goals: make the app boot on this machine with no cloud dependency, and 
**empirically verify** that page numbers reach the vector index. The second is 
non-negotiable — the whole demo rests on it and it has only been inferred from 
reading code so far.

## 0.1 — Dependencies

Ten packages are missing and the app cannot start. Install everything in 
requirements.txt into the active environment. Report anything that fails to resolve.

`flashrank` downloads its ranking model on first import. **Trigger that download now**, 
then verify that a second import works with no network access — I need to know whether 
this app can run offline once warmed up. Report the answer explicitly.

## 0.2 — Local database

`.env:25` sets POSTGRES_URL to the Docker service hostname `db:5432`, which will not 
resolve when running uvicorn on the host.

Give me both options working:
- `docker-compose up db -d` with POSTGRES_URL pointing at `localhost:5432`
- confirm whether `docker-compose up --build` (everything in containers) also works

Do not remove the Docker path. I want the choice on the day.

## 0.3 — Boot verification

Start the API and confirm `/health` responds. Report which environment variables are 
strictly required to boot and which have working defaults. If anything fails hard when 
a remote service is unreachable, say which and where.

## 0.4 — THE CRITICAL CHECK: page metadata

The investigation concluded that `PyPDFLoader` yields one Document per page and that 
`split_documents(pages)` therefore keeps chunks inside page boundaries, so `page` 
propagates into Pinecone. This was **not verified empirically** because 
`langchain_experimental` was not installed.

Verify it now, end to end:

1. Take any multi-page PDF (use one already in `docs/`, or generate a 5-page test PDF 
   with distinct text on each page)
2. Run it through the actual ingestion path — not a simplified reproduction
3. Query the index directly and **print the full metadata dict of at least 5 chunks 
   drawn from different pages**
4. Report verbatim: exactly which metadata keys exist, and whether `page` is present, 
   correct, and zero- or one-indexed

If `page` is **not** present, stop and tell me before changing anything — that changes 
the plan materially.

Also report: does `_rerank` (tools.py:83-92) preserve the metadata dict on the documents 
it returns, or does reranking drop or rebuild it? Citations die there just as easily 
as in ingestion.

## 0.5 — Report

Short written summary: what works, what doesn't, the verbatim metadata output from 0.4, 
and the offline answer from 0.1.

**Do not modify any application logic in this step.** Configuration and environment only.
```

---

## STEP 1 — Potatura

Ogni agente in più è una strada in cui il supervisor può sbagliare davanti al cliente.

```markdown
# Step 1 — Remove the agents that don't belong in this demo

The demo covers document QA over food-industry documents. Calendar, ML and Report agents 
are not just unused — they are routing risk. If the supervisor misroutes a question about 
a supply specification to the calendar agent in front of the client, the demo is over.

## Remove completely

- **Calendar agent** and the entire HITL flow: the graph node, `interrupt_before`, 
  `pending_calendar` in SupervisorState, the `/v1/approve` endpoint, the MCP server and 
  its thread startup in the FastAPI lifespan, the Google service account loading in config
- **ML agent**: node, tool, prompts
- **Report agent**: node, tool, prompts

Delete the code, don't comment it out. It's on a branch; git remembers.

## Keep and restructure

The supervisor keeps two routes:

- `documenti_contrattuali` — supply specifications from retail chains, price lists
- `documenti_tecnici_e_sistema` — product technical sheets, HACCP procedures, 
  non-conformity reports

Each queries its own Pinecone namespaces. Reuse the existing parallel multi-namespace 
retrieval with `asyncio.gather` — that pattern is exactly right here and must survive 
the pruning intact.

## Check RoutingDecision specifically

Report what type the agent field is (enum, Literal, plain string). If it's free-form, 
make it a Literal over the two remaining routes, so a hallucinated agent name fails 
loudly at Pydantic validation instead of falling through the graph silently.

## Also

- JWT auth stays. Don't touch it.
- **Remove `groq_api_key` from the mandatory Settings fields.** Step 0 confirmed it 
  blocks boot, for a fallback that does not work.
- The Groq fallback claim must go from the README — `config.py:131` catches 
  AuthenticationError/RateLimitError around a constructor that makes no network call. 
  Those exceptions raise at `.invoke()`. Either fix it properly with a try/except at 
  invocation, or delete the claim from the docs. Tell me which you did and why.
- Fix the misleading comment at `docker-compose.yml:47` — it says `@db:5433`, but inside 
  the Docker network the port is 5432.
- **Report whether `document_type`, `language` or `topic` are used in any metadata filter 
  at retrieval time.** Step 0 showed these LLM-generated tags are unreliable (a PRML page 
  tagged `faq`, Italian text tagged `en`). If they filter retrieval, wrong tags break it 
  silently. Verify; do not assume they are decorative.

## Verify

App still boots, `/health` responds, a question still routes and answers. 
Report the routing decision for two sample questions.
```

---

## STEP 2 — Documenti

Li preparo io. Corpus previsto:

- Capitolato di fornitura di una catena inventata (Gruppo Distributivo Nord), 8-12 pagine
- Due schede tecniche Carni Valseriana S.r.l. — una con il buco voluto sull'origine
- Estratto di manuale HACCP, 4-5 pagine, con la procedura sulle temperature in cella
- Verbale di non conformità con azione correttiva
- Listino con codici e pezzature

---

## STEP 3 — Ingestion del nuovo corpus

```markdown
# Step 3 — Ingest the food-sector corpus

## 3.0 — Fresh Pinecone index, not a new namespace

The current index holds an undocumented `italiano` namespace with 4034 vectors — two 
thirds of the total — plus HR and ML content including Bishop's PRML. None of that 
belongs in this demo, and a namespace-filter bug would surface it in front of a client 
who has been told the system answers only from their documents.

Create a **new, dedicated index** for the demo. The free tier allows five and only one 
is in use. Do not delete the existing index or its namespaces — isolate, don't destroy.

Set `PINECONE_INDEX` to the new index. Confirm it starts empty.

Note: the README declares `exerag`, the live index is `rag-ai`. Fix the README.

## 3.1 — Corpus structure

Replace the HR/ML corpus with food-industry documents. Structure:

docs/
  capitolati/          → namespace "capitolati"
  schede_tecniche/     → namespace "schede_tecniche"
  procedure/           → namespace "procedure"
  non_conformita/      → namespace "non_conformita"
  listini/             → namespace "listini"

## Metadata — mandatory

Every chunk must carry, at minimum:
- `source` — human-readable document title, not the file path. Currently `source` is the 
  full Windows path (`C:\Users\ricca\Desktop\...`), which is unusable in a citation and 
  leaks the machine layout onto a projector. Define a mapping from filename to display 
  title; do not derive it from the path.
- `page` and `page_label` — both. See the citation rule below.
- `articolo` — for capitolati and procedures. **This is the primary citation unit**, 
  not the page. Extract the article or section reference (`Art. 4.2`, `§ 5.2`) where 
  the document structure provides it.
- `namespace`
- `revisione` — where present. Answering from a superseded revision is the failure mode 
  I care most about.

If a document declares a revision number and date on its first page, extract it during 
ingestion and attach it to every chunk of that document. If extraction is unreliable, 
build a manual lookup table in the ingestion script and say so — a hand-maintained map 
is better than a fragile regex here.

## Citation rule — decided

Verified in step 0: `page_label` is not reliably `page + 1`. Front matter shifts it, 
and it can be non-numeric (roman numerals).

- **Primary citation is document + revisione + articolo.** That reference survives 
  revisions and reformatting; a page number does not.
- **Page is a navigation aid**, and the number shown is `page_label` — what is printed 
  on the sheet, which is what the reader sees.
- If `page_label` is missing or non-numeric, fall back to `page + 1` and mark it as a 
  PDF position rather than a printed page.
- Always format as an integer. Pinecone returns `page` as a float; `pag. 13.0` in a 
  citation reads as carelessness.

The demo corpus is authored without front matter, so the two coincide. The code must 
still handle the divergence, because real client documents will have it.

## 3.2 — Check the LLM-generated metadata

Ingestion tags chunks with `document_type`, `language`, `topic` and similar, and those 
tags are demonstrably wrong on the existing corpus (a PRML page tagged `faq`, Italian 
text tagged `en`).

**Verify whether any of these fields is used in a metadata filter at retrieval time.** 
If they are, unreliable tagging breaks retrieval silently. Report the answer explicitly; 
do not assume they are decorative.

## Verify after ingestion

Print, for each namespace: chunk count, and the full metadata dict of 3 sample chunks. 
I want to see `source` as a display title, `page_label`, `articolo` and `revisione` 
populated correctly before moving on.

## Do not

Re-tune chunking parameters. The current SemanticChunker setup is verified to produce 
page-bounded chunks (32/32 in step 0). Changing it now would invalidate that.
```

---

## STEP 4 — Citazioni

```markdown
# Step 4 — Every answer cites its document and page

Right now `page` reaches the index and is then silently dropped: `_docs_to_string` 
(tools.py:95-114) reads only `source`, `namespace`, `topic`. This is the core promise 
of the demo — "the system always tells you which document and which page" — so it has 
to be structural, not a prompt suggestion.

## 4.1 — Surface the metadata

Verified in step 0: `_docs_to_string` (tools.py:95-114) currently passes the LLM
`[Documento 1 | procedure | HACCP.pdf | temperature]` — no page, no revision, and 
`source` as a full Windows path.

Change it to carry `source` (display title), `articolo`, `revisione`, `page_label` 
and `page`, in a format the model can reliably parse and attribute.

`_rerank` was verified to preserve metadata but discard the Flashrank score. Metadata 
is safe; the score is step 5's problem.

## 4.2 — Structured output

The RAG path currently returns free-form text. Move it to `with_structured_output`, 
the same mechanism already used for routing:

```python
class Fonte(BaseModel):
    documento: str              # display title
    revisione: str | None
    articolo: str | None        # PRIMARY reference where the document has one
    pagina: int | None          # from page_label, integer-formatted
    pagina_e_posizione_pdf: bool = False   # True when falling back to page + 1

class RispostaDocumentale(BaseModel):
    risposta: str
    fonti: list[Fonte]
    sufficiente: bool           # False when the context doesn't answer the question
    note: str | None            # e.g. "conflicting revisions found"
```

`sufficiente` is what step 5 builds on. Get the field in place now.

## 4.3 — Prompt

Instruct the model to cite from the metadata provided, never from memory, and never to 
invent a page or article number. If it cannot attribute a statement to a chunk, that 
statement does not belong in the answer.

**The article is the citation; the page is navigation.** Where a chunk carries an 
`articolo`, that must appear in the citation. The page number supports it, never 
replaces it.

## Verify

Ask three questions spanning different namespaces. For each, print the structured output 
in full. Then **open the source PDF and confirm both the article and the page number are 
correct** — Pinecone returns `page` as a float, and an integer-formatting slip or an 
off-by-one would be caught here and nowhere else.
```

---

## STEP 5 — Meccanismo di rifiuto

Lo step più delicato. Contiene la correzione del prompt che oggi istruisce
esplicitamente il modello a fare l'opposto.

```markdown
# Step 5 — The system must say "not found" instead of inferring

Two problems, and the first is worse than a missing feature: 
**prompts.py:108-109 actively instructs the model to answer from general knowledge 
when the documents fall short.** On an HACCP question that produces a confident, 
unsourced, plausible claim — precisely the behaviour I am selling against, delivered 
in front of a quality manager.

## 5.1 — Prompt surgery

Remove that instruction. Replace it with, in Italian:

"Rispondi esclusivamente sulla base dei documenti forniti. Se i documenti non 
contengono l'informazione richiesta, dichiaralo esplicitamente e indica quali 
documenti hai consultato. Non dedurre, non inferire, non integrare con conoscenza 
generale. In ambito di sicurezza alimentare una risposta plausibile ma non 
documentata è un errore grave."

Audit **every** prompt in prompts.py for similar instructions. Report anything else 
that permits or encourages unsourced answers.

## 5.2 — Relevance floor

Flashrank already computes a score per passage and `_rerank` (tools.py:83-92) throws 
it away. Stop discarding it.

- Keep the score on each reranked document
- Introduce a configurable threshold, `RERANK_SCORE_THRESHOLD`, in Settings
- Discard chunks below it before they ever reach the LLM
- If **no** chunk survives, do not call the LLM at all: return a deterministic 
  "not found" response with `sufficiente: false`

This is the part that makes refusal a mechanism rather than a hope.

## 5.3 — Calibrating the threshold

Do not guess the value. Run the corpus against two sets of questions:
- 5 questions that are answerable from the documents
- 5 questions that plainly are not (adjacent topics, plausible but absent)

Print the score distribution for both sets and **propose a threshold with the evidence 
behind it**. Report the false-negative risk: any answerable question whose top chunk 
scores near the cut.

## 5.4 — Conflicting revisions

If two chunks with the same `source` but different `revisione` both survive retrieval, 
the system must not silently pick one. It should answer from the most recent and state 
the fact, or flag the ambiguity in `note`.

## Verify

Show me both behaviours: an answerable question producing a cited answer, and an 
unanswerable one producing a clean refusal with no invented content.
```

---

## STEP 6 — Streamlit

```markdown
# Step 6 — Strip the UI down

This will be projected in a client meeting immediately after a polished design mock. 
Do not make it pretty — making it plain is the point. The contrast is deliberate: 
"that was a mock, this is code that runs."

## Single screen

- Login stays but is not the first thing seen — I'll authenticate before the meeting. 
  If a session survives a page reload, so much the better.
- One input box, full width
- The answer below it
- Under the answer, the sources: document, page, revision. Visually distinct from the 
  answer body — a hairline separator, smaller type. **They must be readable from the 
  back of a room.**
- When `sufficiente` is false, render the refusal plainly. No red, no warning icon, 
  no alarm. It is normal correct behaviour, not an error.

## Remove

Sidebar, calendar approval UI, conversation list, agent selectors, debug panels, 
anything not in the list above.

## Keep available but not visible

A way to show which chunks were retrieved and their scores — collapsed by default. 
If someone asks how it works, I want to open it. If nobody asks, it stays shut.

Font size large enough for projection. Test at 1440px.
```

---

## STEP 7 — Protocollo di prova

```markdown
# Step 7 — Test protocol before the meeting

Build a script that runs the full battery and prints results in a readable table. 
I need to run this the morning of, in one command.

## Battery A — must answer, with correct citation
1. "Il capitolato del Gruppo Distributivo Nord che limiti pone sulla carica microbica?"
2. "Che procedura seguiamo se in cella la temperatura supera i 4 gradi?"
3. "Che pezzature sono previste per gli hamburger a marchio del distributore?"
4. "Quali sono gli allergeni dichiarati nella scheda dell'hamburger di scottona?"
5. "Che azione correttiva è stata presa sull'ultima non conformità registrata?"

For each: the answer, the cited source and page, and **manual confirmation that the 
page is correct**.

## Battery B — must refuse, cleanly
6. "Qual è il fatturato dell'azienda?"
7. "Quanti dipendenti lavorano nello stabilimento?"
8. "Che cosa prevede il contratto collettivo per le ferie?"
9. "Qual è il prezzo di vendita al pubblico consigliato?"
10. "Come si prepara la picanha alla brace?"

None of these are in the corpus. Each must produce a refusal with no invented content. 
**A single fabricated answer here means the demo is not ready.**

## Battery C — adversarial
11. A question in dialect or with typos
12. A question spanning two namespaces
13. A question about a topic present in the corpus but at a level of detail that isn't
14. A deliberately ambiguous question

## Output

A table: question, routed agent, sufficiente, cited sources, top rerank score, 
response time. Plus a summary line: how many of Battery B refused correctly.

Also report **cold-start latency** — first query after boot, when caches are empty. 
That is the number that matters in a room, and it is the one nobody measures.
```

---

## STEP 8 — Atto 3: verifica di conformità (saltabile)

Non è recupero semantico. È estrazione strutturata più verifica mirata.
Se il tempo stringe, questo salta e in riunione lo racconti indicando lo schermo.

```markdown
# Step 8 — Conformity check: specification vs technical sheet

This is not semantic retrieval and must not be built as a chat question. It is a 
two-stage structured operation, exposed as its own endpoint and its own UI section.

## Stage 1 — Extract requirements

Input: a supply specification document.
Use `with_structured_output` to produce a list of:

```python
class Requisito(BaseModel):
    descrizione: str
    valore_richiesto: str | None
    pagina: int
    articolo: str | None
```

Print the extracted list. I want to inspect it before stage 2 consumes it.

## Stage 2 — Verify each requirement

For each requirement, run a **targeted** retrieval against the technical sheet's 
namespace only — not a general search:

```python
class Esito(BaseModel):
    requisito: Requisito
    stato: Literal["coperto", "non_coperto", "non_verificabile"]
    valore_trovato: str | None
    fonte: Fonte | None
    motivazione: str
```

`non_coperto` means the sheet addresses the point but fails to meet it.
`non_verificabile` means the sheet says nothing about it at all.
The distinction matters commercially and must not blur.

Parallelise with `asyncio.gather` if it's slow — ten sequential LLM calls will feel 
long in a room.

## The moment this exists for

The corpus contains a deliberate gap: the specification requires a declaration of 
place of slaughter, and the technical sheet does not carry it. The system must return 
`non_verificabile` on that requirement, with the specification's page cited.

**Verify this specific case works before anything else.** It is the only reason this 
step exists.

## UI

A separate section: pick a specification, pick a technical sheet, run. Results as three 
grouped lists — conformi, non conformi, non verificabili — with counts. Each row shows 
its source and page.
```

---

## Appendice A — RAGAS (dopo la riunione, non prima)

Percorso reale: `evalutation/ragas_eval.py` — la cartella ha un refuso, manca una `n`.
Il README documenta `evaluation/`, che non esiste.

`evalutation/ragas_eval.py:171` imposta `_cfg._settings = None`, un attributo che sul
modulo non esiste. Python lo accetta in silenzio; `get_settings` è `@lru_cache` e
continua a restituire l'oggetto in cache. Le due run hanno usato configurazione
identica, quindi i numeri nel README non provengono da quello script.

**Oggi, cinque minuti:** togli la tabella dei risultati dal README e sostituiscila con
una riga onesta. È su GitHub ed è materiale da portfolio.

**Dopo la riunione:**

```markdown
# Fix the RAGAS A/B harness

The real path is `evalutation/ragas_eval.py` — the directory name has a typo, missing 
an `n`. The README documents `evaluation/`, which does not exist. Fix the README too.

`evalutation/ragas_eval.py:171` sets `_cfg._settings = None`. That attribute does not 
exist on the module — Python accepts the assignment silently and it has no effect. 
`get_settings` is `@lru_cache`-decorated and keeps returning the cached Settings.

Both arms of the A/B test therefore ran with identical configuration, and the results 
in the README cannot have come from this script.

**Note the fix may not be one line.** Clearing the cache makes Settings re-read, but if 
the script never changes `os.environ["USE_PROMPT_REPETITION"]` first, it re-reads the 
same value from `.env` and nothing changes. Check both, and prove the flag actually 
differs between arms — assert it, don't assume it.

Then re-run and report the real numbers, whatever they are. Add an assertion that fails 
loudly if the two arms ever run with the same configuration again.
```

---

## Appendice B — Igiene dei segreti (prima della riunione)

- Il `.env` contiene chiavi vive e `credentials/` un service account reale.
  Correttamente non tracciati, ma **visibili se condividi lo schermo**.
- Prima della riunione: un solo file aperto nell'editor, niente file explorer in
  proiezione, cronologia del terminale svuotata.
- Il rapporto di investigazione sta in una cartella temporanea. Copialo nel repo in
  un file gitignored prima che venga ripulita.
- `flashrank` scarica il modello al primo import. Fallo scaricare oggi e verifica che
  al secondo avvio non serva rete.
