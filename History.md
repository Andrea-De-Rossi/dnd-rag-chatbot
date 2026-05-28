# 🐉 D&D RAG Chatbot — Cronologia del Progetto

## Obiettivo
Costruire un chatbot RAG (Retrieval-Augmented Generation) che risponde a domande sul Player's Handbook di D&D 5e, partendo da zero e migliorando iterativamente ogni componente.

---

## Fase 1 — Setup iniziale

### Stack scelto
- **Python 3.11** via Miniconda (3.11 per compatibilità con chromadb e sentence-transformers)
- **Ollama + Mistral 7B** come LLM locale
- **ChromaDB** come vector database
- **LangChain** come framework di orchestrazione
- **PyPDF** per l'estrazione del testo dal PDF

### Problemi incontrati
- **LangChain breaking changes**: le versioni nuove di LangChain hanno spostato molti moduli. `langchain.text_splitter` → `langchain_text_splitters`, `langchain.chains.RetrievalQA` rimosso del tutto, `langchain.schema.Document` → `langchain_core.documents`.
- **Soluzione**: riscritto il codice senza usare le chain deprecate, facendo retrieval e chiamata LLM "a mano" — più chiaro e meno soggetto a rotture future.

### Risultato
Il chatbot funziona ma le risposte sono scadenti. Il modello allucina nomi ("Cacciatore di mostri" al posto di "Warlock") e molte informazioni non vengono trovate.

---

## Fase 2 — Miglioramento estrazione PDF

### Problema
PyPDF estraeva male il PDF del manuale italiano (scansione). **83 pagine su 321 risultavano illeggibili** — pagine quasi vuote o con caratteri corrotti (�).

### Tentativo 1: PyMuPDF
- Pagine problematiche scese da 83 a 25 (quelle con solo illustrazioni)
- Aggiunta funzione `clean_text()` per rimuovere caratteri corrotti
- **Miglioramento sensibile**, ma artefatti OCR ancora presenti ("SENtINELLA", "I1" al posto di "Il")

### Tentativo 2: MinerU (pipeline backend, su Kaggle con GPU T4)
- Installato su Kaggle perché richiede GPU e modelli pesanti
- Backend `hybrid-auto-engine` fallito per incompatibilità CUDA su Kaggle (vLLM non trovava `libcuda.so`, filesystem read-only)
- Usato backend `pipeline` — 321 pagine processate in 19 secondi
- Output: Markdown strutturato da **1.551.716 caratteri**, 19.108 righe
- Tutte le 12 classi e sezioni principali presenti nel Markdown
- **Artefatti OCR ancora presenti** negli header ("SENtINELLA", "PriVILEGI Di CLassE") perché il PDF di origine è una scansione

### Tentativo 3: PDF inglese
- Provato un PDF in inglese del Player's Handbook
- Anche questo risulta essere una scansione con artefatti OCR (793 errori trovati: "c1ass", "Vou", "levei", "creatllre")
- Meglio dell'italiano (niente caratteri � corrotti) ma non pulito al 100%

### Lezione appresa
La qualità del PDF di origine è il fattore più critico. Un PDF nativo (es. acquistato digitalmente) darebbe risultati perfetti. Le scansioni richiedono sempre post-processing.

---

## Fase 3 — Miglioramento del modello LLM

### Problema
Mistral 7B in locale allucinava e non rispettava il contesto. Inventava traduzioni di nomi ("Cacciatore di mostri" per Warlock) e rispondeva in modo impreciso.

### Soluzione: Groq API (gratuita)
Passato da Ollama locale a Groq cloud, che offre modelli molto più potenti gratis:
- Llama 3.3 70B
- GPT-OSS 120B (OpenAI open-weight)
- Llama 4 Scout 17B
- Qwen3 32B

### Benchmark comparativo (7 domande × 4 modelli)

| Modello | Tempo medio | Risposte trovate | 12 classi |
|---|---|---|---|
| Llama 4 Scout 17B | **0.54s** | **4/7** | 11/12 |
| Llama 3.3 70B | 0.73s | 3/7 | 10/12 |
| GPT-OSS 120B | 6.46s | 3/7 | **12/12** |
| Qwen3 32B | 16.39s | 3/7 | 11/12 |

### Osservazioni
- **Llama 4 Scout**: miglior tradeoff velocità/qualità
- **GPT-OSS 120B**: più preciso sulle domande complesse, ma 10x più lento
- **Qwen3 32B**: include tag `<think>` visibili all'utente (ragionamento interno esposto), da filtrare
- **Nessun modello trovava il talento Sentinella** → il problema era nel retrieval, non nel modello

### Lezione appresa
Oltre un certo livello, cambiare modello non migliora le risposte se il retrieval non fornisce i chunk giusti. Il bottleneck era l'indicizzazione, non la generazione.

---

## Fase 4 — Miglioramento del retrieval

### Problema 1: Embedding model inadeguato
Il modello `all-MiniLM-L6-v2` era addestrato principalmente su inglese e non capiva bene le query in italiano.

- **Soluzione**: passato a `intfloat/multilingual-e5-large` (~560MB), molto più preciso per il retrieval multilingue.

### Problema 2: Solo ricerca vettoriale (semantica)
La ricerca per similarità coseno trovava chunk con significato simile, ma non quelli con le stesse parole. "Sentinella" come query non matchava il chunk "SENtINELLA" perché il significato semantico era troppo lontano.

- **Soluzione**: implementato **Hybrid Search** combinando:
  - **BM25** (ricerca keyword classica, tipo Google) per trovare match esatti di parole
  - **Vector Search** (ricerca semantica) per trovare match di significato
  - **Reciprocal Rank Fusion (RRF)** per unire i due ranking: `score = Σ 1/(k + rank)` per ogni sistema. Se un chunk appare in entrambe le ricerche, il suo score viene sommato (boost).

### Problema 3: Chunk troppo piccoli
I chunk da 1000 caratteri tagliavano le informazioni a metà. La lista delle 12 classi poteva stare a cavallo tra due chunk.

- **Soluzione**: aumentato `CHUNK_SIZE` a 1500 e `CHUNK_OVERLAP` a 300.

### Risultato
- Domanda "12 classi giocabili": **12/12 classi trovate** al primo tentativo
- Domanda "talento Sentinella": **trovato e descritto correttamente** con tutti e 3 gli effetti
- Il BM25 trova il chunk per keyword match diretto, il vector search porta contesto aggiuntivo

### Lezione appresa
L'hybrid search è lo standard in produzione per un motivo. La ricerca puramente semantica ha punti ciechi (nomi propri, termini tecnici). BM25 li copre perfettamente.

---

## Fase 4b — Context Window Expansion

### Problema
Il Cammino Totemico del Barbaro ha tre opzioni (Orso, Aquila, Lupo). La sezione nel Markdown è ~1600 caratteri, ma con `CHUNK_SIZE=1500` veniva tagliata in due chunk: il primo con Aquila e Lupo, il secondo con Orso. Il chatbot rispondeva solo con Aquila e Lupo, perdendo Orso.

### Analisi
Il problema non era nel retrieval (trovava il chunk giusto) né nel modello, ma nel **chunking che spezzava una sezione logica a metà**. Anche aumentando `TOP_K`, il secondo chunk non veniva recuperato perché la query "opzioni totem barbaro" matchava solo il primo.

### Soluzione 1: Aumentare chunk size
Portato `CHUNK_SIZE` da 1500 a 2000 e `CHUNK_OVERLAP` da 300 a 400, così sezioni più lunghe restano intere.

### Soluzione 2: Context Window Expansion
Implementata una funzione `expand_context()` che per ogni chunk trovato dal retrieval, include automaticamente anche i chunk adiacenti (1 prima + 1 dopo). Questo garantisce che se il retrieval trova la prima metà di una sezione, la seconda metà viene inclusa nel contesto.

```python
def expand_context(results, all_chunks, window=1):
    # Per ogni chunk trovato, aggiungi chunk[i-1] e chunk[i+1]
```

Tecnica nota come **Sentence Window Retrieval** o **Small-to-Big Retrieval**: si cercano chunk piccoli (più precisi nel matching) ma si espande il contesto passato al LLM.

### Risultato
- Domanda "opzioni animali cammino totemico": **Orso, Aquila, Lupo** — tutti e tre trovati
- Il contesto ora include l'intera sezione del Cammino Totemico con tutti i sotto-privilegi (Spirito Totemico, Aspetto della Bestia, Sintonia Totemica)

### Lezione appresa
In un sistema RAG, la dimensione dei chunk è un tradeoff:
- **Chunk piccoli** → matching più preciso, ma rischio di informazioni tagliate
- **Chunk grandi** → più contesto, ma matching meno preciso e più rumore
- **Context Window Expansion** → il meglio di entrambi: cerca con precisione, rispondi con contesto completo

---

## Fase 5 — Post-processing del Markdown (in corso)

### Problema
Gli header del Markdown hanno casing strana ("SENtINELLA", "PriVILEGI Di CLassE") e artefatti OCR ("I1" → "Il", parole attaccate).

### Soluzione pianificata
Script `fix_markdown_ai.py` che manda le sezioni a Llama 3.3 70B su Groq per correzione automatica degli errori OCR, senza alterare il contenuto.

### Prossimi passi
- [ ] Eseguire pulizia AI del Markdown
- [ ] Testare MinerU con backend `hybrid-auto-engine` su Google Colab (CUDA compatibile)
- [ ] Confrontare qualità Markdown: pipeline vs hybrid backend
- [ ] Re-eseguire benchmark dopo pulizia per misurare il miglioramento
- [ ] Aggiungere query rewriting per riformulare domande ambigue
- [ ] Limitare chunk espansi con `MAX_CONTEXT_CHUNKS` per non sovraccaricare il contesto LLM

---

## Stack finale

```
PDF → MinerU (pipeline, Kaggle GPU) → Markdown
                                        ↓
                                   fix_markdown_ai.py (Groq)
                                        ↓
                                   Markdown pulito
                                        ↓
                            ┌───────────────────────────┐
                            │      ingest.py            │
                            │  ┌─────────┐ ┌─────────┐ │
                            │  │  BM25   │ │ ChromaDB│ │
                            │  │ (keyword)│ │(vector) │ │
                            │  └─────────┘ └─────────┘ │
                            └───────────────────────────┘
                                        ↓
                            ┌───────────────────────────┐
                            │      chat.py              │
                            │  Hybrid Search (RRF)      │
                            │  + Context Expansion      │
                            │  + Llama 4 Scout (Groq)   │
                            └───────────────────────────┘
```

## Metriche di miglioramento

| Versione | Classi trovate | Sentinella | Totem completo | Tempo risposta |
|---|---|---|---|---|
| v1 (PyPDF + Mistral locale) | 4/12 | ❌ | ❌ | ~15s |
| v2 (MinerU + Groq + vector only) | 10-12/12 | ❌ | ❌ | ~0.6s |
| v3 (MinerU + Groq + hybrid search) | **12/12** | **✅** | ❌ (2/3) | ~0.5s |
| v3b (+ context window expansion) | **12/12** | **✅** | **✅ (3/3)** | ~0.5s |

---

*Progetto realizzato come preparazione per colloquio presso ZenData AI (Roma) — Maggio 2026*