"""
app.py — D&D RAG Chatbot con interfaccia web

Lancia con: streamlit run app.py

Features:
- Interfaccia chat con messaggi stile ChatGPT
- Memoria conversazionale (ricorda le domande precedenti)
- Hybrid search (BM25 + Vector)
- Context window expansion
- Selezione modello dalla sidebar
"""

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import pickle
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Se siamo su Streamlit Cloud, leggi la API key dai Secrets
import config
if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
    config.GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

@st.cache_resource
def ensure_database():
    """Se il database non esiste, lo crea automaticamente."""
    if not os.path.exists(config.CHROMA_DIRECTORY) or not os.path.exists(config.BM25_INDEX_PATH):
        import shutil
        if os.path.exists(config.CHROMA_DIRECTORY):
            shutil.rmtree(config.CHROMA_DIRECTORY)
        st.info("⏳ Prima esecuzione: indicizzazione del manuale in corso...")
        # Importa e lancia direttamente (no subprocess)
        from ingest import main as run_ingest
        run_ingest()
    return True

# === CONFIGURAZIONE PAGINA ===
st.set_page_config(
    page_title="🐉 D&D RAG Chatbot",
    page_icon="🐉",
    layout="centered",
)

# === MODELLI DISPONIBILI ===
AVAILABLE_MODELS = {
    "Llama 4 Scout (veloce)": "meta-llama/llama-4-scout-17b-16e-instruct",
    "Llama 3.3 70B (bilanciato)": "llama-3.3-70b-versatile",
    "GPT-OSS 120B (preciso)": "openai/gpt-oss-120b",
}

# === PROMPT CON MEMORIA ===
PROMPT_WITH_MEMORY = """Sei un esperto di Dungeons & Dragons. Rispondi alle domande 
basandoti SOLO sul contesto fornito dal manuale. NON tradurre, NON parafrasare 
e NON inventare nomi: usa ESATTAMENTE i termini che trovi nel contesto.
Se il contesto non contiene l'informazione richiesta, dì chiaramente che non 
hai trovato quella informazione nel manuale. Rispondi in italiano.

Contesto dal manuale:
{context}

Conversazione precedente:
{history}

Domanda: {question}

Risposta:"""


# === FUNZIONI DI CARICAMENTO (cached) ===

@st.cache_resource
def load_vector_store():
    """Carica il vector store (cached, si carica una volta sola)."""
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )
    return Chroma(
        persist_directory=config.CHROMA_DIRECTORY,
        embedding_function=embeddings,
    )


@st.cache_resource
def load_bm25_index():
    """Carica l'indice BM25 (cached)."""
    if not os.path.exists(config.BM25_INDEX_PATH):
        return None
    with open(config.BM25_INDEX_PATH, "rb") as f:
        return pickle.load(f)


def get_llm(model_name: str) -> ChatGroq:
    """Crea l'istanza del modello LLM."""
    return ChatGroq(
        model=model_name,
        api_key=config.GROQ_API_KEY,
        temperature=0.3,
    )


# === HYBRID SEARCH ===

def expand_context(results, all_chunks, window=1):
    """Per ogni chunk trovato, include anche i chunk adiacenti."""
    expanded = []
    seen = set()

    for doc in results:
        for i, chunk in enumerate(all_chunks):
            if chunk.page_content == doc.page_content:
                start = max(0, i - window)
                end = min(len(all_chunks), i + window + 1)
                for j in range(start, end):
                    content_hash = hash(all_chunks[j].page_content[:200])
                    if content_hash not in seen:
                        seen.add(content_hash)
                        expanded.append(all_chunks[j])
                break

    return expanded


def hybrid_search(question, vectorstore, bm25_data, top_k=8):
    """Hybrid search con BM25 + Vector + RRF."""
    k_rrf = 60
    scores = {}
    chunk_map = {}

    # Vector search
    vector_results = vectorstore.similarity_search(question, k=top_k * 2)
    for rank, doc in enumerate(vector_results):
        doc_id = f"vec_{rank}_{hash(doc.page_content[:100])}"
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k_rrf + rank + 1)
        chunk_map[doc_id] = doc

    # BM25 search
    if bm25_data:
        bm25 = bm25_data["bm25"]
        chunks = bm25_data["chunks"]
        query_tokens = question.lower().split()
        bm25_scores = bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(bm25_scores)),
                             key=lambda i: bm25_scores[i], reverse=True)[:top_k * 2]

        for rank, idx in enumerate(top_indices):
            if bm25_scores[idx] > 0:
                doc = chunks[idx]
                doc_id = f"bm25_{rank}_{hash(doc.page_content[:100])}"
                found_existing = False
                for existing_id, existing_doc in chunk_map.items():
                    if existing_doc.page_content == doc.page_content:
                        scores[existing_id] += 1.0 / (k_rrf + rank + 1)
                        found_existing = True
                        break
                if not found_existing:
                    scores[doc_id] = 1.0 / (k_rrf + rank + 1)
                    chunk_map[doc_id] = doc

    # Ordina e deduplicaresults
    sorted_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
    results = []
    seen = set()
    for doc_id in sorted_ids:
        doc = chunk_map[doc_id]
        h = hash(doc.page_content[:200])
        if h not in seen:
            seen.add(h)
            results.append(doc)

    # Context expansion
    if bm25_data:
        results = expand_context(results, bm25_data["chunks"], window=1)

    return results[:10]


# === MEMORIA CONVERSAZIONALE ===

def format_history(messages, max_turns=4):
    """
    Formatta gli ultimi N scambi come stringa per il prompt.

    Mantiene solo le ultime max_turns domande/risposte
    per non riempire il contesto del modello.
    """
    history_pairs = []
    for i in range(0, len(messages) - 1, 2):
        if i + 1 < len(messages):
            q = messages[i]["content"]
            a = messages[i + 1]["content"]
            # Tronca risposte lunghe nella history
            if len(a) > 300:
                a = a[:300] + "..."
            history_pairs.append(f"Utente: {q}\nAssistente: {a}")

    # Prendi solo gli ultimi max_turns scambi
    recent = history_pairs[-max_turns:]

    if not recent:
        return "Nessuna conversazione precedente."

    return "\n\n".join(recent)


def ask_question(question, vectorstore, bm25_data, llm, messages):
    """Esegue la domanda con hybrid search e memoria."""

    # Retrieval
    source_docs = hybrid_search(question, vectorstore, bm25_data)
    context = "\n\n---\n\n".join(doc.page_content for doc in source_docs)

    # History
    history = format_history(messages)

    # Prompt
    prompt = PromptTemplate(
        template=PROMPT_WITH_MEMORY,
        input_variables=["context", "question", "history"],
    )
    formatted_prompt = prompt.format(
        context=context,
        question=question,
        history=history,
    )

    # LLM
    answer = llm.invoke(formatted_prompt)
    if hasattr(answer, 'content'):
        answer = answer.content

    return answer, source_docs


# === SIDEBAR ===

with st.sidebar:
    st.header("⚙️ Impostazioni")

    selected_model = st.selectbox(
        "Modello LLM",
        options=list(AVAILABLE_MODELS.keys()),
        index=0,
    )

    st.divider()

    st.markdown("**Statistiche sessione**")
    msg_count = len(st.session_state.get("messages", []))
    st.metric("Messaggi", msg_count)

    st.divider()

    if st.button("🗑️ Nuova conversazione"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.markdown("""
    **Stack tecnico:**
    - 📄 MinerU (PDF → Markdown)
    - 🔍 Hybrid Search (BM25 + Vector)
    - 🧠 Embeddings: multilingual-e5-large
    - 🤖 LLM: Groq Cloud
    - 💾 ChromaDB
    """)


# === INTERFACCIA CHAT ===

st.title("🐉 D&D RAG Chatbot")
st.caption("Fammi una domanda sul Player's Handbook di D&D 5e!")

# Inizializza stato sessione
if "messages" not in st.session_state:
    st.session_state.messages = []

ensure_database()

# Carica risorse
vectorstore = load_vector_store()
bm25_data = load_bm25_index()

# Mostra messaggi precedenti
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧙" if message["role"] == "user" else "🐉"):
        st.markdown(message["content"])

        # Mostra fonti se presenti
        if "sources" in message and message["sources"]:
            with st.expander(f"📖 Fonti ({len(message['sources'])})"):
                for i, src in enumerate(message["sources"], 1):
                    source = src.get("source_file", "?")
                    section = src.get("section_index", "?")
                    preview = src.get("preview", "")
                    st.caption(f"{i}. {source} (sez. {section}): {preview}")

# Input utente
if prompt := st.chat_input("Scrivi la tua domanda..."):

    # Mostra messaggio utente
    with st.chat_message("user", avatar="🧙"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    # Genera risposta
    with st.chat_message("assistant", avatar="🐉"):
        with st.spinner("🔍 Cerco nel manuale..."):
            model_id = AVAILABLE_MODELS[selected_model]
            llm = get_llm(model_id)

            answer, sources = ask_question(
                prompt, vectorstore, bm25_data, llm,
                st.session_state.messages
            )

        st.markdown(answer)

        # Mostra fonti
        if sources:
            source_data = []
            with st.expander(f"📖 Fonti ({len(sources)})"):
                for i, doc in enumerate(sources, 1):
                    source = doc.metadata.get("source_file", "?")
                    section = doc.metadata.get("section_index", "?")
                    preview = doc.page_content[:100].replace("\n", " ")
                    st.caption(f"{i}. {source} (sez. {section}): {preview}...")
                    source_data.append({
                        "source_file": source,
                        "section_index": section,
                        "preview": preview,
                    })

    # Salva risposta
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": source_data if sources else [],
    })