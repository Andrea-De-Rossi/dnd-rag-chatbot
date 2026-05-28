"""
Configurazione del progetto D&D RAG v3.
"""

# === PERCORSI ===
PDF_DIRECTORY = "docs"
CHROMA_DIRECTORY = "chroma_db"
BM25_INDEX_PATH = "bm25_index.pkl"

# === MODELLO LLM (Groq Cloud) ===
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# === MODELLO LLM LOCALE (Ollama, se preferisci) ===
OLLAMA_MODEL = "mistral"
OLLAMA_BASE_URL = "http://localhost:11434"

# === EMBEDDINGS ===
# Opzione 1: Multilingual E5 Large (MIGLIORE per retrieval multilingue)
#   - ~560MB, più lento ma molto più preciso
#   - Richiede prefisso "query: " per le query
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# Opzione 2: Se il PC è lento, usa questo (più leggero)
# EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# === CHUNKING ===
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300

# === RETRIEVAL ===
TOP_K_RESULTS = 5

# === PROMPT ===
SYSTEM_PROMPT = """Sei un esperto di Dungeons & Dragons. Rispondi alle domande 
basandoti SOLO sul contesto fornito dal manuale. NON tradurre, NON parafrasare 
e NON inventare nomi: usa ESATTAMENTE i termini che trovi nel contesto.
Se il contesto non contiene l'informazione richiesta, dì chiaramente che non 
hai trovato quella informazione nel manuale. Rispondi in italiano.

Contesto dal manuale:
{context}

Domanda: {question}

Risposta:"""