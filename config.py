"""
Configurazione del progetto D&D RAG.
"""

# === PERCORSI ===
PDF_DIRECTORY = "docs"
FAISS_INDEX_PATH = "faiss_index"
BM25_INDEX_PATH = "bm25_index.pkl"

# === MODELLO LLM (Groq Cloud) ===
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# === EMBEDDINGS ===
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# === CHUNKING ===
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400

# === RETRIEVAL ===
TOP_K_RESULTS = 8

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