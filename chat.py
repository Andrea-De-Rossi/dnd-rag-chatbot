"""
chat_v3.py — Chatbot RAG con Hybrid Search

Miglioramenti:
1. Hybrid Search: combina BM25 (keyword) + Vector (semantic)
2. Reciprocal Rank Fusion per unire i risultati
3. Embeddings migliori

Il risultato è MOLTO più preciso perché:
- La ricerca vettoriale trova chunk con significato simile
- BM25 trova chunk con le stesse parole
- Insieme coprono sia "sentinella" (keyword match)
  che "quali classi ci sono" (semantic match)
"""

import os
import pickle
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

import config

console = Console()


def load_vector_store() -> Chroma:
    """Carica il vector store."""
    if not os.path.exists(config.CHROMA_DIRECTORY):
        console.print("[red]Database non trovato! Esegui prima: python ingest.py[/red]")
        raise SystemExit(1)

    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )

    return Chroma(
        persist_directory=config.CHROMA_DIRECTORY,
        embedding_function=embeddings,
    )


def load_bm25_index():
    """Carica l'indice BM25."""
    if not os.path.exists(config.BM25_INDEX_PATH):
        console.print("[yellow]⚠ Indice BM25 non trovato, uso solo ricerca vettoriale[/yellow]")
        return None

    with open(config.BM25_INDEX_PATH, "rb") as f:
        return pickle.load(f)

def expand_context(results: list, all_chunks: list, window: int = 1) -> list:
    """
    Per ogni chunk trovato, aggiunge anche i chunk adiacenti.
    
    Così se il retrieval trova il chunk con Aquila e Lupo,
    automaticamente include anche quello dopo con Orso.
    
    window=1 significa: 1 chunk prima + il chunk trovato + 1 chunk dopo
    """
    expanded = []
    seen = set()
    
    for doc in results:
        # Trova l'indice di questo chunk nella lista completa
        for i, chunk in enumerate(all_chunks):
            if chunk.page_content == doc.page_content:
                # Aggiungi i chunk nella finestra
                start = max(0, i - window)
                end = min(len(all_chunks), i + window + 1)
                
                for j in range(start, end):
                    content_hash = hash(all_chunks[j].page_content[:200])
                    if content_hash not in seen:
                        seen.add(content_hash)
                        expanded.append(all_chunks[j])
                break
    
    return expanded

def hybrid_search(question: str, vectorstore: Chroma, bm25_data: dict, top_k: int = 6) -> list:
    """
    Hybrid Search con Reciprocal Rank Fusion (RRF).

    Come funziona:
    1. Cerca con BM25 (keyword) → classifica per parole in comune
    2. Cerca con Vector (semantic) → classifica per significato simile
    3. Combina i due ranking con RRF → il meglio di entrambi

    RRF formula: score = sum(1 / (k + rank)) per ogni sistema
    dove k=60 è una costante che bilancia i contributi.
    """
    k_rrf = 60  # Costante RRF (standard)
    scores = {}  # {chunk_id: score}
    chunk_map = {}  # {chunk_id: Document}

    # --- RICERCA VETTORIALE ---
    vector_results = vectorstore.similarity_search(question, k=top_k * 2)

    for rank, doc in enumerate(vector_results):
        doc_id = f"vec_{rank}_{hash(doc.page_content[:100])}"
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k_rrf + rank + 1)
        chunk_map[doc_id] = doc

    # --- RICERCA BM25 ---
    if bm25_data:
        bm25 = bm25_data["bm25"]
        chunks = bm25_data["chunks"]

        query_tokens = question.lower().split()
        bm25_scores = bm25.get_scores(query_tokens)

        # Prendi i top risultati BM25
        top_indices = sorted(range(len(bm25_scores)),
                             key=lambda i: bm25_scores[i], reverse=True)[:top_k * 2]

        for rank, idx in enumerate(top_indices):
            if bm25_scores[idx] > 0:  # Solo risultati con score > 0
                doc = chunks[idx]
                doc_id = f"bm25_{rank}_{hash(doc.page_content[:100])}"

                # Controlla se questo chunk è già nei risultati vettoriali
                # (in quel caso, somma lo score = boost!)
                found_existing = False
                for existing_id, existing_doc in chunk_map.items():
                    if existing_doc.page_content == doc.page_content:
                        scores[existing_id] += 1.0 / (k_rrf + rank + 1)
                        found_existing = True
                        break

                if not found_existing:
                    scores[doc_id] = 1.0 / (k_rrf + rank + 1)
                    chunk_map[doc_id] = doc

    # --- ORDINA PER SCORE RRF E PRENDI I TOP K ---
    sorted_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]

    results = []
    seen_content = set()
    for doc_id in sorted_ids:
        doc = chunk_map[doc_id]
        # Evita duplicati
        content_hash = hash(doc.page_content[:200])
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            results.append(doc)
    
    # Espandi il contesto con i chunk adiacenti
    if bm25_data:
        results = expand_context(results, bm25_data["chunks"], window=1)

    return results


def ask_question(question: str, vectorstore: Chroma, bm25_data: dict, llm) -> dict:
    """Il cuore del RAG con hybrid search."""

    # Step 1: Hybrid Search
    source_docs = hybrid_search(question, vectorstore, bm25_data, top_k=config.TOP_K_RESULTS)

    # Step 2: Costruisci il contesto
    context = "\n\n---\n\n".join(doc.page_content for doc in source_docs)

    # Step 3: Prompt
    prompt = PromptTemplate(
        template=config.SYSTEM_PROMPT,
        input_variables=["context", "question"],
    )
    formatted_prompt = prompt.format(context=context, question=question)

    # Step 4: LLM
    answer = llm.invoke(formatted_prompt)
    if hasattr(answer, 'content'):
        answer = answer.content

    return {
        "result": answer,
        "source_documents": source_docs,
    }


def show_sources(source_docs: list) -> None:
    """Mostra le fonti utilizzate."""
    console.print("\n[dim]📖 Fonti utilizzate:[/dim]")
    for i, doc in enumerate(source_docs, 1):
        source = doc.metadata.get("source_file", "?")
        section = doc.metadata.get("section_index", "?")
        preview = doc.page_content[:100].replace("\n", " ")
        console.print(f"[dim]  {i}. {source} (sez. {section}): {preview}...[/dim]")


def main():
    console.print(Panel.fit(
        "🐉 [bold]D&D RAG Chatbot v3[/bold] — Hybrid Search\n"
        "Fammi una domanda sul manuale di D&D!\n"
        "Scrivi [bold]'esci'[/bold] per uscire.",
        border_style="red",
    ))

    # Carica i database
    console.print("⏳ Caricamento vector store...")
    vectorstore = load_vector_store()

    console.print("⏳ Caricamento indice BM25...")
    bm25_data = load_bm25_index()

    # Connessione al LLM
    console.print("⏳ Connessione a Groq...")
    llm = ChatGroq(
        model=config.GROQ_MODEL,
        api_key=config.GROQ_API_KEY,
        temperature=0.3,
    )

    console.print("[green]✅ Pronto! (Hybrid Search attivo)[/green]\n")

    while True:
        question = console.input("[bold red]Tu:[/bold red] ").strip()

        if not question:
            continue
        if question.lower() in ("esci", "exit", "quit", "q"):
            console.print("👋 Alla prossima avventura!")
            break

        console.print("[dim]🔍 Cerco nel manuale (BM25 + Vector)...[/dim]")

        try:
            result = ask_question(question, vectorstore, bm25_data, llm)

            console.print()
            console.print("[bold green]🐉 DM:[/bold green]")
            console.print(Markdown(result["result"]))

            show_sources(result["source_documents"])
            console.print()

        except Exception as e:
            console.print(f"[red]Errore: {e}[/red]")


if __name__ == "__main__":
    main()