"""
ingest_v3.py — Indicizzazione avanzata con BM25 + Vector Search

Miglioramenti rispetto alla versione precedente:
1. Embeddings migliori (multilingual-e5-large)
2. Salva anche un indice BM25 per ricerca keyword
3. Chunking più intelligente
"""

import os
import re
import json
import shutil
import pickle
from rich.console import Console

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import config

console = Console()


def load_markdown(directory: str) -> list:
    """Carica i file Markdown dalla cartella docs/."""
    documents = []
    md_files = [f for f in os.listdir(directory) if f.endswith((".md", ".markdown"))]

    if not md_files:
        console.print(f"[red]Nessun Markdown trovato in '{directory}/'[/red]")
        raise SystemExit(1)

    for filename in md_files:
        filepath = os.path.join(directory, filename)
        console.print(f"📄 Caricamento: [cyan]{filename}[/cyan]")

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Spezza per sezioni Markdown (## Titolo)
        raw_sections = re.split(r'(?=^## )', content, flags=re.MULTILINE)

        for i, section in enumerate(raw_sections):
            if len(section.strip()) < 50:
                continue

            documents.append(Document(
                page_content=section.strip(),
                metadata={
                    "source_file": filename,
                    "section_index": i,
                }
            ))

        console.print(f"   → {len(documents)} sezioni estratte")

    return documents


def split_documents(documents: list) -> list:
    """Spezza in chunk con separatori ottimizzati per Markdown."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=[
            "\n## ", "\n### ", "\n#### ",
            "\n\n", "\n", ". ", " ",
        ],
    )
    return splitter.split_documents(documents)


def build_bm25_index(chunks: list) -> None:
    """
    Costruisce un indice BM25 per la ricerca keyword.

    BM25 è un algoritmo di ricerca testuale classico (come Google).
    Trova documenti che contengono le stesse PAROLE della query.
    È complementare alla ricerca vettoriale che trova documenti
    con lo stesso SIGNIFICATO.

    Insieme = hybrid search = molto più efficace.
    """
    from rank_bm25 import BM25Okapi

    # Tokenizza ogni chunk (split per parole, lowercase)
    tokenized = []
    for chunk in chunks:
        tokens = chunk.page_content.lower().split()
        tokenized.append(tokens)

    bm25 = BM25Okapi(tokenized)

    # Salva l'indice e i chunk
    with open(config.BM25_INDEX_PATH, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "chunks": chunks,
            "tokenized": tokenized,
        }, f)

    console.print(f"[green]✅ Indice BM25 salvato in '{config.BM25_INDEX_PATH}'[/green]")


def build_vector_store(chunks: list) -> None:
    """Crea gli embeddings e salvali in ChromaDB."""
    if os.path.exists(config.CHROMA_DIRECTORY):
        shutil.rmtree(config.CHROMA_DIRECTORY)

    console.print(f"🧠 Modello embeddings: [cyan]{config.EMBEDDING_MODEL}[/cyan]")
    console.print("   (il primo avvio scarica il modello, potrebbe volerci un po')")

    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )

    console.print(f"📦 Creazione vector store con {len(chunks)} chunk...")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=config.CHROMA_DIRECTORY,
    )

    console.print(f"[green]✅ Vector store salvato in '{config.CHROMA_DIRECTORY}/'[/green]")


def main():
    console.print("\n[bold]🐉 D&D RAG — Indicizzazione Avanzata (BM25 + Vector)[/bold]\n")

    documents = load_markdown(config.PDF_DIRECTORY)
    console.print(f"\n📚 Sezioni caricate: [bold]{len(documents)}[/bold]")

    chunks = split_documents(documents)
    console.print(f"✂️  Chunk creati: [bold]{len(chunks)}[/bold]")

    # Costruisci entrambi gli indici
    console.print("\n[bold]1/2 — Indice BM25 (keyword search)[/bold]")
    build_bm25_index(chunks)

    console.print("\n[bold]2/2 — Vector Store (semantic search)[/bold]")
    build_vector_store(chunks)

    console.print("\n[dim]--- Esempio di chunk ---[/dim]")
    console.print(f"[dim]{chunks[0].page_content[:300]}...[/dim]\n")


if __name__ == "__main__":
    main()