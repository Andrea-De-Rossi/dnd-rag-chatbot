"""
ingest.py — Indicizzazione con FAISS + BM25

Usa FAISS al posto di ChromaDB:
- Niente SQLite = niente problemi di compatibilità
- File binari portabili (funzionano ovunque)
- Puoi committare il database su GitHub
"""

import os
import re
import shutil
import pickle
from rich.console import Console

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import config

console = Console()


def load_markdown(directory: str) -> list:
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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " "],
    )
    return splitter.split_documents(documents)


def build_bm25_index(chunks: list) -> None:
    from rank_bm25 import BM25Okapi

    tokenized = []
    for chunk in chunks:
        tokens = chunk.page_content.lower().split()
        tokenized.append(tokens)

    bm25 = BM25Okapi(tokenized)

    with open(config.BM25_INDEX_PATH, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "chunks": chunks,
            "tokenized": tokenized,
        }, f)

    console.print(f"[green]✅ Indice BM25 salvato in '{config.BM25_INDEX_PATH}'[/green]")


def build_vector_store(chunks: list) -> None:
    if os.path.exists(config.FAISS_INDEX_PATH):
        shutil.rmtree(config.FAISS_INDEX_PATH)

    console.print(f"🧠 Modello embeddings: [cyan]{config.EMBEDDING_MODEL}[/cyan]")

    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )

    console.print(f"📦 Creazione FAISS index con {len(chunks)} chunk...")

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    # Salva su disco — crea una cartella con index.faiss e index.pkl
    vectorstore.save_local(config.FAISS_INDEX_PATH)

    console.print(f"[green]✅ FAISS index salvato in '{config.FAISS_INDEX_PATH}/'[/green]")


def main():
    console.print("\n[bold]🐉 D&D RAG — Indicizzazione (FAISS + BM25)[/bold]\n")

    documents = load_markdown(config.PDF_DIRECTORY)
    console.print(f"\n📚 Sezioni caricate: [bold]{len(documents)}[/bold]")

    chunks = split_documents(documents)
    console.print(f"✂️  Chunk creati: [bold]{len(chunks)}[/bold]")

    console.print("\n[bold]1/2 — Indice BM25 (keyword search)[/bold]")
    build_bm25_index(chunks)

    console.print("\n[bold]2/2 — FAISS Index (semantic search)[/bold]")
    build_vector_store(chunks)

    console.print("\n[dim]--- Esempio di chunk ---[/dim]")
    console.print(f"[dim]{chunks[0].page_content[:300]}...[/dim]\n")


if __name__ == "__main__":
    main()