"""
ingest_markdown.py — Indicizza il Markdown di MinerU in ChromaDB

Versione alternativa che legge il Markdown prodotto da MinerU
invece del PDF originale. Il Markdown è GIA' pulito e strutturato,
quindi l'indicizzazione è molto più precisa.

Eseguilo UNA VOLTA dopo aver ottenuto il .md da MinerU.
"""

import os
import shutil
from rich.console import Console

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import config

console = Console()


def load_markdown(directory: str) -> list:
    """
    Carica tutti i file Markdown dalla cartella docs/.
    Ogni file viene letto come testo puro — molto più pulito del PDF.
    """
    documents = []
    md_files = [f for f in os.listdir(directory) if f.endswith((".md", ".markdown"))]

    # Fallback: cerca anche PDF (per compatibilità)
    if not md_files:
        console.print(f"[yellow]Nessun Markdown trovato in '{directory}/'[/yellow]")
        console.print("Cercando PDF come fallback...")
        console.print("[red]Per usare questo script, metti il .md di MinerU in docs/[/red]")
        raise SystemExit(1)

    for filename in md_files:
        filepath = os.path.join(directory, filename)
        console.print(f"📄 Caricamento: [cyan]{filename}[/cyan]")

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Spezza per sezioni Markdown (## Titolo)
        # Questo mantiene la struttura del documento
        sections = content.split("\n## ")

        for i, section in enumerate(sections):
            if i > 0:
                section = "## " + section  # riaggiungi il ## tolto dallo split

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
    """
    Spezza i documenti in chunk.
    Con il Markdown di MinerU usiamo separatori specifici
    per rispettare la struttura del documento.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=[
            "\n## ",    # Titoli di sezione (priorità massima)
            "\n### ",   # Sotto-sezioni
            "\n#### ",  # Sotto-sotto-sezioni
            "\n\n",     # Paragrafi
            "\n",       # Righe
            ". ",       # Frasi
            " ",        # Parole
        ],
    )

    chunks = splitter.split_documents(documents)
    return chunks


def create_vector_store(chunks: list) -> None:
    """Crea gli embeddings e salvali in ChromaDB."""
    if os.path.exists(config.CHROMA_DIRECTORY):
        console.print("🗑️  Rimuovo database esistente...")
        shutil.rmtree(config.CHROMA_DIRECTORY)

    console.print(f"🧠 Modello embeddings: [cyan]{config.EMBEDDING_MODEL}[/cyan]")

    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )

    console.print(f"📦 Creazione vector store con {len(chunks)} chunk...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=config.CHROMA_DIRECTORY,
    )

    console.print(f"[green]✅ Database creato in '{config.CHROMA_DIRECTORY}/'[/green]")
    return vectorstore


def main():
    console.print("\n[bold]🐉 D&D RAG — Indicizzazione Markdown (MinerU)[/bold]\n")

    # Step 1: Carica i Markdown
    documents = load_markdown(config.PDF_DIRECTORY)
    console.print(f"\n📚 Totale sezioni caricate: [bold]{len(documents)}[/bold]")

    # Step 2: Spezza in chunk
    chunks = split_documents(documents)
    console.print(f"✂️  Totale chunk creati: [bold]{len(chunks)}[/bold]")

    # Step 3: Crea embeddings e salva
    console.print()
    create_vector_store(chunks)

    # Mostra un esempio di chunk
    console.print("\n[dim]--- Esempio di chunk ---[/dim]")
    console.print(f"[dim]{chunks[0].page_content[:300]}...[/dim]\n")


if __name__ == "__main__":
    main()