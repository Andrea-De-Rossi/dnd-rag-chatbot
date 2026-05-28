"""
benchmark.py — Confronta diversi modelli LLM sulle stesse domande

Lancia le stesse domande a tutti i modelli e salva:
- Le risposte di ognuno
- Il tempo di risposta
- Un report finale di confronto

Perfetto da mostrare al colloquio!
"""

import time
import json
import os
from datetime import datetime
from rich.console import Console
from rich.table import Table

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

import config

console = Console()

# === MODELLI DA CONFRONTARE ===
MODELS = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen/qwen3-32b",
]

# === DOMANDE DI TEST ===
QUESTIONS = [
    "Quali sono tutte e 12 le classi giocabili? Elencale.",
    "Come funziona l'attacco furtivo del ladro?",
    "Spiega le regole sulla concentrazione degli incantesimi.",
    "Cos'è il talento Sentinella e cosa fa?",
    "Come si calcola la classe armatura?",
    "Quali incantesimi di livello 1 può usare un chierico?",
    "Spiega la differenza tra tiro salvezza e prova di caratteristica.",
]


def load_retriever():
    """Carica il vector store per il retrieval."""
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )
    vectorstore = Chroma(
        persist_directory=config.CHROMA_DIRECTORY,
        embedding_function=embeddings,
    )
    return vectorstore


def ask_with_model(question, vectorstore, model_name):
    """Fa una domanda usando un modello specifico e misura il tempo."""
    
    # Retrieval
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.TOP_K_RESULTS},
    )
    source_docs = retriever.invoke(question)
    context = "\n\n---\n\n".join(doc.page_content for doc in source_docs)

    # Prompt
    prompt = PromptTemplate(
        template=config.SYSTEM_PROMPT,
        input_variables=["context", "question"],
    )
    formatted_prompt = prompt.format(context=context, question=question)

    # LLM
    llm = ChatGroq(
        model=model_name,
        api_key=config.GROQ_API_KEY,
        temperature=0.3,
    )

    start = time.time()
    answer = llm.invoke(formatted_prompt)
    elapsed = time.time() - start

    if hasattr(answer, 'content'):
        answer = answer.content

    return {
        "answer": answer,
        "time_seconds": round(elapsed, 2),
        "sources": len(source_docs),
    }


def main():
    console.print("\n[bold]🏆 D&D RAG — Benchmark Modelli[/bold]\n")
    console.print(f"Modelli: {len(MODELS)}")
    console.print(f"Domande: {len(QUESTIONS)}")
    console.print(f"Totale chiamate: {len(MODELS) * len(QUESTIONS)}\n")

    # Carica retriever
    console.print("⏳ Caricamento database...")
    vectorstore = load_retriever()
    console.print("[green]✅ Pronto![/green]\n")

    results = {}

    for model in MODELS:
        console.print(f"\n[bold cyan]📊 Test: {model}[/bold cyan]")
        results[model] = []

        for i, question in enumerate(QUESTIONS):
            console.print(f"  [{i+1}/{len(QUESTIONS)}] {question[:50]}...", end="")

            try:
                result = ask_with_model(question, vectorstore, model)
                results[model].append({
                    "question": question,
                    **result,
                })
                console.print(f" ✅ ({result['time_seconds']}s)")
            except Exception as e:
                console.print(f" ❌ {e}")
                results[model].append({
                    "question": question,
                    "answer": f"ERRORE: {e}",
                    "time_seconds": 0,
                    "sources": 0,
                })

            # Rate limit: pausa tra le chiamate
            time.sleep(3)

        # Pausa extra tra modelli
        time.sleep(5)

    # === REPORT ===
    console.print("\n\n" + "=" * 70)
    console.print("[bold]📊 RISULTATI DEL BENCHMARK[/bold]")
    console.print("=" * 70)

    # Tabella tempi medi
    table = Table(title="Tempo Medio di Risposta")
    table.add_column("Modello", style="cyan")
    table.add_column("Tempo Medio", style="green")
    table.add_column("Più Veloce", style="yellow")
    table.add_column("Più Lento", style="red")

    for model in MODELS:
        times = [r["time_seconds"] for r in results[model] if r["time_seconds"] > 0]
        if times:
            avg = sum(times) / len(times)
            table.add_row(
                model.split("/")[-1],
                f"{avg:.2f}s",
                f"{min(times):.2f}s",
                f"{max(times):.2f}s",
            )

    console.print(table)

    # Mostra risposte per ogni domanda
    for i, question in enumerate(QUESTIONS):
        console.print(f"\n[bold]❓ {question}[/bold]\n")
        for model in MODELS:
            r = results[model][i]
            model_short = model.split("/")[-1]
            answer_preview = r["answer"][:200].replace("\n", " ")
            console.print(f"  [cyan]{model_short}[/cyan] ({r['time_seconds']}s):")
            console.print(f"  {answer_preview}...")
            console.print()

    # Salva risultati completi
    output = {
        "timestamp": datetime.now().isoformat(),
        "models": MODELS,
        "questions": QUESTIONS,
        "results": results,
    }

    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    console.print(f"\n[green]✅ Risultati salvati in benchmark_results.json[/green]")
    console.print("Puoi aprirlo per vedere le risposte complete di ogni modello.\n")


if __name__ == "__main__":
    main()