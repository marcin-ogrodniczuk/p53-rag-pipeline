"""Command-line interface for querying the p53 RAG pipeline.

Usage:
    python cli.py "How does p53 regulate apoptosis?"
    python cli.py            # interactive mode
"""

import sys

import config
import rag


def ask(question: str):
    print(f"\nQ: {question}\n")
    stream, hits = rag.answer(question, stream=True)
    print("A: ", end="", flush=True)
    for token in stream:
        print(token, end="", flush=True)
    print("\n\nSources:")
    seen = set()
    for h in hits:
        m = h["meta"]
        if m["pmid"] in seen:
            continue
        seen.add(m["pmid"])
        print(f"  [{m['pmid']}] {m['title']} — {m['url']}")
    print()


def main():
    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]))
        return

    print(f"p53 RAG — chat model: {config.CHAT_MODEL}. Type 'exit' to quit.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit", ""}:
            break
        ask(q)


if __name__ == "__main__":
    main()
