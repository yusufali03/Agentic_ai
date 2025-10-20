"""
main.py — Terminal entry point.

Commands:
  ingest <PDF> [--collection ...] [--store-dir ...] [--chunk-size ...] [--chunk-overlap ...]
  ask "your question" [--collection ...] [--store-dir ...] [--source FILENAME] [-k 5] [--provider ollama|openai] [--model ...]

Examples:
  python main.py ingest data/ModuleHandbook.pdf
  python main.py ask "What are the assessment weights?" --source ModuleHandbook.pdf --provider ollama --model "llama3.1:8b"
"""


import argparse
from app.ingest_pdf import ingest_pdf
from app.qa import answer_question
import shutil
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

DEFAULT_COLLECTION = os.getenv("CHROMA_COLLECTION", "handbook")
DEFAULT_STORE = os.getenv("CHROMA_STORE", "chroma_store/handbook")

def cmd_chat(args):
    """
        Interactive REPL:
          - You type questions.
          - Type \q (or :q, quit, exit, Ctrl+C) to leave.
          - Provider/model come from .env automatically.
        """
    from app.qa import QASession
    session = QASession(
        collection_name=args.collection,
        persist_directory=args.store_dir,
        source_filter=args.source,  # None by default; searches all ingested docs
        k=args.k
        # provider/model omitted -> read from .env
    )
    print("RAG chat ready. Type your question and press enter.")
    print("Type \\q to quit. \n")

    while True:
        try:
            q = input("Ask >> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
        if not q:
            continue
        if q in {"\\q", ":q", "quit", "exit"}:
            print("Bye.")
            break

        answer = session.ask(q)
        print("\n" + answer + "\n")

def cmd_ask(args):
    ans = answer_question(
        question=args.question,
        collection_name=args.collection,
        persist_directory= args.store_dir,
        source_filter=args.source,
        k = args.k,
        provider = args.provider,
        model=args.model,
    )
    print(ans)

def cmd_reset(args):
    p = Path(args.store_dir)
    if p.exists():
        shutil.rmtree(p)
        print(f"Deleted store: {p}")
    else:
        print("Nothing to delete")

def build_parser():
    p = argparse.ArgumentParser(description="Mini RAG (Chroma + Langchain, PDF).")
    sub = p.add_subparsers(dest="command", required=True)

    pc = sub.add_parser("chat", help="Interactive Q&A loop (type \\q to quit)")
    pc.add_argument("--collection", default=DEFAULT_COLLECTION)
    pc.add_argument("--store-dir", default=DEFAULT_STORE)
    pc.add_argument("--source", default=None, help="Optional filename filter, e.g., ModuleHandbook.pdf")
    pc.add_argument("-k", type=int, default=int(os.getenv("RAG_K", "5")))
    pc.set_defaults(func=cmd_chat)


    # pi = sub.add_parser("ingest", help="Ingest a PDF into Chroma")
    # pi.add_argument("pdf", help="Path to PDF in ./data")
    # pi.add_argument("--collection", default=DEFAULT_COLLECTION)
    # pi.add_argument("--store-dir", default="chroma_store/handbook")
    # pi.add_argument("--chunk-size", type=int, default=1000)
    # pi.add_argument("--chunk-overlap", type=int, default=150)
    # pi.set_defaults(func=cmd_ingest)

    pa = sub.add_parser("ask", help="Ask a question grounded in the ingested PDF")
    pa.add_argument("question", help="Your question in quotes")
    pa.add_argument("--collection", default=DEFAULT_COLLECTION)
    pa.add_argument("--store-dir", default=DEFAULT_STORE)
    pa.add_argument("--source", default=None, help="Optional filename filter, e.g., ModuleHandbook.pdf")
    pa.add_argument("-k", type=int, default=5)
    pa.add_argument("--provider", default="ollama", choices=["ollama", "openai"])
    pa.add_argument("--model", default=None)
    pa.set_defaults(func=cmd_ask)

    pr = sub.add_parser("reset", help="Delete the Chroma store directory")
    pr.add_argument("--store-dir", default="chroma_store/handbook")
    pr.set_defaults(func=cmd_reset)

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()