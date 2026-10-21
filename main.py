import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient
import anthropic
import torch

MODEL_NAME = "microsoft/harrier-oss-v1-270m"
COLLECTION = "docs"
TOP_K = 3

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the context "
    "provided below. If the context does not contain enough information, say so. "
    "Be concise and accurate."
)

DIVIDER = "─" * 60


def load_api_key() -> str:
    load_dotenv()
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        env_file = Path(".env")
        if env_file.exists():
            key = env_file.read_text(encoding="utf-8").strip().splitlines()[0].strip()
    if not key:
        sys.exit("ANTHROPIC_API_KEY not found. Set it in .env or as an env var.")
    return key


def load_embedding_model():
    print("Loading embedding model …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    print(f"Embedding model ready  ({device})\n")
    return tokenizer, model, device


def embed_query(text: str, model, tokenizer, device: str) -> list[float]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden = outputs.last_hidden_state[:, -1, :]
    normed = torch.nn.functional.normalize(last_hidden, p=2, dim=-1)
    return normed.cpu().float().numpy()[0].tolist()


def retrieve_chunks(query_vector: list[float], client: QdrantClient) -> list[dict]:
    results = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=TOP_K,
        with_payload=True,
    ).points
    return [
        {"score": r.score, "source": r.payload["source"], "text": r.payload["text"]}
        for r in results
    ]


def build_prompt(question: str, chunks: list[dict]) -> str:
    context_block = "\n\n".join(
        f"[Chunk {i+1} — {c['source']} (score: {c['score']:.3f})]\n{c['text']}"
        for i, c in enumerate(chunks)
    )
    return f"Context:\n{context_block}\n\nQuestion: {question}"


def main():
    api_key = load_api_key()
    tokenizer, model, device = load_embedding_model()

    qdrant = QdrantClient(host="localhost", port=6333)
    claude = anthropic.Anthropic(api_key=api_key)

    print("RAG Chatbot ready. Type your question (or 'quit' to exit).\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not question or question.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        # ── Step 1: embed the question ──────────────────────────────────────
        print(f"\n{DIVIDER}")
        print("STEP 1 — Embedding query …")
        query_vec = embed_query(question, model, tokenizer, device)
        print(f"Query vector: dim={len(query_vec)}  norm≈{sum(v**2 for v in query_vec[:8]):.3f} (first 8 dims)")

        # ── Step 2: retrieve chunks ─────────────────────────────────────────
        print(f"\n{DIVIDER}")
        print(f"STEP 2 — Retrieving top {TOP_K} chunks from Qdrant …")
        chunks = retrieve_chunks(query_vec, qdrant)
        for i, c in enumerate(chunks, 1):
            print(f"\n  Chunk {i}  [{c['source']}]  score={c['score']:.4f}")
            print(f"  {c['text'][:200].replace(chr(10), ' ')} …")

        # ── Step 3: bind prompt ─────────────────────────────────────────────
        print(f"\n{DIVIDER}")
        print("STEP 3 — Building prompt …")
        user_prompt = build_prompt(question, chunks)
        preview = user_prompt[:400].replace("\n", "\n  ")
        print(f"\n  [System] {SYSTEM_PROMPT[:80]} …")
        print(f"\n  [User]\n  {preview} …")

        # ── Step 4: stream answer ───────────────────────────────────────────
        print(f"\n{DIVIDER}")
        print("STEP 4 — Streaming answer from claude-sonnet-4-6 …\n")
        print("Assistant: ", end="", flush=True)

        with claude.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_prompt}],
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
            final = stream.get_final_message()

        usage = final.usage
        print(f"\n\n  [tokens — in:{usage.input_tokens}  out:{usage.output_tokens}"
              f"  cache_read:{getattr(usage, 'cache_read_input_tokens', 0)}]\n")


if __name__ == "__main__":
    main()
