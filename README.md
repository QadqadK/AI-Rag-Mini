# Documentation RAG Assistant

A small command-line retrieval-augmented generation (RAG) chatbot for answering questions from the Markdown files in `docs/`. It finds relevant document chunks with a local embedding model, retrieves them from Qdrant, then asks Claude to answer using that retrieved context. This is developed during my internship in Xsolla for learning concepts of AI Rag, which uses the Xsolla .md files. You can try it by inserting .md files in /docs and perform the steps below. 

## How it works

1. `embed.py` reads each `docs/*.md` file, splits its text into 150-token chunks with 15-token overlap, and embeds each chunk.
2. The vectors and their source text are stored in a local Qdrant collection named `docs`.
3. `main.py` embeds each question, retrieves the three closest chunks, and sends those chunks to Claude with instructions to answer only from the provided context.
4. Claude's response is streamed to the terminal, along with token usage.

## Step-by-step flow

### Building the knowledge base (`embed.py`)

1. Load the `microsoft/harrier-oss-v1-270m` tokenizer and model from Hugging Face. Use a CUDA device when available; otherwise use the CPU.
2. Connect to Qdrant at `localhost:6333`.
3. Delete the existing `docs` collection, if present, and create it again with 640-dimensional vectors and cosine distance.
4. Read each top-level Markdown file in `docs/`, in filename order.
5. Tokenize each file and split it into chunks of up to 150 tokens, advancing 135 tokens at a time to provide 15 tokens of overlap.
6. Embed each chunk using the model's last-token hidden state, then L2-normalize the vector.
7. Write the vectors to Qdrant with each chunk's source filename, chunk index, and text.

### Answering a question (`main.py`)

1. Load `ANTHROPIC_API_KEY` from the environment or project `.env` file.
2. Load the same embedding tokenizer and model, then connect to Qdrant at `localhost:6333`.
3. Read a question from the terminal and embed it using last-token pooling and L2 normalization.
4. Query the `docs` collection for the three closest vectors, including their stored text and source metadata.
5. Build a prompt containing the retrieved chunks and the question. The system instructions tell Claude to use only that context and to say when it is insufficient.
6. Send the prompt to `claude-sonnet-4-6` through the Anthropic API and stream the answer to the terminal.
7. Print input, output, and cache-read token counts when available. Continue prompting until the user enters `quit`, `exit`, or `q`.

## Models and storage

- **Embedding model:** [`microsoft/harrier-oss-v1-270m`](https://huggingface.co/microsoft/harrier-oss-v1-270m), loaded through Hugging Face Transformers. The code uses last-token pooling and L2 normalization; vectors have 640 dimensions. It runs on CUDA when available and otherwise on CPU.
- **Vector database:** Qdrant, expected at `localhost:6333`. The `docs` collection uses cosine distance and stores each chunk's source filename, chunk index, and text as payload.
- **Answer model:** `claude-sonnet-4-6`, called through the Anthropic API. The application needs an Anthropic API key.

## Requirements

- Python 3.11 or newer
- Qdrant running locally on port 6333
- An Anthropic API key
- Internet access the first time the embedding model is loaded, so Transformers can download it from Hugging Face

## Setup

Install [uv](https://docs.astral.sh/uv/) if needed, then install the project dependencies:

```powershell
uv sync
```

Start Qdrant locally. For example, with Docker:

```powershell
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v "${PWD}/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

Set your API key in the process environment or in a `.env` file in the project root:

```dotenv
ANTHROPIC_API_KEY=your-key-here
```

Keep `.env` private and do not commit API keys.

## Index the documentation

With Qdrant running and the Markdown files in `docs/` in place, build the vector index:

```powershell
uv run python embed.py
```

**Running this command deletes and recreates the existing Qdrant `docs` collection.** Re-run it after changing the source Markdown files to rebuild the index.

## Ask questions

```powershell
uv run python main.py
```

Type a question at the prompt. Enter `quit`, `exit`, or `q` to finish.

## Project files

- `embed.py` — chunks and embeds source Markdown, then writes vectors to Qdrant.
- `main.py` — interactive retrieval and Claude response loop.
- `docs/` — Markdown source documents that are indexed.
- `pyproject.toml` / `uv.lock` — Python project dependencies and locked versions.

## Current limitations

- Only top-level `*.md` files under `docs/` are indexed.
- Qdrant host, port, collection, and retrieval count are currently set in the Python source.
- The answer is grounded only in the three retrieved chunks; if they do not contain enough information, the system prompt instructs Claude to say so.
