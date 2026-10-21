import uuid
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import torch

MODEL_NAME = "microsoft/harrier-oss-v1-270m"
DOCS_DIR = Path("docs")
COLLECTION = "docs"
EMBEDDING_DIM = 640  # harrier-oss-v1-270m output dimension
CHUNK_SIZE = 150
CHUNK_OVERLAP = 15


def chunk_text(text: str, tokenizer) -> list[str]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(token_ids):
        end = min(start + CHUNK_SIZE, len(token_ids))
        chunks.append(tokenizer.decode(token_ids[start:end]))
        if end == len(token_ids):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def embed(text: str, model, tokenizer, device: str) -> list[float]:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=CHUNK_SIZE + 10,
        padding=False,
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # Harrier: last-token pooling + L2 normalization
    last_hidden = outputs.last_hidden_state[:, -1, :]
    normed = torch.nn.functional.normalize(last_hidden, p=2, dim=-1)
    return normed.cpu().float().numpy()[0].tolist()


def main():
    print(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    print(f"Model loaded on {device}\n")

    client = QdrantClient(host="localhost", port=6333)

    # Recreate collection so re-runs don't duplicate points
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    print(f"Qdrant collection '{COLLECTION}' ready\n")

    points: list[PointStruct] = []
    for doc_path in sorted(DOCS_DIR.glob("*.md")):
        text = doc_path.read_text(encoding="utf-8")
        chunks = chunk_text(text, tokenizer)
        print(f"{doc_path.name}  →  {len(chunks)} chunk(s)")

        for i, chunk in enumerate(chunks):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=embed(chunk, model, tokenizer, device),
                payload={"source": doc_path.name, "chunk_index": i, "text": chunk},
            ))

    client.upsert(collection_name=COLLECTION, points=points)
    print(f"\nUpserted {len(points)} points into Qdrant collection '{COLLECTION}'")


if __name__ == "__main__":
    main()
