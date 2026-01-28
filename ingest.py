from pathlib import Path
import json
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 300
OVERLAP = 50

model = SentenceTransformer('all-MiniLM-L6-v2')



# ========= Caso usuário deseje contexto:  ============


INTENTS = {
    "show_context": [
        "mostre o contexto",
        "qual a fonte da resposta",
        "de onde veio essa informação",
        "mostre o texto original",
        "qual documento foi usado"
    ]
}

def load_document(path):
    return Path(path).read_text(encoding='utf-8')   

def create_chunks(text):
    # Cada linha = um chunk (simples e didático)
    chunks = []
    for line in text.split("\n"):
        line = line.strip()
        if line:
            chunks.append(line)
    return chunks

def embed_chunks(texts):
    return model.encode(texts)

if __name__ == "__main__":
    # =========================
    # INGESTÃO DOS DOCUMENTOS
    # =========================
    text = load_document("data/docs.txt")
    chunks = create_chunks(text)
    embeddings = embed_chunks(chunks)

    data = []
    for chunk, embedding in zip(chunks, embeddings):
        data.append({
            "text": chunk,
            "embedding": embedding.tolist()
        })

    Path("data/embeddings.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"{len(data)} chunks de documentos processados.")

    # =========================
    # INGESTÃO DAS INTENÇÕES
    # =========================
    intent_data = {}

    for intent_name, phrases in INTENTS.items():
        intent_embeddings = embed_chunks(phrases)
        intent_data[intent_name] = {
            "phrases": phrases,
            "embeddings": [e.tolist() for e in intent_embeddings]
        }

    Path("data/intents.json").write_text(
        json.dumps(intent_data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("Intenções processadas e salvas.")
