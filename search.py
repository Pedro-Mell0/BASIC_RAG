import json
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

model = SentenceTransformer("all-MiniLM-L6-v2")

#Carregar as embeddings salvas
def load_embeddings(path="data/embeddings.json"):
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    texts = [item["text"] for item in data]
    embeddings = np.array([item["embedding"] for item in data])

    return texts, embeddings

#Função de busca semântica (Transformar pergunta em embedding e comparar com as embeddings salvas)
def search(query, texts, embeddings, top_k=1):
    query_embedding = model.encode(query)

    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            "text": texts[idx],
            "score": float(similarities[idx])
        })

    return results


#Executar aplicação de busca

if __name__ == "__main__":
    texts, embeddings = load_embeddings()

    print("🔎 Busca semântica iniciada!")

    while True:
        query = input("\nPergunta (ou 'sair'): ")

        if query.lower() == "sair":
            print("Encerrando...")
            break

        results = search(query, texts, embeddings)

        for r in results:
            print(f"\nScore: {r['score']:.4f}")
            print(r["text"])