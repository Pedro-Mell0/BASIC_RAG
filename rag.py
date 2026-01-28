from search import search, load_embeddings
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os
from openai import OpenAI

# ================== CONFIGURAÇÕES ==================

USE_LOCAL_LLM = False  # False = OpenAI | True = Ollama

MIN_RAG_SCORE = 0.60
MIN_INTENT_SCORE = 0.65

# ==================================================

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = SentenceTransformer("all-MiniLM-L6-v2")


# ================== LLM ROUTER ==================

def call_llm(prompt):
    """
    Roteador central de LLM
    """
    if USE_LOCAL_LLM:
        return call_llm_local(prompt)
    return call_llm_openai(prompt)


# ================== PLANO A - OPENAI ==================

def call_llm_openai(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Você responde apenas em português brasileiro."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content


# ================== PLANO B - LLM LOCAL (OLLAMA) ==================
# 1° Descomente o código abaixo
# 2️° Instale requests
# 3️° Altere USE_LOCAL_LLM = True
"""
import requests

def call_llm_local(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        }
    )
    data = response.json()
    return data["response"]
"""

# Fallback seguro caso USE_LOCAL_LLM=True sem ativar Ollama
def call_llm_local(prompt):
    raise RuntimeError(
        "LLM local não está ativa. "
        "Descomente o código do Ollama e configure corretamente."
    )


# ================== PROMPTS ==================

def build_rag_prompt(context, question):
    return f"""
Você é um assistente que responde APENAS em português brasileiro.

Responda SOMENTE com base no contexto abaixo.
Não utilize nenhum conhecimento externo.

Contexto:
{context}

Pergunta:
{question}

Resposta:
"""


def build_general_prompt(question):
    return f"""
Você é um assistente em português brasileiro.

Responda a pergunta abaixo usando seu conhecimento geral.

Pergunta:
{question}

Resposta:
"""


# ================== INTENTS (Verificação sêmantica caso o usuário queira o contexto/fonte da resposta) ==================

def load_intents():
    with open("data/intents.json", encoding="utf-8") as f:
        return json.load(f)


def detect_intent(question, intents):
    question_emb = model.encode([question])[0]

    best_intent = None
    best_score = 0

    for intent_name, data in intents.items():
        for emb in data["embeddings"]:
            emb = np.array(emb)
            score = np.dot(question_emb, emb) / (
                np.linalg.norm(question_emb) * np.linalg.norm(emb)
            )

            if score > best_score:
                best_score = score
                best_intent = intent_name

    if best_score >= MIN_INTENT_SCORE:
        return best_intent, best_score

    return None, best_score


# ================== MAIN ==================

if __name__ == "__main__":
    texts, embeddings = load_embeddings()
    intents = load_intents()

    while True:
        question = input("\nPergunta (ou 'sair'): ")
        if question.lower() == "sair":
            break

        # 1️⃣ Detectar intenção
        intent, intent_score = detect_intent(question, intents)

        # 2️⃣ Recuperação RAG
        results = search(question, texts, embeddings, top_k=1)

        # 3️⃣ RAG fraco → conhecimento geral da LLM
        if not results or results[0]["score"] < MIN_RAG_SCORE:
            prompt = build_general_prompt(question)
            answer = call_llm(prompt)

            print("\n🧠 Resposta:")
            print(answer)
            continue

        context = results[0]["text"]

        # 4️⃣ Usuário pediu explicitamente o contexto
        if intent == "show_context":
            print("\n📄 Contexto utilizado:")
            print(context)
            continue

        # 5️⃣ Prompt RAG
        prompt = build_rag_prompt(context, question)

        # 6️⃣ Chamada da LLM
        answer = call_llm(prompt)

        print("\n🧠 Resposta:")
        print(answer)
