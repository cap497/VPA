import os
import sys
import time
import subprocess
import pickle
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# === CONFIG ===
BM25_DIR = "bm25_index"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "local-model"
SIMILARITY_THRESHOLD = 0.8

# LM Studio model name you want to ensure is loaded
TARGET_MODEL_NAME = "meta-llama-3-8b-instruct"

# === OpenAI-compatible Client (LM Studio) ===
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

# --------------------------------------------------------------------------------
# LM Studio CLI integration
# --------------------------------------------------------------------------------

def run_lms_command(args):
    try:
        result = subprocess.run(
            ["lms"] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running 'lms {' '.join(args)}':", e.stderr)
        return None

def get_downloaded_models():
    output = run_lms_command(["ls"])
    if not output:
        return []

    models = []
    lines = output.splitlines()
    in_llm_section = False

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("LLMs (Large Language Models"):
            in_llm_section = True
            continue
        if in_llm_section and ("Embedding Models" in line):
            break
        if in_llm_section:
            if line.startswith("PARAMS") or "ARCHITECTURE" in line or "SIZE" in line:
                continue
            parts = line.split()
            if parts:
                models.append(parts[0])
    return models

def get_loaded_models():
    output = run_lms_command(["ps"])
    if not output:
        return []

    models = set()
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("Identifier:"):
            identifier = line[len("Identifier:"):].strip()
            base_name = identifier.split(":")[0].strip()
            models.add(base_name)
    return list(models)

def lms_load_model(model_name):
    print(f"⚡ Loading model into LM Studio: {model_name}")
    run_lms_command(["load", model_name])

def ensure_model_loaded():
    print("🧭 Checking LM Studio for target model...")

    # Check if downloaded
    downloaded = get_downloaded_models()
    if not any(TARGET_MODEL_NAME in m for m in downloaded):
        print(f"❌ Model '{TARGET_MODEL_NAME}' is NOT downloaded in LM Studio.")
        print("➡️ Continuing without forcing load. User must download manually.")
        return

    print(f"✅ Model '{TARGET_MODEL_NAME}' is downloaded in LM Studio.")

    # Check if already loaded
    loaded = get_loaded_models()
    if any(TARGET_MODEL_NAME in m for m in loaded):
        print(f"✅ Model '{TARGET_MODEL_NAME}' is already loaded in LM Studio RAM.")
        return

    # Load model if not yet loaded
    lms_load_model(TARGET_MODEL_NAME)

    # Wait for LM Studio to finish loading
    for attempt in range(5):
        print(f"⌛ Waiting for load... (attempt {attempt+1}/5)")
        time.sleep(3)
        loaded = get_loaded_models()
        if any(TARGET_MODEL_NAME in m for m in loaded):
            print(f"✅ Model '{TARGET_MODEL_NAME}' is now loaded and ready!")
            return

    print(f"❌ Could not load model '{TARGET_MODEL_NAME}' after multiple attempts. Aborting.")
    sys.exit(1)

def unload_model():
    print(f"⚡ Unloading model from LM Studio: {TARGET_MODEL_NAME}")
    run_lms_command(["unload", TARGET_MODEL_NAME])

# --------------------------------------------------------------------------------
# BM25 & Embedder Loading
# --------------------------------------------------------------------------------

def load_bm25_index():
    with open(os.path.join(BM25_DIR, "bm25.pkl"), "rb") as f:
        data = pickle.load(f)
    print("✅ BM25 index loaded from disk.")
    return data["bm25"], data["sections"], data["texts"]

def load_embedder():
    model = SentenceTransformer(EMBED_MODEL)
    print("✅ Embedding model loaded into RAM.")
    return model

# --------------------------------------------------------------------------------
# Utility
# --------------------------------------------------------------------------------

def estimate_tokens(text):
    return len(text.split())

def generate_llm_answer(context, query):
    prompt = f"""
Você é um assistente automotivo. Responda em português. Use SOMENTE o contexto abaixo.

### Contexto
{context}

### Pergunta
{query}

### Resposta
"""
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512
    )
    return completion.choices[0].message.content.strip()  # type: ignore

# --------------------------------------------------------------------------------
# Core Retrieval-Augmented Generation Pipeline
# --------------------------------------------------------------------------------

def run_rag_pipeline(bm25, sections, embedder, user_query, top_n_bm25=10, top_k_final=5, max_context_tokens=500):
    print("\n==========================")
    print(f"❓ Pergunta recebida: {user_query}")

    # 1️⃣ BM25 retrieval
    query_tokens = user_query.lower().split()
    scores = bm25.get_scores(query_tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top_bm25_indices = [i for i, score in ranked[:top_n_bm25]]

    candidate_sections = [sections[i] for i in top_bm25_indices]
    candidate_texts = [f"{s['title']}\n{s['body']}" for s in candidate_sections]

    if not candidate_texts:
        print("⚠️ Nenhum resultado relevante no BM25.")
        return "Não encontrei nada no manual para essa pergunta."

    # 2️⃣ Embedding & Similarity
    query_emb = embedder.encode(user_query, normalize_embeddings=True).reshape(1, -1)
    candidate_embs = embedder.encode(candidate_texts, normalize_embeddings=True)
    sims = np.dot(candidate_embs, query_emb.T).flatten()
    ranked_candidates = sorted(zip(sims, candidate_sections, candidate_texts), key=lambda x: x[0], reverse=True)

    # 3️⃣ Select top paragraphs
    selected_chunks = []
    total_tokens = 0
    for sim, section, chunk_text in ranked_candidates:
        if sim > SIMILARITY_THRESHOLD:
            continue
        tokens = estimate_tokens(chunk_text)
        if total_tokens + tokens > max_context_tokens:
            continue
        selected_chunks.append(chunk_text)
        total_tokens += tokens
        if len(selected_chunks) >= top_k_final:
            break

    if not selected_chunks:
        print("⚠️ Nenhum chunk selecionado após filtragem.")
        return "Não encontrei informações relevantes no manual."

    print("✅ Chunks selecionados como contexto:")
    for idx, chunk in enumerate(selected_chunks, 1):
        print(f"  {idx}. {chunk}...\n")

    # 4️⃣ Generate final answer
    context = "\n\n".join(selected_chunks)
    answer = generate_llm_answer(context, user_query)

    print(f"✅ Resposta gerada pelo LLM:\n\n {answer}")
    print("==========================\n")
    return answer
