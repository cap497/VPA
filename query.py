import os
import pickle
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from openai import OpenAI

BM25_DIR = "bm25_index"

EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "local-model"

DEFAULT_TOP_N_BM25 = 10
DEFAULT_TOP_K_FINAL = 5
DEFAULT_MAX_CONTEXT_TOKENS = 500
SIMILARITY_THRESHOLD = 0.8

# For merging short subheadings like "Atenção" or "Nota" with their content
HEADING_KEYWORDS = {"atenção", "nota", "notas", "cuidado"}

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

def estimate_tokens(text):
    return len(text.split())

def load_bm25():
    with open(os.path.join(BM25_DIR, "bm25.pkl"), "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["sections"], data["texts"]

def is_subheading_line(line):
    """
    Check if line is a known subheading keyword.
    """
    normalized = line.lower().strip(":").strip()
    return normalized in HEADING_KEYWORDS

def split_body_into_paragraphs(body):
    """
    Split on double newlines but merge known subheading lines with following paragraphs.
    """
    paras = [p.strip() for p in body.split('\n\n') if p.strip()]

    merged = []
    i = 0
    while i < len(paras):
        if i + 1 < len(paras) and is_subheading_line(paras[i]):
            # Merge this subheading with next paragraph
            merged.append(paras[i] + "\n\n" + paras[i+1])
            i += 2
        else:
            merged.append(paras[i])
            i += 1
    return merged

def rechunk_with_headings(selected_sections):
    """
    For each selected (section, chunk_text), split body into paragraphs,
    and attach the real section['title'] as heading.
    """
    new_chunks = []
    for section, chunk_text in selected_sections:
        if '\n' not in chunk_text:
            continue
        # Remove the first line (section title) from chunk_text
        _, body = chunk_text.split('\n', 1)
        body = body.strip()

        paragraphs = split_body_into_paragraphs(body)
        if not paragraphs:
            continue

        heading = section['title']
        for para in paragraphs:
            new_chunks.append(f"{heading}\n\n{para}")
    return new_chunks

def generate_answer(context, query):
    prompt = f"""
Você é um assistente automotivo. Responda em português. Use SOMENTE o contexto abaixo.

### Contexto
{context}

### Pergunta
{query}

### Resposta
"""
    try:
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=512
        )
        return completion.choices[0].message.content.strip()  # type: ignore
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        return "[ERROR]"

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str)
    parser.add_argument("--top_n_bm25", type=int, default=DEFAULT_TOP_N_BM25)
    parser.add_argument("--top_k_final", type=int, default=DEFAULT_TOP_K_FINAL)
    parser.add_argument("--max_context_tokens", type=int, default=DEFAULT_MAX_CONTEXT_TOKENS)
    args = parser.parse_args()

    print("🤖 Loading embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL)

    print("📦 Loading BM25 index...")
    bm25, sections, texts = load_bm25()

    print("🎯 Running BM25 search...")
    query_tokens = args.query.lower().split()
    scores = bm25.get_scores(query_tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top_bm25_indices = [i for i, score in ranked[:args.top_n_bm25]]

    candidate_sections = [sections[i] for i in top_bm25_indices]
    candidate_texts = [f"{s['title']}\n{s['body']}" for s in candidate_sections]

    print(f"✅ Selected {len(candidate_sections)} candidates from BM25.")

    print("🤖 Embedding query and candidates...")
    query_emb = embedder.encode(args.query, normalize_embeddings=True).reshape(1, -1)
    candidate_embs = embedder.encode(candidate_texts, normalize_embeddings=True)

    print("🔎 Computing semantic similarities...")
    sims = np.dot(candidate_embs, query_emb.T).flatten()

    ranked_candidates = sorted(zip(sims, candidate_sections, candidate_texts), key=lambda x: x[0], reverse=True)

    print("\n✅ Selecting top-k within token budget (first pass)...")
    selected_chunks = []
    total_tokens = 0
    for i, (similarity, section, chunk_text) in enumerate(ranked_candidates):
        if similarity > SIMILARITY_THRESHOLD:
            print(f"❌ Skipping Chunk {i+1} - low similarity ({similarity:.4f})")
            continue
        chunk_tokens = estimate_tokens(chunk_text)
        if total_tokens + chunk_tokens > args.max_context_tokens:
            print(f"⛔ Skipping - would exceed budget ({total_tokens + chunk_tokens} > {args.max_context_tokens})")
            continue

        selected_chunks.append((section, chunk_text))
        total_tokens += chunk_tokens
        print(f"✅ Included Chunk {i+1} ({chunk_tokens} tokens, cumulative {total_tokens})")
        print(f"   - Chapter: {section['chapter']}")
        print(f"   - Title: {section['title']}")
        print(f"   - Similarity: {similarity:.4f}\n")

        if len(selected_chunks) >= args.top_k_final:
            break

    if not selected_chunks:
        print("❌ No chunks fit within the token budget!")
        return

    print("\n🪄 Rechunking selected chunks into paragraphs with headings...")
    selected_chunks = rechunk_with_headings(selected_chunks)
    print(f"✅ Rechunked into {len(selected_chunks)} paragraph-level chunks.")

    print("🤖 Re-embedding rechunked paragraphs...")
    para_embs = embedder.encode(selected_chunks, normalize_embeddings=True)
    para_sims = np.dot(para_embs, query_emb.T).flatten()

    print("🔎 Re-ranking paragraphs by semantic similarity...")
    ranked_paragraphs = sorted(zip(para_sims, selected_chunks), key=lambda x: x[0], reverse=True)

    print("\n✅ Selecting final paragraphs within token budget (second pass)...")
    final_chunks = []
    total_tokens = 0
    for i, (similarity, para) in enumerate(ranked_paragraphs):
        if similarity > SIMILARITY_THRESHOLD:
            print(f"❌ Skipping Paragraph {i+1} - low similarity ({similarity:.4f})")
            continue
        para_tokens = estimate_tokens(para)
        if total_tokens + para_tokens > args.max_context_tokens:
            print(f"⛔ Skipping - would exceed budget ({total_tokens + para_tokens} > {args.max_context_tokens})")
            continue

        final_chunks.append(para)
        total_tokens += para_tokens
        print(f"✅ Included Paragraph {i+1} ({para_tokens} tokens, cumulative {total_tokens})")
        print(f"   - Similarity: {similarity:.4f}\n")

        if len(final_chunks) >= args.top_k_final:
            break

    if not final_chunks:
        print("❌ No paragraphs fit within the token budget after reranking!")
        return

    print("\n🧩 Final Selected Paragraphs:")
    for idx, chunk in enumerate(final_chunks):
        print(f"\n{'='*80}")
        print(f"✅ SELECTED PARAGRAPH {idx+1}")
        print(f"{'='*80}")
        print(chunk)
        print(f"\n[Approx. {estimate_tokens(chunk)} tokens]\n")

    print(f"\n🧠 Generating answer with ~{total_tokens} tokens of context...")
    context = "\n\n".join(final_chunks)
    answer = generate_answer(context, args.query)
    print("\n💬 Answer:\n", answer)

if __name__ == "__main__":
    main()
