import os, json, time, re, threading
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

from rank_bm25 import BM25Okapi
import faiss
import numpy as np

# Embeddings
from sentence_transformers import SentenceTransformer

# ---------------- Config ----------------
EXTRACT_ROOT = os.environ.get("EXTRACT_ROOT", os.path.abspath("./assets_out"))
INDEX_ROOT = os.environ.get("INDEX_ROOT", os.path.abspath("./indices"))
EMB_MODEL_NAME = os.environ.get("EMB_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# LM Studio (OpenAI-compatible) — if not available, fallback to extractive answer
LMSTUDIO_BASE = os.environ.get("OPENAI_API_BASE", "http://localhost:1234/v1")
LMSTUDIO_MODEL = os.environ.get("TARGET_MODEL_NAME", "meta-llama-3-8b-instruct")
LMSTUDIO_API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")

IDLE_UNLOAD_SECS = int(os.environ.get("IDLE_UNLOAD_SECS", "1800"))  # 30min
_last_use_ts = time.time()
_idle_thread_started = False

# ---------------- Asset loading ----------------
def load_assets_index(extract_root: str = EXTRACT_ROOT) -> Dict[str, Any]:
    root = Path(extract_root)
    manifest_path = root / "manifest.jsonl"
    text_dir = root / "text"

    text_blocks: List[Dict[str, Any]] = []
    if text_dir.exists():
        for jf in sorted(text_dir.glob("page_*.jsonl")):
            try:
                page_num = int(jf.stem.split("_")[1])
            except Exception:
                continue
            with jf.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if rec.get("type") == "text" and rec.get("text"):
                        text_blocks.append({"page": page_num, "text": rec["text"]})

    page_images: Dict[int, List[str]] = {}
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("type") in ("table_image", "image"):
                    page = rec.get("page")
                    rel = (rec.get("path") or "").replace("\\", "/")
                    if not page or not rel:
                        continue
                    page_images.setdefault(page, []).append(rel)

    for p, lst in list(page_images.items()):
        table_first = [x for x in lst if "table_images" in x]
        others = [x for x in lst if x not in table_first]
        page_images[p] = table_first + others

    return {"text_blocks": text_blocks, "page_images": page_images, "extract_root": str(root)}

# ---------------- Token estimator ----------------
_token_re = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_-]+", re.UNICODE)
def estimate_tokens(s: str) -> int:
    # Heuristic: ~ 1 token per 4 chars, with floor by words
    if not s: return 0
    words = _token_re.findall(s)
    est = max(len(words), int(len(s) / 4))
    return est

def _tokenize(s: str) -> List[str]:
    return [t.lower() for t in _token_re.findall(s or "")]

# ---------------- BM25 index ----------------
def load_bm25_index(index_root: str = INDEX_ROOT) -> Tuple[BM25Okapi, List[Dict[str, Any]], List[List[str]]]:
    meta_path = Path(index_root) / "bm25_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"BM25 meta not found at {meta_path}. Run start.py to build indices.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    docs_meta = meta["docs_meta"]  # list of {id, page, text}
    corpus_tokens = [m["tokens"] for m in docs_meta]
    bm25 = BM25Okapi(corpus_tokens)
    return bm25, docs_meta, corpus_tokens

# ---------------- FAISS index ----------------
def load_faiss_index(index_root: str = INDEX_ROOT) -> Tuple[faiss.IndexFlatIP, np.ndarray, List[int]]:
    idx_path = Path(index_root) / "faiss.index"
    ids_path = Path(index_root) / "faiss_ids.npy"
    if not idx_path.exists() or not ids_path.exists():
        raise FileNotFoundError("FAISS index files missing. Run start.py.")
    index = faiss.read_index(str(idx_path))
    ids = np.load(str(ids_path))
    return index, ids, ids.tolist()

# ---------------- Embeddings model ----------------
_emb_model: Optional[SentenceTransformer] = None
def get_embedder() -> SentenceTransformer:
    global _emb_model
    if _emb_model is None:
        _emb_model = SentenceTransformer(EMB_MODEL_NAME)
    return _emb_model

def encode(texts: List[str]) -> np.ndarray:
    em = get_embedder()
    vecs = em.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    return vecs.astype("float32")

# ---------------- Retrieval ----------------
def retrieve_with_bm25(bm25: BM25Okapi, docs_meta: List[Dict[str, Any]], query: str, top_n: int = 200) -> List[int]:
    q_tokens = _tokenize(query)
    scores = bm25.get_scores(q_tokens)
    order = np.argsort(-np.array(scores))[:top_n]
    return order.tolist()

def retrieve_with_faiss(faiss_index: faiss.Index, faiss_ids: List[int], docs_meta: List[Dict[str, Any]], query: str, k: int = 100) -> List[int]:
    qv = encode([query])
    D, I = faiss_index.search(qv, k)
    indices = [faiss_ids[i] for i in I[0] if i >= 0]
    return indices

def rerank_with_embeddings(candidates: List[int], docs_meta: List[Dict[str, Any]], query: str, top_k: int = 20, threshold: float = 0.30) -> List[int]:
    # Compute cosine via dot product on normalized vectors
    texts = [docs_meta[i]["text"] for i in candidates]
    qv = encode([query])[0]
    dv = encode(texts)  # normalized
    sims = (dv @ qv)  # cosine
    order = np.argsort(-sims)
    ranked = [candidates[i] for i in order if sims[i] >= threshold]
    return ranked[:top_k]

def select_final_context(doc_ids: List[int], docs_meta: List[Dict[str, Any]], max_context_tokens: int = 500) -> Tuple[str, List[int]]:
    total = 0
    chosen_ids = []
    parts = []
    for i in doc_ids:
        txt = docs_meta[i]["text"].strip()
        t = estimate_tokens(txt)
        if total + t > max_context_tokens:
            break
        parts.append(f"(p.{docs_meta[i]['page']}) {txt}")
        total += t
        chosen_ids.append(i)
    return "\n\n".join(parts), chosen_ids

# ---------------- LM Studio (OpenAI-compatible) ----------------
def generate_llm_answer(context: str, question: str) -> str:
    global _last_use_ts
    _last_use_ts = time.time()
    # If LM Studio is reachable, use it via OpenAI-compatible API.
    try:
        import requests
        url = f"{LMSTUDIO_BASE}/chat/completions"
        headers = {"Authorization": f"Bearer {LMSTUDIO_API_KEY}"}
        payload = {
            "model": LMSTUDIO_MODEL,
            "messages": [
                {"role": "system", "content": "Responda em português, de forma concisa e objetiva. Use apenas o contexto fornecido."},
                {"role": "user", "content": f"Pergunta: {question}\n\nContexto:\n{context}"}
            ],
            "temperature": 0.2,
            "max_tokens": 300
        }
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        msg = data["choices"][0]["message"]["content"].strip()
        return msg
    except Exception as e:
        # Fallback: devolve o contexto "extractive"
        if context.strip():
            return "Aqui está o que encontrei com base no manual:\n\n" + context[:1500]
        return "Não encontrei informações suficientes no contexto."

# ---------------- Page → images ----------------
def collect_page_images(pages: List[int], assets_index: Dict[str, Any], limit: int = 6) -> List[str]:
    out = []
    imgs = assets_index.get("page_images") or {}
    for p in pages:
        for rel in imgs.get(p, []):
            out.append(rel)
            if len(out) >= limit: return out
    return out

# ---------------- Orchestrator ----------------
def run_rag_pipeline(user_query: str,
                     vehicle_model: str,
                     assets_index: Dict[str, Any],
                     bm25: BM25Okapi,
                     docs_meta: List[Dict[str, Any]],
                     faiss_index: faiss.Index,
                     faiss_ids: List[int],
                     max_context_tokens: int = 500) -> Tuple[str, List[str]]:

    query = (vehicle_model + " " + user_query).strip() if vehicle_model else user_query

    # 1) BM25 recall
    bm25_top = retrieve_with_bm25(bm25, docs_meta, query, top_n=200)

    # 2) FAISS ANN — global topK, depois interseção com BM25 para "filtrar" o recall
    faiss_top = retrieve_with_faiss(faiss_index, faiss_ids, docs_meta, query, k=200)
    shortlist = [i for i in bm25_top if i in set(faiss_top)]
    if not shortlist:
        shortlist = bm25_top[:50]  # fallback

    # 3) Rerank embeddings (cosine) nos candidatos
    reranked = rerank_with_embeddings(shortlist, docs_meta, query, top_k=20, threshold=0.30)

    # 4) Seleção por orçamento de tokens
    context, chosen = select_final_context(reranked, docs_meta, max_context_tokens=max_context_tokens)

    # 5) Geração (LM Studio se disponível)
    answer = generate_llm_answer(context, user_query)

    # 6) Imagens das páginas escolhidas
    pages = []
    for i in chosen:
        p = docs_meta[i]["page"]
        if p not in pages:
            pages.append(p)
    image_paths = collect_page_images(pages, assets_index, limit=6)

    # -------- NORMALIZAÇÃO DE CAMINHO --------
    def _norm_image_rel(root: str, p: str, slug: str) -> str:
        # normaliza separadores e remove prefixos redundantes
        p = (p or "").replace("\\", "/").lstrip("./")
        root = (root or "").replace("\\", "/")
        # se for absoluto e dentro de root, calcula relativo
        try:
            ap = os.path.abspath(os.path.join(root, p)) if not os.path.isabs(p) else p
            ap = ap.replace("\\", "/")
            if ap.startswith(root.rstrip("/") + "/"):
                rel = os.path.relpath(ap, root).replace("\\", "/")
            else:
                rel = p
        except Exception:
            rel = p
        # remove prefixo do slug, se já vier com ele
        if rel.startswith(slug + "/"):
            rel = rel[len(slug) + 1:]
        # também remove um eventual 'assets_out/<slug>/' no começo
        for prefix in ("assets_out/", "./assets_out/"):
            pref = f"{prefix}{slug}/"
            if rel.startswith(pref):
                rel = rel[len(pref):]
        return rel

    root = assets_index.get("extract_root") or ""
    # tenta inferir o slug a partir do root (…/assets_out/<slug>)
    slug = os.path.basename(os.path.normpath(root))

    rel_images = []
    for p in image_paths:
        rel_images.append(_norm_image_rel(root, p, slug))

    return answer, rel_images

# ---------------- Idle Unload (placeholder) ----------------
def _idle_monitor():
    global _last_use_ts
    while True:
        time.sleep(60)
        if time.time() - _last_use_ts > IDLE_UNLOAD_SECS:
            # In a real setup, call LM Studio CLI to unload; here we just log.
            # Example: os.system("lms unload --model '%s'" % LMSTUDIO_MODEL)
            _last_use_ts = time.time()  # reset to avoid repeated logs

def ensure_idle_monitor():
    global _idle_thread_started
    if not _idle_thread_started:
        t = threading.Thread(target=_idle_monitor, daemon=True)
        t.start()
        _idle_thread_started = True
