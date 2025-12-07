#!/usr/bin/env python3
import os, json, re, sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

if len(sys.argv) < 2:
    print("Uso: python start.py <modelo>")
    print("Exemplo: python start.py hb20")
    sys.exit(1)

MODEL = sys.argv[1].lower()  # pega o argumento e normaliza

EXTRACT_ROOT = os.environ.get(
    "EXTRACT_ROOT",
    os.path.abspath(f"./assets_out/{MODEL}")
)

INDEX_ROOT = os.environ.get(
    "INDEX_ROOT",
    os.path.abspath(f"./indices/{MODEL}")
)
EMB_MODEL_NAME = os.environ.get("EMB_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

_token_re = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_-]+", re.UNICODE)

def _tokenize(s: str) -> List[str]:
    return [t.lower() for t in _token_re.findall(s or "")]

def load_text_blocks(extract_root: str) -> List[Dict[str, Any]]:
    root = Path(extract_root)
    text_dir = root / "text"
    blocks = []
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
                    blocks.append({"page": page_num, "text": rec["text"]})
    return blocks

def build_bm25_meta(blocks: List[Dict[str, Any]], index_root: str):
    docs_meta = []
    for i, b in enumerate(blocks):
        toks = _tokenize(b["text"])
        docs_meta.append({"id": i, "page": b["page"], "text": b["text"], "tokens": toks})
    meta_path = Path(index_root) / "bm25_meta.json"
    meta_path.write_text(json.dumps({"docs_meta": docs_meta}, ensure_ascii=False), encoding="utf-8")
    print(f"BM25 meta saved to {meta_path}")

def build_faiss_index(blocks: List[Dict[str, Any]], index_root: str):
    model = SentenceTransformer(EMB_MODEL_NAME)
    texts = [b["text"] for b in blocks]
    print(f"Encoding {len(texts)} blocks with {EMB_MODEL_NAME}...")
    vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True).astype("float32")
    index = faiss.IndexFlatIP(vecs.shape[1])  # cosine with normalized vectors
    index.add(vecs)
    faiss_path = Path(index_root) / "faiss.index"
    faiss.write_index(index, str(faiss_path))
    np.save(str(Path(index_root) / "faiss_ids.npy"), np.arange(len(texts), dtype=np.int32))
    print(f"FAISS index saved to {faiss_path}")

def main():
    blocks = load_text_blocks(EXTRACT_ROOT)
    if not blocks:
        raise SystemExit("No text blocks found. Ensure EXTRACT_ROOT/text/page_*.jsonl exists.")
    Path(INDEX_ROOT).mkdir(parents=True, exist_ok=True)
    build_bm25_meta(blocks, INDEX_ROOT)
    build_faiss_index(blocks, INDEX_ROOT)
    print("Done.")

if __name__ == "__main__":
    main()
