import os
import pickle
import time
import sys
import subprocess
from pathlib import Path

from rank_bm25 import BM25Okapi

CHAPTERS_FOLDER = "cleaned_chapters"
BM25_OUT = "bm25_index"
MAX_PARAGRAPH_GROUP_TOKENS = 200

# ✅ Hard-coded target model name
TARGET_MODEL_NAME = "meta-llama-3-8b-instruct"


# --------------------------------------------------------------------------------
# LM Studio CLI integration
# --------------------------------------------------------------------------------

def run_lms_command(args):
    try:
        result = subprocess.run(
            ["lms"] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",       # <<<<<< This line is the fix
            errors="replace",       # <<<<<< Optional but recommended
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running 'lms {' '.join(args)}':", e.stderr)
        return None

def get_downloaded_models():
    """
    Uses `lms ls` to list all downloaded LLM models only.
    """
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

        # Start of LLM section
        if line.startswith("LLMs (Large Language Models"):
            in_llm_section = True
            continue

        # End of LLM section
        if in_llm_section and ("Embedding Models" in line):
            break

        if in_llm_section:
            # Skip header row
            if line.startswith("PARAMS") or "ARCHITECTURE" in line or "SIZE" in line:
                continue

            # Extract first column (model name)
            parts = line.split()
            if parts:
                models.append(parts[0])

    return models

def get_loaded_models():
    """
    Uses `lms ps` to list currently loaded models.
    """
    output = run_lms_command(["ps"])
    if not output:
        return []

    models = set()
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("Identifier:"):
            identifier = line[len("Identifier:"):].strip()
            # Remove any suffix after colon (for multi-instances)
            base_name = identifier.split(":")[0].strip()
            models.add(base_name)
    return list(models)

def lms_load_model(model_name):
    print(f"⚡ Loading model: {model_name}")
    run_lms_command(["load", model_name])

def ensure_model_loaded():
    print("🧭 Checking LM Studio models...")

    # Check downloaded models
    downloaded = get_downloaded_models()
    if not any(TARGET_MODEL_NAME in m for m in downloaded):
        print(f"❌ Model '{TARGET_MODEL_NAME}' is NOT downloaded.")
        print("➡️ Continuing without loading. Existing code will handle failure if no model is loaded.")
        return

    print(f"✅ Model '{TARGET_MODEL_NAME}' is downloaded.")

    # Check if already loaded
    loaded = get_loaded_models()
    if any(TARGET_MODEL_NAME in m for m in loaded):
        print(f"✅ Model '{TARGET_MODEL_NAME}' is already loaded in RAM.")
        return

    # Load model
    lms_load_model(TARGET_MODEL_NAME)

    # Wait up to 5 cycles of 3 seconds
    for attempt in range(5):
        print(f"⌛ Waiting for load... (attempt {attempt+1}/5)")
        time.sleep(3)
        loaded = get_loaded_models()
        if any(TARGET_MODEL_NAME in m for m in loaded):
            print(f"✅ Model '{TARGET_MODEL_NAME}' is now loaded!")
            return

    print(f"❌ Could not load model '{TARGET_MODEL_NAME}' after 5 attempts. Aborting.")
    sys.exit(1)


# --------------------------------------------------------------------------------
# BM25 indexing logic
# --------------------------------------------------------------------------------

def estimate_tokens(text):
    return len(text.split())

def split_subsections(md_text):
    """
    Split text on lines starting with #. Returns list of (title, body).
    """
    lines = md_text.splitlines()
    sections = []
    current_title = None
    current_body = []

    for line in lines:
        if line.strip().startswith("#"):
            if current_title:
                sections.append((current_title.strip(), "\n".join(current_body).strip()))
            current_title = line.lstrip("#").strip()
            current_body = []
        else:
            current_body.append(line)
    if current_title:
        sections.append((current_title.strip(), "\n".join(current_body).strip()))
    return sections

def split_body_into_chunks(body, max_tokens=MAX_PARAGRAPH_GROUP_TOKENS):
    """
    Split body into groups of paragraphs where each group is under max_tokens.
    """
    paragraphs = body.split('\n\n')
    chunks = []
    current_chunk = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_len = estimate_tokens(para)
        if para_len > max_tokens:
            chunks.append(para)
            continue
        if current_len + para_len > max_tokens:
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_len = para_len
        else:
            current_chunk.append(para)
            current_len += para_len
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    return chunks


# --------------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------------

def main():
    ensure_model_loaded()

    Path(BM25_OUT).mkdir(exist_ok=True)

    all_sections = []
    all_texts_for_bm25 = []

    print(f"📂 Reading chapters from: {CHAPTERS_FOLDER}")
    for file in Path(CHAPTERS_FOLDER).glob("*.md"):
        chapter_name = file.stem
        print(f"✅ Processing {file.name}")

        md_text = file.read_text(encoding="utf-8")
        subsections = split_subsections(md_text)

        for title, body in subsections:
            if not body.strip():
                continue

            body_chunks = split_body_into_chunks(body)
            for idx, chunk in enumerate(body_chunks):
                part_title = title
                if len(body_chunks) > 1:
                    part_title += f" (part {idx+1})"

                section = {
                    "chapter": chapter_name,
                    "title": part_title,
                    "body": chunk
                }
                all_sections.append(section)
                bm25_text = f"{part_title}\n{chunk}".lower()
                all_texts_for_bm25.append(bm25_text)

    print(f"✅ Total BM25 units: {len(all_sections)}")

    print("🤖 Fitting BM25...")
    tokenized_corpus = [doc.split() for doc in all_texts_for_bm25]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(os.path.join(BM25_OUT, "bm25.pkl"), "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "sections": all_sections,
            "texts": all_texts_for_bm25
        }, f)
    print("✅ BM25 index saved.")


# --------------------------------------------------------------------------------
# CLI entry
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
