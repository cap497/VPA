import re
from pathlib import Path

# === Hard-coded paths ===
CLEAN_TITLES_FILE = "cleaned_index.txt"
CHAPTER_TITLES_FILE = "chapter_titles.md"
RAW_FOLDER = "raw_chapters"
OUTPUT_FOLDER = "cleaned_chapters"

def load_valid_titles(clean_titles_file):
    """
    Load cleaned valid section titles from cleaned_index.txt
    """
    titles = set()
    with open(clean_titles_file, encoding="utf-8") as f:
        for line in f:
            clean = line.strip().lower()
            if clean:
                titles.add(clean)
    return titles

def load_chapter_titles(chapter_titles_file):
    """
    Load chapter titles from chapter_titles.md and normalize them.
    """
    titles = set()
    with open(chapter_titles_file, encoding="utf-8") as f:
        for line in f:
            clean = normalize_chapter_title(line)
            if clean:
                titles.add(clean)
    return titles

def normalize_chapter_title(line):
    """
    Normalize a line from chapter_titles.md.
    Remove numbering, hashes, dashes, dots.
    Example: '# 1. Introdução' -> 'introdução'
    """
    line = line.strip().lower()
    line = re.sub(r"^[#\-\s\d\.]*", "", line)
    return line.strip()

def is_two_digit_chapter_number(line):
    """
    Detect lines with just a 2-digit chapter number like 01–09,
    even with extra spaces or punctuation.
    """
    # Remove all non-digits
    digits_only = re.sub(r"[^\d]", "", line)
    return digits_only in [f"0{i}" for i in range(1, 10)]

def is_noise_line(line):
    """
    Detect if the line is noise:
    - Horizontal rules (---)
    - Page numbers like '4-10' or '3'
    - 'null' artifacts
    - Lines with only 2-digit chapter numbers 01–09
    """
    stripped = line.strip().lower()

    if re.fullmatch(r"-+", stripped):
        return True

    if re.fullmatch(r"\d+(-\d+)?", stripped):
        return True

    if "null" in stripped:
        return True

    if is_two_digit_chapter_number(line):
        return True

    return False

def is_repeated_chapter_title(line, chapter_titles_set):
    """
    Check if a line is a repeated chapter title to remove.
    """
    normalized = normalize_chapter_title(line)
    return normalized in chapter_titles_set

def normalize_line_for_comparison(line):
    """
    Normalize heading lines in the chapter for comparison.
    Removes leading hashes/dashes/numbers and trailing dots/page numbers.
    """
    # Remove leading symbols/numbers
    line = re.sub(r"^[#\-\s\d\.]*", "", line).strip()
    # Remove trailing dots and page numbers
    line = re.sub(r'\.*\s*\d[\d\-]*\s*$', "", line).strip()
    return line.lower()

def fix_colon_double_newlines(text):
    """
    Collapse multiple newlines after colon to just one.
    """
    return re.sub(r":\s*\n\s*\n+", ":\n", text)

def clean_chapter_text(chapter_text, valid_titles, chapter_titles_set):
    """
    Clean the entire text of a chapter .md file:
    - Remove noise lines
    - Remove repeated chapter title lines
    - Keep only real section headings as '#', demote others to text
    - Fix colon + double-newlines
    """
    lines = chapter_text.splitlines()
    new_lines = []

    for line in lines:
        # 1. Skip noise lines entirely
        if is_noise_line(line):
            continue

        # 2. Skip repeated chapter title lines
        if is_repeated_chapter_title(line, chapter_titles_set):
            continue

        # 3. Process headings
        if line.strip().startswith("#"):
            heading_normalized = normalize_line_for_comparison(line)
            if heading_normalized in valid_titles:
                # Keep as real section heading
                new_lines.append(f"# {line.lstrip('#').strip()}\n")
            else:
                # Demote to normal text
                new_lines.append(f"{line.lstrip('#').strip()}\n")
        else:
            # Keep other lines as is
            new_lines.append(line + "\n")

    # Join and fix colon + double-newlines
    cleaned_text = "".join(new_lines)
    cleaned_text = fix_colon_double_newlines(cleaned_text)
    return cleaned_text

def clean_all_chapters():
    """
    Process all .md files in RAW_FOLDER using CLEAN_TITLES_FILE and CHAPTER_TITLES_FILE
    Save cleaned .md files to OUTPUT_FOLDER.
    """
    Path(OUTPUT_FOLDER).mkdir(exist_ok=True)
    valid_titles = load_valid_titles(CLEAN_TITLES_FILE)
    chapter_titles_set = load_chapter_titles(CHAPTER_TITLES_FILE)

    print(f"✅ Loaded {len(valid_titles)} valid section titles from {CLEAN_TITLES_FILE}")
    print(f"✅ Loaded {len(chapter_titles_set)} chapter titles from {CHAPTER_TITLES_FILE}")

    raw_files = list(Path(RAW_FOLDER).glob("*.md"))
    print(f"📂 Found {len(raw_files)} raw chapter files in {RAW_FOLDER}")

    for file_path in raw_files:
        print(f"🧹 Cleaning {file_path.name}...")
        raw_text = Path(file_path).read_text(encoding="utf-8")
        cleaned_text = clean_chapter_text(
            raw_text,
            valid_titles,
            chapter_titles_set
        )

        output_path = Path(OUTPUT_FOLDER) / file_path.name
        output_path.write_text(cleaned_text, encoding="utf-8")
        print(f"✅ Saved cleaned file to {output_path}")

if __name__ == "__main__":
    clean_all_chapters()
    print("✅ All chapters cleaned successfully.")
