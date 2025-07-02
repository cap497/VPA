import re
from pathlib import Path

def normalize_title(line):
    """
    Remove leading symbols and numbers, trailing page dots and numbers.
    Example:
    - '# 8. Situações de emergência' -> 'Situações de emergência'
    - 'Como utilizar este manual..................1-3' -> 'Como utilizar este manual'
    """
    # Remove leading hashes/dashes/numbers
    line = re.sub(r"^[#\-\s\d\.]*", "", line).strip()

    # Remove dots and page numbers at end
    line = re.sub(r'\.*\s*\d[\d\-]*\s*$', "", line).strip()

    return line

def extract_clean_titles(index_file, output_file):
    lines = Path(index_file).read_text(encoding="utf-8").splitlines()
    titles = set()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Only process lines that look like titles or list items
        if stripped.startswith("#") or stripped.startswith("-") or not stripped.startswith(" "):
            cleaned = normalize_title(stripped)
            if cleaned:
                titles.add(cleaned.lower())

    # Write results
    with open(output_file, "w", encoding="utf-8") as f:
        for title in sorted(titles):
            f.write(title + "\n")

    print(f"✅ Extracted {len(titles)} clean titles to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract clean titles from index.md")
    parser.add_argument("index_file", help="Path to index.md")
    parser.add_argument("output_file", help="Path to save clean titles list (one per line)")
    args = parser.parse_args()

    extract_clean_titles(args.index_file, args.output_file)
