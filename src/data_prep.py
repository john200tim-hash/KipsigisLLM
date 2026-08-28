import os
import glob
import json
import hashlib

CACHE_FILE = "data/processed/corpus_meta.json"

def _get_source_fingerprint(text_sources):
    """Create a fingerprint of all source files based on count + total size + newest mtime."""
    total_files = 0
    total_size = 0
    newest_mtime = 0.0

    for source_dir in text_sources:
        if os.path.isdir(source_dir):
            files = glob.glob(os.path.join(source_dir, "*.txt"))
            total_files += len(files)
            for f in files:
                stat = os.stat(f)
                total_size += stat.st_size
                if stat.st_mtime > newest_mtime:
                    newest_mtime = stat.st_mtime

    return {"files": total_files, "size": total_size, "mtime": round(newest_mtime, 2)}


def merge_raw_data(output_file="data/processed/corpus.txt"):
    """
    Merges text from ALL known text data sources into a single corpus file.
    Skips re-merging if nothing has changed since last build.

    Current text sources (grammar-first approach):
      - data/raw/           : Hand-curated Kipsigis grammar .txt files
      - data/transcripts/   : Transcription .txt files downloaded from HF ASR dataset

    Future sources (add here when ready):
      # - data/audio_transcripts/ : Transcripts paired with audio
    """

    # --- TEXT DATA SOURCES ---
    text_sources = [
        "data/raw",
        "data/transcripts",
        # "data/audio_transcripts",  # Uncomment when audio phase begins
    ]
    # --------------------------

    fingerprint = _get_source_fingerprint(text_sources)

    # Check if corpus is already up to date
    if os.path.exists(output_file) and os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            cached = json.load(f)
        if cached == fingerprint:
            print(f"  -> Corpus is up to date ({fingerprint['files']:,} source files). Skipping re-merge.")
            return True

    # Something changed — rebuild corpus
    all_text_files = []
    for source_dir in text_sources:
        if os.path.isdir(source_dir):
            found = glob.glob(os.path.join(source_dir, "*.txt"))
            if found:
                print(f"  -> Found {len(found):,} .txt files in '{source_dir}'")
                all_text_files.extend(found)
            else:
                print(f"  -> [Notice] No .txt files in '{source_dir}' (skipping)")
        else:
            print(f"  -> [Notice] Folder '{source_dir}' does not exist yet (skipping)")

    if not all_text_files:
        print("Warning: No .txt files found in any data source folder!")
        return False

    print(f"  -> Merging {len(all_text_files):,} total files into {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as out_f:
        for file_path in all_text_files:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as in_f:
                content = in_f.read().strip()
                if content:
                    out_f.write(content + "\n\n")

    # Save fingerprint so next run skips this step
    with open(CACHE_FILE, "w") as f:
        json.dump(fingerprint, f)

    print(f"  -> Corpus built successfully.")
    return True
