import os
import glob

def merge_raw_data(output_file="data/processed/corpus.txt"):
    """
    Merges text from ALL known text data sources into a single corpus file.

    Current text sources (grammar-first approach):
      - data/raw/           : Your hand-curated Kipsigis grammar .txt files
      - data/transcripts/   : Transcription .txt files downloaded from HF ASR dataset

    Future sources (add here when ready):
      # - data/audio_transcripts/ : Transcripts paired with audio
    """
    
    # --- TEXT DATA SOURCES ---
    # Add or comment out folders here to include/exclude them from training
    text_sources = [
        "data/raw",
        "data/transcripts",
        # "data/audio_transcripts",  # Uncomment when audio phase begins
    ]
    # --------------------------

    all_text_files = []
    for source_dir in text_sources:
        if os.path.isdir(source_dir):
            found = glob.glob(os.path.join(source_dir, "*.txt"))
            if found:
                print(f"  -> Found {len(found)} .txt files in '{source_dir}'")
                all_text_files.extend(found)
            else:
                print(f"  -> [Notice] No .txt files in '{source_dir}' (skipping)")
        else:
            print(f"  -> [Notice] Folder '{source_dir}' does not exist yet (skipping)")

    if not all_text_files:
        print("Warning: No .txt files found in any data source folder!")
        return False

    print(f"  -> Merging {len(all_text_files)} total files into {output_file}...")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as out_f:
        for file_path in all_text_files:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as in_f:
                content = in_f.read().strip()
                if content:
                    out_f.write(content + "\n\n")

    print(f"  -> Corpus built successfully.")
    return True
