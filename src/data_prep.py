import os
import glob
import json
import random

CACHE_FILE = "data/processed/corpus_meta.json"

def _get_source_fingerprint(grammar_dir, merged_transcript, grammar_repeat):
    """Create a fingerprint based on source file count, total size, newest mtime, and grammar_repeat."""
    total_files = 0
    total_size = 0
    newest_mtime = 0.0

    # Check grammar dir
    if os.path.isdir(grammar_dir):
        files = glob.glob(os.path.join(grammar_dir, "*.txt"))
        total_files += len(files)
        for f in files:
            stat = os.stat(f)
            total_size += stat.st_size
            if stat.st_mtime > newest_mtime:
                newest_mtime = stat.st_mtime
                
    # Check merged transcript file
    if os.path.exists(merged_transcript):
        total_files += 1
        stat = os.stat(merged_transcript)
        total_size += stat.st_size
        if stat.st_mtime > newest_mtime:
            newest_mtime = stat.st_mtime

    return {
        "files": total_files,
        "size": total_size,
        "mtime": round(newest_mtime, 2),
        "grammar_repeat": grammar_repeat,
    }


def _read_file(path):
    """Read a file, returning stripped text or empty string."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except Exception:
        return ""


def _consolidate_from_merged_transcript(merged_file, sentences_per_chunk=10):
    """
    Reads the single merged transcript file (one sentence per line),
    and groups them into paragraph chunks.
    """
    if not os.path.exists(merged_file):
        return []
        
    with open(merged_file, "r", encoding="utf-8", errors="ignore") as f:
        sentences = [line.strip() for line in f if line.strip()]
        
    if not sentences:
        return []

    # Shuffle so grouped chunks aren't all from the same recording session sequentially
    random.shuffle(sentences)

    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = "\n".join(sentences[i : i + sentences_per_chunk])
        if chunk.strip():
            chunks.append(chunk)

    return chunks


def merge_raw_data(grammar_dir="data/raw", merged_transcript="data/all_transcripts_merged.txt", output_file="data/processed/corpus.txt", grammar_repeat=5):
    """
    Merges text from ALL known text sources into a single grammar-boosted corpus.

    Grammar-first strategy:
      - grammar_dir    : Curated Kipsigis text (grammar, prose). Repeated
                         `grammar_repeat` times.
      - merged_transcript: ASR transcription file (one sentence per line).
                         Consolidated into 10-sentence paragraph chunks before
                         merging, giving the model sentence-to-sentence context.
    """
    SENTENCES_PER_CHUNK = 10

    fingerprint = _get_source_fingerprint(grammar_dir, merged_transcript, grammar_repeat)

    # ── Cache check: skip rebuild if nothing changed ──────────────────────────
    if os.path.exists(output_file) and os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                cached = json.load(f)
            if cached == fingerprint:
                print(
                    f"  -> Corpus is up to date "
                    f"({fingerprint['files']:,} source files, "
                    f"grammar_repeat={grammar_repeat}). Skipping re-merge."
                )
                return True
        except Exception:
            pass

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    paragraphs = []  # list of text blocks to write

    # ── 1. Grammar files (repeated grammar_repeat times) ─────────────────────
    grammar_files = glob.glob(os.path.join(grammar_dir, "*.txt")) if os.path.isdir(grammar_dir) else []
    if grammar_files:
        print(
            f"  -> Found {len(grammar_files)} grammar file(s) in '{grammar_dir}' "
            f"(repeating {grammar_repeat}x for grammar emphasis)"
        )
        for _ in range(grammar_repeat):
            for gf in grammar_files:
                text = _read_file(gf)
                if text:
                    paragraphs.append(text)
    else:
        print(f"  -> [Notice] No text files in '{grammar_dir}' — add .txt files here!")

    # ── 2. Transcript files (consolidated into paragraph chunks) ──────────────
    chunks = _consolidate_from_merged_transcript(merged_transcript, SENTENCES_PER_CHUNK)
    if chunks:
        print(
            f"  -> Consolidated sentences from '{merged_transcript}' into {len(chunks):,} paragraph chunks "
            f"({SENTENCES_PER_CHUNK} sentences each)"
        )
        paragraphs.extend(chunks)
    else:
        if os.path.exists(merged_transcript):
            print(f"  -> [Notice] No transcripts read from '{merged_transcript}'")

    if not paragraphs:
        print("Warning: No text found in any data source folder!")
        return False

    # ── 3. Shuffle all paragraphs so grammar and transcripts interleave ───────
    random.shuffle(paragraphs)

    print(f"  -> Writing {len(paragraphs):,} paragraphs to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as out_f:
        for para in paragraphs:
            out_f.write(para.strip() + "\n\n")

    # Save fingerprint
    with open(CACHE_FILE, "w") as f:
        json.dump(fingerprint, f)

    corpus_size_kb = os.path.getsize(output_file) / 1024
    print(f"  -> Corpus built: {corpus_size_kb:.1f} KB  ({len(paragraphs):,} paragraphs)")
    return True
