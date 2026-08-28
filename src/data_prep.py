import os
import glob

def merge_raw_data(raw_dir="data/raw", output_file="data/processed/corpus.txt"):
    text_files = glob.glob(os.path.join(raw_dir, "*.txt"))
    if not text_files:
        print(f"Warning: No .txt files found in {raw_dir}")
        return False
        
    print(f"-> Found {len(text_files)} .txt files in {raw_dir}. Merging into {output_file}...")
    combined_text = ""
    for file_path in text_files:
        with open(file_path, "r", encoding="utf-8") as f:
            combined_text += f.read() + "\n\n"
            
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(combined_text)
    return True
