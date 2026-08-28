import os
import soundfile as sf
from datasets import load_dataset

def main():
    # Retrieve Hugging Face token
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN environment variable is not set. If this dataset is private, it will fail.")

    # Directories
    audio_dir = "data/audio_raw"
    transcripts_dir = "data/transcripts"
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(transcripts_dir, exist_ok=True)

    # Calculate offset by checking how many files already exist
    existing_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
    start_index = len(existing_files)
    print(f"-> Found {start_index} downloaded files. Resuming at index {start_index + 1}...")

    # Load dataset in streaming mode (doesn't download the massive file at once)
    print("-> Connecting to Hugging Face dataset stream...")
    try:
        dataset = load_dataset(
            "kln001/kalenjin-asr-data", 
            split="validated", 
            streaming=True, 
            token=hf_token
        )
    except Exception as e:
        print(f"Error connecting to dataset: {e}")
        return

    batch_size = 20
    downloaded_this_run = 0

    for i, row in enumerate(dataset):
        # Skip rows we have already downloaded
        if i < start_index:
            continue
            
        # Stop if we hit the batch limit
        if downloaded_this_run >= batch_size:
            print(f"\n[INFO] Batch of {batch_size} completed successfully.")
            print("Run the script again to download the next batch!")
            break

        file_index = i + 1
        audio_path = os.path.join(audio_dir, f"file_{file_index:03d}.wav")
        txt_path = os.path.join(transcripts_dir, f"file_{file_index:03d}.txt")

        try:
            # Locate the transcription field (common field names in HF ASR datasets)
            transcript = row.get("sentence") or row.get("text") or row.get("transcription")
            if not transcript:
                print(f"[Warning] Skipping row {file_index}: Missing text transcription.")
                continue

            # Locate the audio array
            audio_data = row.get("audio")
            if not audio_data or "array" not in audio_data or "sampling_rate" not in audio_data:
                print(f"[Warning] Skipping row {file_index}: Missing or invalid audio data.")
                continue

            # 1. Save Audio (WAV format)
            sf.write(audio_path, audio_data["array"], audio_data["sampling_rate"])

            # 2. Save Transcript
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(transcript)

            print(f" [+] Saved file_{file_index:03d}.wav / .txt")
            downloaded_this_run += 1

        except Exception as e:
            print(f"[Error] Failed processing row {file_index}: {e}")

if __name__ == "__main__":
    main()
