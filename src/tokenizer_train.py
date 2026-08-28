import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

def train_tokenizer(corpus_path, save_path, vocab_size=6000):
    # Skip if tokenizer already exists and is newer than the corpus
    if os.path.exists(save_path):
        tok_mtime = os.path.getmtime(save_path)
        corpus_mtime = os.path.getmtime(corpus_path)
        if tok_mtime >= corpus_mtime:
            print(f"-> Tokenizer is up to date. Skipping retrain.")
            return

    print(f"-> Training custom BPE Tokenizer on {corpus_path}...")
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["<unk>", "<pad>"])
    tokenizer.train([corpus_path], trainer)
    tokenizer.save(save_path)
    print(f"-> Tokenizer successfully built and saved as '{save_path}'.")

if __name__ == "__main__":
    train_tokenizer("data/processed/corpus.txt", "tokenizer/kipsigis_tokenizer.json")
