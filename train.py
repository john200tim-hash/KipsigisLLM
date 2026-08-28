import yaml
import torch
import hashlib
import os
from src.model import TinyCustomLLM
from src.dataset import get_dataloader
from src.data_prep import merge_raw_data
from src.tokenizer_train import train_tokenizer

def get_checkpoint_sha(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def main():
    print("-> Loading Config...")
    with open("configs/train_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 1. Dynamically merge all .txt files from all data source folders into corpus.txt
    print("-> Scanning data sources and building corpus...")
    merge_raw_data(output_file="data/processed/corpus.txt")
    
    # 2. Retrain Tokenizer dynamically on the new corpus
    train_tokenizer("data/processed/corpus.txt", "tokenizer/kipsigis_tokenizer.json", vocab_size=config['vocab_size'])

    print("-> Initializing Model & Data Loader...")
    model = TinyCustomLLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

    dataloader = get_dataloader(
        text_path="data/processed/corpus.txt",
        tokenizer_path="tokenizer/kipsigis_tokenizer.json",
        block_size=config['block_size'],
        batch_size=config['batch_size']
    )

    print(f"\n-> Starting Training Loop ({config['epochs']} Epochs)...")
    model.train()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    total_steps = 0
    for epoch in range(config['epochs']):
        for step, (xb, yb) in enumerate(dataloader):
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, loss = model(xb, yb)
            loss.backward()
            optimizer.step()
            total_steps += 1

        print(f"Epoch {epoch+1}/{config['epochs']} | Final Loss: {loss.item():.4f}")

    ckpt_path = "checkpoints/kipsigis_model.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n[Training Complete & Checkpoint Saved]")
    print(f"-> File: {ckpt_path}")
    print(f"-> Checkpoint SHA-256: {get_checkpoint_sha(ckpt_path)}")

if __name__ == "__main__":
    main()
