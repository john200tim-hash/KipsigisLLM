import yaml
import torch
import hashlib
import math
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

    # 1. Merge all text sources into one corpus
    print("-> Scanning data sources and building corpus...")
    merge_raw_data(output_file="data/processed/corpus.txt")

    # 2. Retrain tokenizer on the new corpus
    train_tokenizer("data/processed/corpus.txt", "tokenizer/kipsigis_tokenizer.json", vocab_size=config['vocab_size'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Using device: {device}")

    # 3. Init model
    print("-> Initializing Model & Data Loader...")
    model = TinyCustomLLM(config).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"-> Model parameters: {total_params:,}")

    # 4. Dataloader
    dataloader = get_dataloader(
        text_path="data/processed/corpus.txt",
        tokenizer_path="tokenizer/kipsigis_tokenizer.json",
        block_size=config['block_size'],
        batch_size=config['batch_size']
    )

    # 5. Optimizer with weight decay (AdamW standard practice)
    # Separate params that should and shouldn't have weight decay
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": 0.1},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=config['learning_rate'], betas=(0.9, 0.95))

    # 6. Cosine LR scheduler with warmup
    total_steps = config['epochs'] * len(dataloader)
    warmup_steps = total_steps // 10  # 10% of training is warmup

    def get_lr(step):
        # Linear warmup
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        # Cosine decay down to 10% of original LR
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    # 7. Training loop
    print(f"\n-> Starting Training Loop ({config['epochs']} Epochs, {total_steps:,} total steps)...")
    model.train()
    os.makedirs("checkpoints", exist_ok=True)
    best_loss = float('inf')

    for epoch in range(config['epochs']):
        epoch_loss = 0.0
        for step, (xb, yb) in enumerate(dataloader):
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, loss = model(xb, yb)
            loss.backward()

            # Gradient clipping prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        current_lr = scheduler.get_last_lr()[0] * config['learning_rate']
        print(f"Epoch {epoch+1:>4}/{config['epochs']} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

        # Save best checkpoint automatically
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = "checkpoints/kipsigis_model.pt"
            torch.save(model.state_dict(), ckpt_path)

    print(f"\n[Training Complete]")
    print(f"-> Best Loss: {best_loss:.4f}")
    print(f"-> Checkpoint: checkpoints/kipsigis_model.pt")
    print(f"-> SHA-256: {get_checkpoint_sha('checkpoints/kipsigis_model.pt')}")

if __name__ == "__main__":
    main()
