import yaml
import torch
import hashlib
import math
import os
import sys
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

    max_steps = config.get('max_steps', None)

    # 1. Merge all text sources into one corpus
    print("-> Scanning data sources and building corpus...")
    merge_raw_data(output_file="data/processed/corpus.txt")

    # 2. Retrain tokenizer on the new corpus
    train_tokenizer("data/processed/corpus.txt", "tokenizer/kipsigis_tokenizer.json", vocab_size=config['vocab_size'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Using device: {device}")

    # 3. Init model
    print("-> Initializing Model & Data Loader...")
    try:
        model = TinyCustomLLM(config).to(device)
    except Exception as e:
        print(f"[FATAL] Failed to initialize model: {e}")
        sys.exit(1)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"-> Model parameters: {total_params:,}")

    # 4. Dataloader
    try:
        dataloader = get_dataloader(
            text_path="data/processed/corpus.txt",
            tokenizer_path="tokenizer/kipsigis_tokenizer.json",
            block_size=config['block_size'],
            batch_size=config['batch_size']
        )
        print(f"-> Dataset loaded: {len(dataloader):,} batches per epoch")
    except Exception as e:
        print(f"[FATAL] Failed to load dataset: {e}")
        sys.exit(1)

    if len(dataloader) == 0:
        print("[FATAL] Dataset is empty! Check your data/processed/corpus.txt file.")
        sys.exit(1)

    # 5. Optimizer
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": 0.1},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=config['learning_rate'], betas=(0.9, 0.95))

    # 6. LR scheduler
    effective_total = max_steps if max_steps else config['epochs'] * len(dataloader)
    warmup_steps = effective_total // 10

    def get_lr(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, effective_total - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    # 7. Training loop
    limit_str = f"max {max_steps:,} steps" if max_steps else f"{config['epochs']} epochs"
    print(f"\n-> Starting Training ({limit_str}, {len(dataloader):,} batches/epoch)...")
    print("   [Progress prints every 100 steps]\n")

    model.train()
    os.makedirs("checkpoints", exist_ok=True)
    best_loss = float('inf')
    global_step = 0
    running_loss = 0.0

    try:
        for epoch in range(config['epochs']):
            epoch_loss = 0.0
            epoch_steps = 0

            for step, (xb, yb) in enumerate(dataloader):
                # Check max_steps cap
                if max_steps and global_step >= max_steps:
                    break

                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits, loss = model(xb, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                loss_val = loss.item()
                epoch_loss += loss_val
                running_loss += loss_val
                global_step += 1
                epoch_steps += 1

                # Print every 100 steps so user can see progress
                if global_step % 100 == 0:
                    avg_100 = running_loss / 100
                    current_lr = scheduler.get_last_lr()[0] * config['learning_rate']
                    print(f"  Step {global_step:>6,} | Loss: {avg_100:.4f} | LR: {current_lr:.6f}")
                    running_loss = 0.0

                    # Save checkpoint every 500 steps if loss improved
                    if global_step % 500 == 0 and avg_100 < best_loss:
                        best_loss = avg_100
                        torch.save(model.state_dict(), "checkpoints/kipsigis_model.pt")
                        print(f"  [Checkpoint saved at step {global_step:,} - Loss: {best_loss:.4f}]")

            if max_steps and global_step >= max_steps:
                print(f"\n-> Reached max_steps ({max_steps:,}). Stopping.")
                break

            avg_loss = epoch_loss / max(1, epoch_steps)
            print(f"\nEpoch {epoch+1}/{config['epochs']} complete | Avg Loss: {avg_loss:.4f}\n")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), "checkpoints/kipsigis_model.pt")
                print(f"  [Best checkpoint saved - Loss: {best_loss:.4f}]")

    except KeyboardInterrupt:
        print("\n\n[INFO] Training interrupted by user. Saving current state...")
    except Exception as e:
        print(f"\n[FATAL ERROR during training at step {global_step}]: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always try to save on exit
        if global_step > 0:
            final_path = "checkpoints/kipsigis_model.pt"
            torch.save(model.state_dict(), final_path)
            print(f"\n[Training ended at step {global_step:,}]")
            print(f"-> Best Loss achieved: {best_loss:.4f}")
            print(f"-> Checkpoint: {final_path}")
            if os.path.exists(final_path):
                print(f"-> SHA-256: {get_checkpoint_sha(final_path)}")
        else:
            print("\n[WARNING] No training steps completed. No checkpoint saved.")

if __name__ == "__main__":
    main()
