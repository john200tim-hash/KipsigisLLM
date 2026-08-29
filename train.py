import argparse
import hashlib
import math
import os
import sys
import yaml
import torch

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


@torch.no_grad()
def evaluate_val_loss(model, val_loader, device):
    """Run one pass over the validation set and return average loss."""
    if val_loader is None:
        return None
    model.eval()
    total_loss = 0.0
    steps = 0
    for xb, yb in val_loader:
        xb, yb = xb.to(device), yb.to(device)
        _, loss = model(xb, yb)
        total_loss += loss.item()
        steps += 1
    model.train()
    return total_loss / max(1, steps)


def main():
    parser = argparse.ArgumentParser(description="Train Kipsigis LLM")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to training YAML config file (default: configs/train_config.yaml)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"[FATAL] Config file not found: {args.config}")
        sys.exit(1)

    print(f"-> Loading Config from '{args.config}'...")
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw_data_dir = config.get("raw_data_dir", "data/raw")
    processed_corpus_path = config.get("processed_corpus_path", "data/processed/corpus.txt")
    merged_transcript_path = config.get("merged_transcript_path", "data/all_transcripts_merged.txt")
    tokenizer_path = config.get("tokenizer_path", "tokenizer/kipsigis_tokenizer.json")
    checkpoint_path = config.get("checkpoint_path", "checkpoints/kipsigis_model.pt")

    grammar_repeat = config.get("grammar_repeat", 5)
    val_split = config.get("val_split", 0.1)
    max_steps = config.get("max_steps", None)

    # 1. Merge all text sources into one grammar-boosted corpus
    print("-> Scanning data sources and building corpus...")
    success = merge_raw_data(
        grammar_dir=raw_data_dir,
        merged_transcript=merged_transcript_path,
        output_file=processed_corpus_path,
        grammar_repeat=grammar_repeat,
    )

    if not success or not os.path.exists(processed_corpus_path) or os.path.getsize(processed_corpus_path) == 0:
        print("[FATAL] Corpus generation failed or resulted in an empty file!")
        print("Please place text files (.txt) inside 'data/raw/' to proceed with training.")
        sys.exit(1)

    # 2. Retrain tokenizer on the new corpus
    print(f"-> Training/updating tokenizer at '{tokenizer_path}'...")
    train_tokenizer(
        corpus_path=processed_corpus_path,
        save_path=tokenizer_path,
        vocab_size=config["vocab_size"],
    )

    # Device selection
    device_cfg = config.get("device", "auto")
    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)
    print(f"-> Using device: {device}")

    # 3. Init model
    print("-> Initializing Model...")
    try:
        model = TinyCustomLLM(config).to(device)
    except Exception as e:
        print(f"[FATAL] Failed to initialize model: {e}")
        sys.exit(1)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"-> Model parameters: {total_params:,}")

    # 4. Dataloader with validation split
    print("-> Loading dataset...")
    try:
        train_loader, val_loader = get_dataloader(
            text_path=processed_corpus_path,
            tokenizer_path=tokenizer_path,
            block_size=config["block_size"],
            batch_size=config["batch_size"],
            val_split=val_split,
        )
        print(f"-> Train: {len(train_loader):,} batches/epoch")
        if val_loader:
            print(f"-> Val:   {len(val_loader):,} batches")
    except Exception as e:
        print(f"[FATAL] Failed to load dataset: {e}")
        sys.exit(1)

    if len(train_loader) == 0:
        print(f"[FATAL] Training dataset is empty! Check {processed_corpus_path}.")
        sys.exit(1)

    # 5. Optimizer — AdamW with weight decay on 2D parameters only
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": 0.1},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config["learning_rate"],
        betas=(0.9, 0.95),
    )

    # 6. Cosine LR scheduler with linear warmup (10% of training)
    effective_total = (
        max_steps if max_steps else config["epochs"] * len(train_loader)
    )
    warmup_steps = max(10, effective_total // 10)

    def get_lr(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, effective_total - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    # 7. Training loop
    limit_str = f"max {max_steps:,} steps" if max_steps else f"{config['epochs']} epochs"
    print(f"\n-> Starting Training ({limit_str}, {len(train_loader):,} batches/epoch)")
    print(f"   Grammar boost: {grammar_repeat}x | Block size: {config['block_size']}")
    print("   [Loss printed every 100 steps | Val loss every 500 steps]\n")

    model.train()
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    best_val_loss = float("inf")
    global_step = 0
    running_loss = 0.0

    try:
        for epoch in range(config["epochs"]):
            epoch_loss = 0.0
            epoch_steps = 0

            for step, (xb, yb) in enumerate(train_loader):
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

                # ── Print training loss every 100 steps ──────────────────────
                if global_step % 100 == 0:
                    avg_100 = running_loss / 100
                    current_lr = scheduler.get_last_lr()[0] * config["learning_rate"]
                    print(
                        f"  Step {global_step:>6,} | Train Loss: {avg_100:.4f} | LR: {current_lr:.2e}"
                    )
                    running_loss = 0.0

                # ── Validate and checkpoint every 500 steps ───────────────────
                if global_step % 500 == 0:
                    val_loss = evaluate_val_loss(model, val_loader, device)
                    if val_loss is not None:
                        print(f"  [Val Loss @ step {global_step:,}: {val_loss:.4f}]", end="")
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            torch.save(model.state_dict(), checkpoint_path)
                            print(f"  <- NEW BEST, checkpoint saved")
                        else:
                            print()
                    else:
                        avg = epoch_loss / max(1, epoch_steps)
                        if avg < best_val_loss:
                            best_val_loss = avg
                            torch.save(model.state_dict(), checkpoint_path)
                            print(f"  [Checkpoint saved at step {global_step:,}]")

            if max_steps and global_step >= max_steps:
                print(f"\n-> Reached max_steps ({max_steps:,}). Stopping.")
                break

            avg_loss = epoch_loss / max(1, epoch_steps)
            print(f"\nEpoch {epoch+1}/{config['epochs']} complete | Avg Train Loss: {avg_loss:.4f}\n")

    except KeyboardInterrupt:
        print("\n\n[INFO] Training interrupted by user. Saving current state...")
    except Exception as e:
        print(f"\n[FATAL ERROR during training at step {global_step}]: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always save on exit
        if global_step > 0:
            torch.save(model.state_dict(), checkpoint_path)
            print(f"\n[Training ended at step {global_step:,}]")
            print(f"-> Best Loss achieved: {best_val_loss:.4f}")
            print(f"-> Checkpoint saved: {checkpoint_path}")
            if os.path.exists(checkpoint_path):
                print(f"-> SHA-256: {get_checkpoint_sha(checkpoint_path)}")
        else:
            print("\n[WARNING] No training steps completed. No checkpoint saved.")


if __name__ == "__main__":
    main()
