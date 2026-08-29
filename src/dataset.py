import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tokenizers import Tokenizer


class TextDataset(Dataset):
    def __init__(self, text_path, tokenizer_path, block_size):
        self.block_size = block_size
        tokenizer = Tokenizer.from_file(tokenizer_path)

        print(f"-> Encoding corpus from '{text_path}'...")
        all_ids = []

        with open(text_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Encode line-by-line to avoid a single massive Rust allocation crash.
        # Blank lines become a natural boundary token (EOS-like) between paragraphs
        # so the model learns to terminate sentences/paragraphs cleanly.
        for i, line in enumerate(lines):
            line = line.strip()
            if line:
                ids = tokenizer.encode(line).ids
                all_ids.extend(ids)
            else:
                # Blank line = paragraph boundary: encode a newline character
                # so the model sees a separator between paragraph chunks.
                ids = tokenizer.encode("\n").ids
                all_ids.extend(ids)

            if (i + 1) % 10_000 == 0:
                print(f"   Encoded {i+1:,}/{len(lines):,} lines ({len(all_ids):,} tokens so far)...")

        print(f"-> Corpus encoded: {len(all_ids):,} total tokens")

        if len(all_ids) < block_size + 1:
            raise ValueError(
                f"Corpus too small! Only {len(all_ids)} tokens but block_size={block_size}. "
                "Add more training data."
            )

        self.data_tensor = torch.tensor(all_ids, dtype=torch.long)

    def __len__(self):
        return len(self.data_tensor) - self.block_size

    def __getitem__(self, idx):
        x = self.data_tensor[idx : idx + self.block_size]
        y = self.data_tensor[idx + 1 : idx + self.block_size + 1]
        return x, y


def get_dataloader(text_path, tokenizer_path, block_size, batch_size, val_split=0.0):
    """
    Returns (train_loader, val_loader).
    If val_split == 0.0, val_loader is None.
    """
    dataset = TextDataset(text_path, tokenizer_path, block_size)

    if val_split > 0.0:
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        if val_size == 0:
            print("Warning: Dataset too small for a validation split. Skipping val split.")
            train_ds, val_ds = dataset, None
        else:
            train_ds, val_ds = random_split(
                dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )
            print(f"-> Train: {train_size:,} samples | Val: {val_size:,} samples")
    else:
        train_ds = dataset
        val_ds = None

    if len(train_ds) < batch_size:
        print(f"Warning: Dataset ({len(train_ds)} samples) smaller than batch size ({batch_size})!")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
        if val_ds is not None
        else None
    )
    return train_loader, val_loader
