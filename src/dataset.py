import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

class TextDataset(Dataset):
    def __init__(self, text_path, tokenizer_path, block_size):
        self.block_size = block_size
        tokenizer = Tokenizer.from_file(tokenizer_path)
        
        print(f"-> Encoding corpus from '{text_path}'...")
        all_ids = []
        
        with open(text_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Encode line-by-line to avoid a single massive Rust allocation crash
        for i, line in enumerate(lines):
            line = line.strip()
            if line:
                ids = tokenizer.encode(line).ids
                all_ids.extend(ids)
            
            if (i + 1) % 10000 == 0:
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
        x = self.data_tensor[idx:idx + self.block_size]
        y = self.data_tensor[idx + 1:idx + self.block_size + 1]
        return x, y


def get_dataloader(text_path, tokenizer_path, block_size, batch_size):
    dataset = TextDataset(text_path, tokenizer_path, block_size)
    
    if len(dataset) < batch_size:
        print(f"Warning: Dataset ({len(dataset)} samples) smaller than batch size ({batch_size})!")

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
