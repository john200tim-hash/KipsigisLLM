import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

class TextDataset(Dataset):
    def __init__(self, text_path, tokenizer_path, block_size):
        self.block_size = block_size
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        with open(text_path, "r", encoding="utf-8") as f:
            text_data = f.read()
            
        self.encoded_tokens = self.tokenizer.encode(text_data).ids
        self.data_tensor = torch.tensor(self.encoded_tokens, dtype=torch.long)

    def __len__(self):
        return len(self.data_tensor) - self.block_size

    def __getitem__(self, idx):
        x = self.data_tensor[idx:idx + self.block_size]
        y = self.data_tensor[idx + 1:idx + self.block_size + 1]
        return x, y

def get_dataloader(text_path, tokenizer_path, block_size, batch_size):
    dataset = TextDataset(text_path, tokenizer_path, block_size)
    
    # Using drop_last=True if batch sizes don't perfectly align
    # If the dataset is too small, fallback
    if len(dataset) < batch_size:
        print("Warning: Dataset smaller than batch size!")
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
