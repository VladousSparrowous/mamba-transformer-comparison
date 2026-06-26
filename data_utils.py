import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

class LRATextDataset(Dataset):
    """LRA Text dataset (IMDb reviews at character level)"""
    
    def __init__(self, split="train", max_seq_len=2048):
        self.max_seq_len = max_seq_len
        
        # Load IMDb dataset
        dataset = load_dataset("imdb", split=split)
        
        # Character-level tokenization
        self.char_to_idx = self._build_vocab(dataset)
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        
        self.data = []
        self.labels = []
        
        for item in tqdm(dataset, desc=f"Processing {split} set"):
            # Tokenize and truncate/pad
            tokens = self._tokenize(item["text"])
            self.data.append(tokens)
            self.labels.append(item["label"])
    
    def _build_vocab(self, dataset):
        """Build character vocabulary"""
        chars = set()
        for item in dataset:
            chars.update(set(item["text"]))
        chars = sorted(list(chars))
        # Add special tokens
        vocab = {"<PAD>": 0, "<MASK>": 1, "<UNK>": 2}
        for i, char in enumerate(chars):
            vocab[char] = i + 3
        return vocab
    
    def _tokenize(self, text):
        """Convert text to token IDs"""
        tokens = []
        for char in text:
            tokens.append(self.char_to_idx.get(char, self.char_to_idx["<UNK>"]))
        
        # Truncate or pad
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        else:
            tokens = tokens + [self.char_to_idx["<PAD>"]] * (self.max_seq_len - len(tokens))
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class LRATextDatasetSPT(Dataset):
    """Dataset for Self-Pretraining (SPT) with masking"""
    
    def __init__(self, base_dataset, masking_ratio=0.15):
        self.base = base_dataset
        self.masking_ratio = masking_ratio
        
    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, idx):
        tokens, label = self.base[idx]
        
        # Create masked version for pretraining
        masked_tokens = tokens.clone()
        mask = torch.rand(len(tokens)) < self.masking_ratio
        masked_tokens[mask] = self.base.char_to_idx["<MASK>"]
        
        # Labels are the original tokens for masked positions
        labels = tokens.clone()
        labels[~mask] = -100  # Ignore non-masked positions
        
        return masked_tokens, labels