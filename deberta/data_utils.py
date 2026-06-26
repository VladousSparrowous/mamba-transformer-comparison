import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from deberta_model import get_deberta_tokenizer, tokenize_for_deberta

class DeBERTaLRADataset(Dataset):
    """LRA Text dataset using DeBERTa tokenizer"""
    
    def __init__(self, split="train", max_seq_len=512, pretrained="base"):
        self.max_seq_len = max_seq_len
        
        # Load IMDb dataset
        dataset = load_dataset("imdb", split=split)
        
        # Get DeBERTa tokenizer
        self.tokenizer = get_deberta_tokenizer(pretrained)
        
        self.data = []
        self.labels = []
        self.vocab_size = len(self.tokenizer.vocab)  # Get vocab size from tokenizer
        
        for item in tqdm(dataset, desc=f"Processing {split} set"):
            # Tokenize with DeBERTa tokenizer
            tokens = tokenize_for_deberta(item["text"], self.tokenizer, max_seq_len)
            self.data.append(tokens)
            self.labels.append(item["label"])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return item['input_ids'], item['attention_mask'], self.labels[idx]

class DeBERTaLRADatasetSPT(Dataset):
    """Dataset for Self-Pretraining (SPT) with masking for DeBERTa"""
    
    def __init__(self, base_dataset, masking_ratio=0.15):
        self.base = base_dataset
        self.masking_ratio = masking_ratio
        self.vocab_size = base_dataset.vocab_size
        
        # Get mask token ID from tokenizer
        self.mask_token_id = 103  # DeBERTa uses same as BERT: [MASK] = 103
        self.pad_token_id = 0
        
    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, idx):
        input_ids, attention_mask, label = self.base[idx]
        
        # Create masked version
        masked_input_ids = input_ids.clone()
        labels = input_ids.clone()
        
        # Randomly mask tokens
        mask = torch.rand(len(input_ids)) < self.masking_ratio
        
        # Only mask actual tokens (not padding, not special tokens)
        mask = mask & (attention_mask == 1)
        mask = mask & (input_ids != self.pad_token_id)
        mask = mask & (input_ids != self.mask_token_id)
        # Don't mask [CLS] and [SEP] tokens (101 and 102 in BERT/DeBERTa)
        mask = mask & (input_ids != 101)  # [CLS]
        mask = mask & (input_ids != 102)  # [SEP]
        
        # Apply mask
        masked_input_ids[mask] = self.mask_token_id
        
        # Labels: only keep original tokens for masked positions
        labels[~mask] = -100  # Ignore in loss
        
        return masked_input_ids, attention_mask, labels