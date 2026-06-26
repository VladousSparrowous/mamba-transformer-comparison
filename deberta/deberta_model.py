import torch
import torch.nn as nn
import torch.nn.functional as F
from DeBERTa import deberta
import math

class DeBERTaForSequenceClassification(nn.Module):
    """DeBERTa model for sequence classification with SPT support"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Load pre-trained DeBERTa
        self.deberta = deberta.DeBERTa(pre_trained=config.deberta_pretrained)
        
        # Get the hidden size from the loaded model
        self.hidden_size = config.d_model  # 768 for base
        
        # Classification head
        self.classifier = nn.Linear(self.hidden_size, 2)
        
        # MLM head for pretraining (tied weights)
        # We'll use the DeBERTa's existing LM head if available
        if hasattr(self.deberta, 'lm_head'):
            self.lm_head = self.deberta.lm_head
        else:
            self.lm_head = nn.Linear(self.hidden_size, self.deberta.vocab_size)
        
        # Apply pre-trained weights
        self.deberta.apply_state()
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, 
                output_hidden_states=False, return_dict=False):
        """Forward pass through DeBERTa"""
        
        # DeBERTa forward
        outputs = self.deberta.bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_all_encoded_layers=False
        )
        
        # Get the last hidden state
        if isinstance(outputs, tuple):
            last_hidden_state = outputs[-1]  # DeBERTa returns (pooled_output, hidden_states)
        else:
            last_hidden_state = outputs
        
        return last_hidden_state
    
    def get_embeddings(self, input_ids, attention_mask=None, token_type_ids=None):
        """Get embeddings for pooling"""
        with torch.no_grad():
            # We don't want to update DeBERTa during SPT if we're just getting embeddings
            # but for fine-tuning we want gradients
            outputs = self.deberta.bert(
                input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                output_all_encoded_layers=False
            )
            if isinstance(outputs, tuple):
                return outputs[-1]
            return outputs
    
    def mlm_forward(self, input_ids, masked_positions=None, attention_mask=None):
        """Forward pass for MLM pretraining"""
        # Get hidden states
        hidden_states = self.forward(input_ids, attention_mask=attention_mask)
        
        # Apply LM head to get logits
        logits = self.lm_head(hidden_states)
        
        return logits
    
    def classification_forward(self, input_ids, attention_mask=None):
        """Forward pass for classification"""
        # Get hidden states
        hidden_states = self.forward(input_ids, attention_mask=attention_mask)
        
        # Pooling: use [CLS] token or mean pooling
        if attention_mask is not None:
            # Mean pooling with attention mask
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            # Use [CLS] token (first token)
            pooled = hidden_states[:, 0, :]
        
        # Classification
        logits = self.classifier(pooled)
        return logits

class DeBERTaForSPT(nn.Module):
    """Wrapper for DeBERTa that supports both SPT and classification"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = DeBERTaForSequenceClassification(config)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """Forward pass for SPT"""
        # For SPT, we return MLM logits
        return self.model.mlm_forward(input_ids, attention_mask=attention_mask)
    
    def classify(self, input_ids, attention_mask=None):
        """Classification forward pass"""
        return self.model.classification_forward(input_ids, attention_mask=attention_mask)
    
    def get_embeddings(self, input_ids, attention_mask=None):
        """Get pooled embeddings for classification"""
        hidden_states = self.model.forward(input_ids, attention_mask=attention_mask)
        
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            pooled = hidden_states[:, 0, :]
        
        return pooled

def get_deberta_tokenizer(pretrained="base"):
    """Get DeBERTa tokenizer"""
    vocab_path, vocab_type = deberta.load_vocab(pretrained_id=pretrained)
    tokenizer = deberta.tokenizers[vocab_type](vocab_path)
    return tokenizer

def tokenize_for_deberta(text, tokenizer, max_seq_len=512):
    """Tokenize text for DeBERTa"""
    # Tokenize
    tokens = tokenizer.tokenize(text)
    
    # Truncate
    tokens = tokens[:max_seq_len - 2]  # -2 for [CLS] and [SEP]
    
    # Add special tokens
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    
    # Convert to IDs
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # Create attention mask
    attention_mask = [1] * len(input_ids)
    
    # Pad
    padding_len = max_seq_len - len(input_ids)
    input_ids = input_ids + [0] * padding_len
    attention_mask = attention_mask + [0] * padding_len
    
    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
    }