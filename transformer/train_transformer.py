# train_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import wandb
import os
import gc

class TransformerTrainer:
    def __init__(self, model, config, device="cuda"):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Set vocab size in config to match model
        self.config.vocab_size = model.token_emb.num_embeddings
        
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Classification head for finetuning
        self.classification_head = nn.Linear(config.d_model, 2).to(device)
        
    def pretrain(self, dataloader):
        """Self-pretraining with masked language modeling"""
        self.model.train()
        
        total_steps = len(dataloader) * self.config.pretrain_epochs
        scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Use gradient accumulation for memory efficiency
        accumulation_steps = 2
        
        for epoch in range(self.config.pretrain_epochs):
            total_loss = 0
            progress_bar = tqdm(dataloader, desc=f"SPT Epoch {epoch+1}")
            self.optimizer.zero_grad()
            
            for step, batch in enumerate(progress_bar):
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass - get logits from transformer
                logits = self.model(inputs)
                
                # MLM loss (only on masked positions)
                loss = F.cross_entropy(
                    logits.view(-1, self.config.vocab_size),
                    labels.view(-1),
                    ignore_index=-100
                )
                
                loss = loss / accumulation_steps
                loss.backward()
                
                if (step + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    scheduler.step()
                    self.optimizer.zero_grad()
                
                total_loss += loss.item() * accumulation_steps
                progress_bar.set_postfix({"loss": loss.item() * accumulation_steps})
                
                # Clear cache periodically
                if step % 50 == 0:
                    torch.cuda.empty_cache()
            
            avg_loss = total_loss / len(dataloader)
            print(f"SPT Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")
            
            if self.config.use_wandb:
                wandb.log({"spt_loss": avg_loss, "spt_epoch": epoch})
            
            # Clear cache after epoch
            torch.cuda.empty_cache()
            gc.collect()
    
    def finetune(self, train_loader, val_loader, test_loader=None):
        """Fine-tune for classification"""
        self.model.train()
        
        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        best_val_acc = 0
        patience = 5
        patience_counter = 0
        
        # Use gradient accumulation for memory efficiency
        accumulation_steps = 2
        
        for epoch in range(self.config.num_epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            progress_bar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}")
            self.optimizer.zero_grad()
            
            for step, batch in enumerate(progress_bar):
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Get sequence representations from transformer
                # Use the CLS token or mean pooling
                with torch.set_grad_enabled(True):
                    logits_full = self.model(inputs)  # (batch, seq_len, vocab_size)
                    
                    # Use mean pooling or first token for classification
                    # Option 1: Mean pooling
                    pooled = logits_full.mean(dim=1)  # (batch, d_model)
                    
                    # Option 2: Use first token (like BERT)
                    # pooled = logits_full[:, 0, :]
                    
                    logits = self.classification_head(pooled)  # (batch, 2)
                
                loss = F.cross_entropy(logits, labels)
                loss = loss / accumulation_steps
                loss.backward()
                
                if (step + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    scheduler.step()
                    self.optimizer.zero_grad()
                
                train_loss += loss.item() * accumulation_steps
                pred = logits.argmax(dim=1)
                train_correct += (pred == labels).sum().item()
                train_total += labels.size(0)
                
                progress_bar.set_postfix({
                    "loss": loss.item() * accumulation_steps,
                    "acc": train_correct / train_total
                })
                
                # Clear cache periodically
                if step % 50 == 0:
                    torch.cuda.empty_cache()
            
            train_acc = train_correct / train_total
            train_loss = train_loss / len(train_loader)
            
            # Validation
            val_acc, val_loss = self.evaluate(val_loader, return_loss=True)
            
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")
            
            if self.config.use_wandb:
                wandb.log({
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "epoch": epoch
                })
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'classification_head': self.classification_head.state_dict()
                }, "best_transformer.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Clear cache after epoch
            torch.cuda.empty_cache()
            gc.collect()
        
        # Load best model
        if os.path.exists("best_transformer.pt"):
            checkpoint = torch.load("best_transformer.pt")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.classification_head.load_state_dict(checkpoint['classification_head'])
        
        # Test
        if test_loader:
            test_acc, _ = self.evaluate(test_loader, return_loss=True)
            print(f"Test Accuracy: {test_acc:.4f}")
            return test_acc
        
        return best_val_acc
    
    def evaluate(self, dataloader, return_loss=False):
        """Evaluate the model"""
        self.model.eval()
        self.classification_head.eval()
        correct = 0
        total = 0
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Get logits from transformer
                logits_full = self.model(inputs)
                
                # Mean pooling or first token
                pooled = logits_full.mean(dim=1)  # (batch, d_model)
                logits = self.classification_head(pooled)  # (batch, 2)
                
                # Calculate loss
                if return_loss:
                    loss = F.cross_entropy(logits, labels)
                    total_loss += loss.item()
                
                pred = logits.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
                
                # Clear cache
                torch.cuda.empty_cache()
        
        accuracy = correct / total
        
        if return_loss:
            return accuracy, total_loss / len(dataloader)
        return accuracy