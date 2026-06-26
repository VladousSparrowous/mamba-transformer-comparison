import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import wandb
import os
from mamba import Mamba, ModelArgs

class Trainer:
    def __init__(self, model, config, device="cuda"):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
    def pretrain(self, dataloader):
        """Self-pretraining with masked language modeling"""
        self.model.train()
        
        total_steps = len(dataloader) * self.config.pretrain_epochs
        scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        for epoch in range(self.config.pretrain_epochs):
            total_loss = 0
            progress_bar = tqdm(dataloader, desc=f"SPT Epoch {epoch+1}")
            
            for batch in progress_bar:
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                logits = self.model(inputs)
                
                # MLM loss (only on masked positions)
                loss = F.cross_entropy(
                    logits.view(-1, self.config.vocab_size),
                    labels.view(-1),
                    ignore_index=-100
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})
            
            avg_loss = total_loss / len(dataloader)
            print(f"SPT Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")
            
            if self.config.use_wandb:
                wandb.log({"spt_loss": avg_loss, "spt_epoch": epoch})
    
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
        patience = 10
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            progress_bar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}")
            for batch in progress_bar:
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                logits = self.model(inputs)
                
                # Classification: use CLS token or mean pooling
                # For simplicity, use mean pooling
                logits = logits.mean(dim=1)  # (batch, d_model)
                logits = self.model.lm_head(logits)  # (batch, vocab_size)
                
                # For binary classification, use first 2 logits
                logits = logits[:, :2]
                
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                pred = logits.argmax(dim=1)
                train_correct += (pred == labels).sum().item()
                train_total += labels.size(0)
                
                progress_bar.set_postfix({
                    "loss": loss.item(),
                    "acc": train_correct / train_total
                })
            
            train_acc = train_correct / train_total
            train_loss = train_loss / len(train_loader)
            
            # Validation
            val_acc = self.evaluate(val_loader)
            
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
                torch.save(self.model.state_dict(), "best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load("best_model.pt"))
        
        # Test
        if test_loader:
            test_acc = self.evaluate(test_loader)
            print(f"Test Accuracy: {test_acc:.4f}")
            return test_acc
        
        return best_val_acc
    
    def evaluate(self, dataloader):
        """Evaluate the model"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(inputs)
                logits = logits.mean(dim=1)
                logits = self.model.lm_head(logits)[:, :2]
                
                pred = logits.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        
        return correct / total