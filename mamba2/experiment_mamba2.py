import torch
import wandb
import random
import numpy as np
from config import ExperimentConfig
from data_utils import LRATextDataset, LRATextDatasetSPT
from train_mamba2 import Trainer
from mamba2 import Mamba2LMHeadModel, Mamba2Config
from torch.utils.data import DataLoader
import gc
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_experiment(config, use_wandb=False):
    set_seed(42)
    
    if use_wandb:
        wandb.init(project="mamba2-lra-experiment", config=config.__dict__)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Clear cache before loading
    torch.cuda.empty_cache()
    gc.collect()
    
    # Load data with smaller batch size
    print("Loading datasets...")
    train_dataset = LRATextDataset("train", config.max_seq_len)
    val_dataset = LRATextDataset("test", config.max_seq_len, vocab=train_dataset.char_to_idx)
        
    # Update vocab_size in config to match dataset
    config.vocab_size = train_dataset.vocab_size
    
    # Ensure vocab_size is multiple of 16 for Mamba-2
    if config.vocab_size % 16 != 0:
        config.vocab_size += (16 - config.vocab_size % 16)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Initialize model with Mamba-2 config
    print("Initializing Mamba-2 model...")
    model_args = config.get_model_args()
    # Create model using Mamba2LMHeadModel
    model = Mamba2LMHeadModel(model_args, device=device)
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / 1024**2:.2f} MB (float32)")
    
    # Create trainer
    trainer = Trainer(model, config, device)
    
    # Self-pretraining
    if config.pretrain:
        print("Starting Self-Pretraining (SPT)...")
        spt_dataset = LRATextDatasetSPT(train_dataset, config.masking_ratio)
        spt_loader = DataLoader(
            spt_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        trainer.pretrain(spt_loader)
    
    # Fine-tuning
    print("Starting Fine-tuning...")
    test_acc = trainer.finetune(train_loader, val_loader)
    
    if use_wandb:
        wandb.log({"final_test_acc": test_acc})
        wandb.finish()
    
    # Clean up
    torch.cuda.empty_cache()
    gc.collect()
    
    return test_acc

def run_comparison():
    print("\n" + "="*50)
    print("Experiment: Mamba-2 with Self-Pretraining (SPT)")
    print("="*50)
    config_spt = ExperimentConfig(
        pretrain=True,
        pretrain_epochs=2,  # Reduced
        num_epochs=3,  # Reduced
        d_model=64,
        n_layer=1,
        d_state=8,
        batch_size=16,
        max_seq_len=512,
        headdim=32,
        chunk_size=32
    )
    acc_spt = run_experiment(config_spt)
    
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    print(f"Mamba-2 with SPT:     {acc_spt:.4f}")
    print("="*50)

if __name__ == "__main__":
    run_comparison()