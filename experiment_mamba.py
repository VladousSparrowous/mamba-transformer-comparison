import torch
import wandb
import random
import numpy as np
from config import ExperimentConfig
from data_utils import LRATextDataset, LRATextDatasetSPT
from train import Trainer
from mamba import Mamba
from torch.utils.data import DataLoader
import gc

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_experiment(config, use_wandb=False):
    set_seed(42)
    
    if use_wandb:
        wandb.init(project="mamba-lra-experiment", config=config.__dict__)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Clear cache before loading
    torch.cuda.empty_cache()
    gc.collect()
    
    # Load data with smaller batch size
    print("Loading datasets...")
    train_dataset = LRATextDataset("train", config.max_seq_len)
    val_dataset = LRATextDataset("test", config.max_seq_len)
    
    # Update vocab_size in config to match dataset
    config.vocab_size = train_dataset.vocab_size
    
    # Ensure vocab_size is multiple of 8 for efficiency
    if config.vocab_size % 8 != 0:
        config.vocab_size += (8 - config.vocab_size % 8)
    
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
    
    # Initialize model with smaller config
    print("Initializing model...")
    model_args = config.get_model_args()
    model = Mamba(model_args)
    
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
    print("Experiment 2: Mamba with Self-Pretraining (SPT)")
    print("="*50)
    config_spt = ExperimentConfig(
        pretrain=True,
        pretrain_epochs=2,  # Reduced
        num_epochs=3,  # Reduced
        d_model=64,
        n_layer=2,
        d_state=8,
        batch_size=8,
        max_seq_len=4096
    )
    acc_spt = run_experiment(config_spt)
    
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    print(f"Mamba with SPT:     {acc_spt:.4f}")
    print("="*50)

if __name__ == "__main__":
    run_comparison()