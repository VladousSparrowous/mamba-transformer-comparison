import torch
import wandb
import random
import numpy as np
from config import ExperimentConfig
from data_utils import DeBERTaLRADataset, DeBERTaLRADatasetSPT
from train_deberta import DeBERTaTrainer
from deberta_model import DeBERTaForSPT, get_deberta_tokenizer
from torch.utils.data import DataLoader
import gc
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_experiment(config, use_wandb=False):
    set_seed(42)
    
    if use_wandb:
        wandb.init(project="deberta-lra-experiment", config=config.__dict__)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Clear cache before loading
    torch.cuda.empty_cache()
    gc.collect()
    
    # Load data with DeBERTa tokenizer
    print("Loading datasets...")
    train_dataset = DeBERTaLRADataset("train", config.max_seq_len, config.deberta_pretrained)
    val_dataset = DeBERTaLRADataset("test", config.max_seq_len, config.deberta_pretrained)
    
    # Update vocab_size in config
    config.vocab_size = train_dataset.vocab_size
    
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
    
    # Initialize DeBERTa model
    print("Initializing DeBERTa model...")
    model = DeBERTaForSPT(config)
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024**2:.2f} MB (float32)")
    
    # Create trainer
    trainer = DeBERTaTrainer(model, config, device)
    
    # Self-pretraining (optional)
    if config.pretrain:
        print("Starting Self-Pretraining (SPT)...")
        spt_dataset = DeBERTaLRADatasetSPT(train_dataset, config.masking_ratio)
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
    """Run comparison: from-scratch vs SPT for DeBERTa"""
    
    # Experiment 1: DeBERTa from scratch (fine-tuning only)
    print("\n" + "="*50)
    print("Experiment 1: DeBERTa fine-tuned from pre-trained")
    print("="*50)
    config_from_scratch = ExperimentConfig(
        pretrain=False,  # No SPT, just fine-tuning
        num_epochs=20,
        deberta_pretrained="base",
        d_model=768,  # DeBERTa base
        batch_size=8,
        max_seq_len=512,
        learning_rate=2e-5,  # Standard fine-tuning LR
        warmup_steps=100
    )
    acc_scratch = run_experiment(config_from_scratch)
    
    # Clear memory between experiments
    torch.cuda.empty_cache()
    gc.collect()
    
    # Experiment 2: DeBERTa with SPT
    print("\n" + "="*50)
    print("Experiment 2: DeBERTa with Self-Pretraining (SPT)")
    print("="*50)
    config_spt = ExperimentConfig(
        pretrain=True,
        pretrain_epochs=10,
        num_epochs=20,
        deberta_pretrained="base",
        d_model=768,
        batch_size=8,
        max_seq_len=512,
        learning_rate=2e-5,
        warmup_steps=100
    )
    acc_spt = run_experiment(config_spt)
    
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    print(f"DeBERTa fine-tuning only: {acc_scratch:.4f}")
    print(f"DeBERTa with SPT:         {acc_spt:.4f}")
    print(f"Improvement:             {(acc_spt - acc_scratch)*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    run_comparison()