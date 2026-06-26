# experiment_transformer.py
import torch
import wandb
import random
import numpy as np
from config_transformer import ExperimentConfig
from data_utils import LRATextDataset, LRATextDatasetSPT
from train_transformer import TransformerTrainer
from transformer import LocalTransformer
from torch.utils.data import DataLoader
import gc

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_transformer_experiment(config, use_wandb=False):
    set_seed(42)
    
    if use_wandb:
        wandb.init(project="transformer-lra-experiment", config=config.__dict__)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Clear cache before loading
    torch.cuda.empty_cache()
    gc.collect()
    
    # Load data
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
    
    # Initialize transformer model
    print("Initializing Transformer model...")
    model = LocalTransformer(
        num_tokens=config.vocab_size,
        max_seq_len=config.max_seq_len,
        dim=config.d_model,
        depth=config.n_layer,
        causal=False,  # For classification we don't need causal
        local_attn_window_size=min(128, config.max_seq_len // 4),
        dim_head=64,
        heads=8,
        ff_mult=config.expand,
        attn_dropout=0.1,
        ff_dropout=0.1,
        ignore_index=-100,
        use_xpos=True,
        xpos_scale_base=512,
        use_dynamic_pos_bias=False
    )
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / 1024**2:.2f} MB (float32)")
    
    # Create trainer
    trainer = TransformerTrainer(model, config, device)
    
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

def run_comparison_transformer():

    
    print("\n" + "="*50)
    print("Experiment: Transformer with Self-Pretraining (SPT)")
    print("="*50)
    config_spt = ExperimentConfig(
        pretrain=True,
        pretrain_epochs=1,
        num_epochs=3,
        d_model=64,
        n_layer=2,
        batch_size=16,
        max_seq_len=256
    )
    acc_spt = run_transformer_experiment(config_spt)
    
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    print(f"Transformer with SPT:    {acc_spt:.4f}")
    print("="*50)

if __name__ == "__main__":
    run_comparison_transformer()