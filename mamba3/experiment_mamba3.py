import torch
import wandb
import random
import numpy as np
from config import ExperimentConfig
from data_utils import LRATextDataset, LRATextDatasetSPT
from train import Trainer
from mamba3 import Mamba3LMHeadModel
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
        wandb.init(project="mamba3-lra-experiment", config=config.__dict__)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    torch.cuda.empty_cache()
    gc.collect()
    
    print("Loading datasets...")
    train_dataset = LRATextDataset("train", config.max_seq_len)
    val_dataset = LRATextDataset("test", config.max_seq_len, vocab=train_dataset.char_to_idx)
    
    config.vocab_size = train_dataset.vocab_size
    
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
    
    print("Initializing model...")
    model_args = config.get_model_args()
    model = Mamba3LMHeadModel(model_args, device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / 1024**2:.2f} MB (float32)")
    
    trainer = Trainer(model, config, device)
    
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
    
    print("Starting Fine-tuning...")
    test_acc = trainer.finetune(train_loader, val_loader)
    
    if use_wandb:
        wandb.log({"final_test_acc": test_acc})
        wandb.finish()
    
    torch.cuda.empty_cache()
    gc.collect()
    
    return test_acc

def run_comparison():
    print("\n" + "="*50)
    print("Experiment: Mamba-3 with Self-Pretraining (SPT)")
    print("="*50)
    
    config_spt = ExperimentConfig(
        pretrain=True,
        pretrain_epochs=3,
        num_epochs=5,
        d_model=64,
        n_layer=4,
        d_state=8,
        expand=2,
        headdim=64,
        chunk_size=32,
        batch_size=16,
        max_seq_len=4096
    )
    acc_spt = run_experiment(config_spt)
    
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    print(f"Mamba-3 with SPT: {acc_spt:.4f}")
    print("="*50)

if __name__ == "__main__":
    run_comparison()