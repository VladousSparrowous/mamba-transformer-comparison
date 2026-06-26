import torch
import wandb
import random
import numpy as np
from config import ExperimentConfig
from data_utils import LRATextDataset, LRATextDatasetSPT
from train import Trainer
from mamba import Mamba

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
    
    # Load data
    print("Loading datasets...")
    train_dataset = LRATextDataset("train", config.max_seq_len)
    val_dataset = LRATextDataset("test", config.max_seq_len)  # Use test as validation for LRA
    
    # Update vocab_size
    config.vocab_size = train_dataset.vocab_size
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Initialize model
    print("Initializing model...")
    model_args = config.get_model_args()
    model = Mamba(model_args)
    
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
            num_workers=4
        )
        trainer.pretrain(spt_loader)
    
    # Fine-tuning
    print("Starting Fine-tuning...")
    test_acc = trainer.finetune(train_loader, val_loader)
    
    if use_wandb:
        wandb.log({"final_test_acc": test_acc})
        wandb.finish()
    
    return test_acc

def run_comparison():
    """Run comparison: from-scratch vs SPT for Mamba"""
    
    # Experiment 1: Mamba from scratch
    print("\n" + "="*50)
    print("Experiment 1: Mamba trained from scratch")
    print("="*50)
    config_from_scratch = ExperimentConfig(
        pretrain=False,
        num_epochs=100
    )
    acc_scratch = run_experiment(config_from_scratch)
    
    # Experiment 2: Mamba with SPT
    print("\n" + "="*50)
    print("Experiment 2: Mamba with Self-Pretraining (SPT)")
    print("="*50)
    config_spt = ExperimentConfig(
        pretrain=True,
        pretrain_epochs=50,
        num_epochs=100
    )
    acc_spt = run_experiment(config_spt)
    
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    print(f"Mamba from scratch: {acc_scratch:.4f}")
    print(f"Mamba with SPT:     {acc_spt:.4f}")
    print(f"Improvement:        {(acc_spt - acc_scratch)*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    run_comparison()