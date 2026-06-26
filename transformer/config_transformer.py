# config_transformer.py (optional - if you want separate config for transformer)
from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    # Model parameters
    model_name: str = "transformer"
    d_model: int = 128
    n_layer: int = 4
    expand: int = 2
    heads: int = 8
    dim_head: int = 64
    local_attn_window_size: int = 64
    
    # Training parameters
    batch_size: int = 8
    learning_rate: float = 1e-3
    num_epochs: int = 10
    warmup_steps: int = 100
    weight_decay: float = 0.01
    
    # SPT parameters
    pretrain: bool = True
    pretrain_epochs: int = 5
    masking_ratio: float = 0.15
    
    # Data parameters
    max_seq_len: int = 256
    vocab_size: int = 128
    
    # Wandb
    use_wandb: bool = False