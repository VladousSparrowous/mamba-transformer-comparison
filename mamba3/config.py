from dataclasses import dataclass
from mamba3 import Mamba3Config

@dataclass
class ExperimentConfig:
    # Model parameters - оптимизировано для памяти
    model_name: str = "mamba3"
    d_model: int = 64
    n_layer: int = 2
    d_state: int = 8
    expand: int = 2
    headdim: int = 32  # Уменьшено с 64
    chunk_size: int = 32  # Уменьшено с 64
    ngroups: int = 1
    
    # Training parameters
    batch_size: int = 8
    learning_rate: float = 3e-4
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
    
    def get_model_args(self):
        return Mamba3Config(
            d_model=self.d_model,
            n_layer=self.n_layer,
            vocab_size=self.vocab_size,
            d_state=self.d_state,
            expand=self.expand,
            headdim=self.headdim,
            chunk_size=self.chunk_size,
            ngroups=self.ngroups
        )