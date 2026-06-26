from dataclasses import dataclass
from mamba2 import Mamba2Config

@dataclass
class ExperimentConfig:
    # Model parameters - REDUCED for memory efficiency
    model_name: str = "mamba2"  # "mamba" or "mamba2"
    d_model: int = 64  # Reduced from 128
    n_layer: int = 2   # Reduced from 4
    d_state: int = 8   # Reduced from 16
    expand: int = 2
    headdim: int = 32  # Head dimension for Mamba-2
    chunk_size: int = 32  # Chunk size for SSD
    
    # Training parameters
    batch_size: int = 8  # Further reduced for stability
    learning_rate: float = 1e-3
    num_epochs: int = 20  # Reduced for debugging
    warmup_steps: int = 100  # Reduced
    weight_decay: float = 0.01  # Reduced from 0.1
    
    # SPT parameters
    pretrain: bool = True
    pretrain_epochs: int = 10  # Reduced
    masking_ratio: float = 0.15
    
    # Data parameters
    max_seq_len: int = 256  # Reduced from 512 for memory
    vocab_size: int = 128
    
    # Wandb
    use_wandb: bool = False
    
    def get_model_args(self):
        return Mamba2Config(
            d_model=self.d_model,
            n_layer=self.n_layer,
            vocab_size=self.vocab_size,
            d_state=self.d_state,
            expand=self.expand,
            headdim=self.headdim,
            chunk_size=self.chunk_size
        )