from dataclasses import dataclass
from mamba import ModelArgs

@dataclass
class ExperimentConfig:
    # Model parameters
    model_name: str = "mamba"  # "mamba" or "transformer"
    d_model: int = 128
    n_layer: int = 4
    d_state: int = 16
    expand: int = 2
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 100
    warmup_steps: int = 1000
    weight_decay: float = 0.1
    
    # SPT parameters
    pretrain: bool = True
    pretrain_epochs: int = 50
    masking_ratio: float = 0.15  # 15% for text tasks
    
    # Data parameters
    max_seq_len: int = 2048
    vocab_size: int = 128  # Character-level
    
    def get_model_args(self):
        return ModelArgs(
            d_model=self.d_model,
            n_layer=self.n_layer,
            vocab_size=self.vocab_size,
            d_state=self.d_state,
            expand=self.expand
        )