from dataclasses import dataclass
from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    # Model parameters
    model_name: str = "deberta"  # "mamba" or "deberta"
    d_model: int = 768  # DeBERTa base hidden size
    n_layer: int = 12   # DeBERTa base layers
    d_state: int = 8   
    expand: int = 2
    
    # DeBERTa specific
    deberta_pretrained: str = "base"  # "base", "large", "xlarge", etc.
    max_seq_len: int = 512
    
    # Training parameters
    batch_size: int = 8  
    learning_rate: float = 2e-5  # Lower learning rate for fine-tuning
    num_epochs: int = 20  
    warmup_steps: int = 100  
    weight_decay: float = 0.01  
    
    # SPT parameters
    pretrain: bool = True
    pretrain_epochs: int = 10  
    masking_ratio: float = 0.15
    
    # Data parameters
    vocab_size: int = 128
    
    # Wandb
    use_wandb: bool = False
    
    def get_model_args(self):
        # For compatibility with existing code
        return None