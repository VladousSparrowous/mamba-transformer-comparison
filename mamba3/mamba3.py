"""
mamba3-minimal - исправленная версия с оптимизациями памяти
"""

import math
from dataclasses import dataclass
from typing import cast
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

Device = str | torch.device | None


@dataclass
class Mamba3Config:
    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    headdim: int = 32
    chunk_size: int = 32
    d_conv: int = 4
    ngroups: int = 1
    rope_fraction: float = 0.5
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    A_floor: float = 1e-4
    pad_vocab_size_multiple: int = 8

    def __post_init__(self):
        self.d_inner = self.expand * self.d_model
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (
                self.pad_vocab_size_multiple
                - self.vocab_size % self.pad_vocab_size_multiple
            )
        
        self.rotary_dim = int(self.d_state * self.rope_fraction)
        if self.rotary_dim % 2 != 0:
            self.rotary_dim -= 1
        self.num_rope_angles = self.rotary_dim // 2
        assert self.num_rope_angles > 0


class InferenceCache:
    def __init__(self, conv_state: Tensor, ssm_state: Tensor, angle_state: Tensor):
        self.conv_state = conv_state
        self.ssm_state = ssm_state
        self.angle_state = angle_state

    @staticmethod
    def alloc(batch_size: int, args: Mamba3Config, device: Device = None):
        return InferenceCache(
            torch.zeros(
                batch_size, args.d_inner + 2 * args.d_state, args.d_conv, device=device
            ),
            torch.zeros(
                batch_size, args.nheads, args.headdim, args.d_state, device=device
            ),
            torch.zeros(
                batch_size, args.nheads, args.num_rope_angles, device=device
            ),
        )


class Mamba3LMHeadModel(nn.Module):
    def __init__(self, args: Mamba3Config, device: Device = None):
        super().__init__()
        self.args = args
        self.device = device

        self.embedding = nn.Embedding(args.vocab_size, args.d_model, device=device)
        self.layers = nn.ModuleList([
            Mamba3Block(args, device=device) for _ in range(args.n_layer)
        ])
        self.norm_f = RMSNorm(args.d_model, device=device)
        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False, device=device)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids: Tensor) -> Tensor:
        """Forward pass without cache (training mode)"""
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits


class Mamba3Block(nn.Module):
    """Mamba-3 block with residual connection"""
    def __init__(self, args: Mamba3Config, device: Device = None):
        super().__init__()
        self.args = args
        self.mixer = Mamba3(args, device=device)
        self.norm = RMSNorm(args.d_model, device=device)

    def forward(self, x: Tensor) -> Tensor:
        output = self.mixer(self.norm(x)) + x
        return output


class Mamba3(nn.Module):
    def __init__(self, args: Mamba3Config, device: Device = None):
        super().__init__()
        self.args = args
        self.device = device
        
        self.num_bc_heads = args.ngroups
        
        # Размерность проекций
        d_in_proj = (
            2 * args.d_inner +
            2 * args.d_state * self.num_bc_heads +
            args.nheads +
            args.nheads +
            args.nheads +
            args.num_rope_angles
        )
        self.in_proj = nn.Linear(args.d_model, d_in_proj, bias=False, device=device)
        
        # dt_bias
        _dt = torch.exp(
            torch.rand(args.nheads, device=device) * (math.log(args.dt_max) - math.log(args.dt_min))
            + math.log(args.dt_min)
        )
        _dt = torch.clamp(_dt, min=args.dt_init_floor)
        _dt_bias = _dt + torch.log(-torch.expm1(-_dt))
        self.dt_bias = nn.Parameter(_dt_bias)
        
        # Biases
        self.B_bias = nn.Parameter(torch.zeros((args.nheads, args.d_state), device=device))
        self.C_bias = nn.Parameter(torch.zeros((args.nheads, args.d_state), device=device))
        self.D = nn.Parameter(torch.ones(args.nheads, device=device))
        
        # Conv1d for B and C
        conv_dim = args.d_inner + 2 * args.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=args.d_conv,
            groups=conv_dim,
            padding=args.d_conv - 1,
            device=device,
        )
        
        self.B_norm = RMSNorm(args.d_state, device=device)
        self.C_norm = RMSNorm(args.d_state, device=device)
        self.norm = RMSNorm(args.d_inner, device=device)
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=False, device=device)

    def forward(self, u: Tensor) -> Tensor:
        """Forward pass without cache"""
        batch, seqlen, _ = u.shape
        
        # In-proj
        zxBCdtAtrap = self.in_proj(u)
        z, x, B, C, dt, A, trap, angles = torch.split(
            zxBCdtAtrap,
            [
                self.args.d_inner,
                self.args.d_inner,
                self.args.d_state * self.num_bc_heads,
                self.args.d_state * self.num_bc_heads,
                self.args.nheads,
                self.args.nheads,
                self.args.nheads,
                self.args.num_rope_angles,
            ],
            dim=-1,
        )
        
        # Reshape
        z = rearrange(z, "b l (h p) -> b l h p", p=self.args.headdim)
        x = rearrange(x, "b l (h p) -> b l h p", p=self.args.headdim)
        B = rearrange(B, "b l (g n) -> b l g n", g=self.num_bc_heads)
        C = rearrange(C, "b l (g n) -> b l g n", g=self.num_bc_heads)
        
        # Expand B and C to nheads
        if self.num_bc_heads < self.args.nheads:
            B = B.expand(-1, -1, self.args.nheads, -1)
            C = C.expand(-1, -1, self.args.nheads, -1)
        
        # DT
        dt = F.softplus(dt + self.dt_bias)
        
        # Normalize B and C
        B = self.B_norm(B)
        C = self.C_norm(C)
        
        # Add biases
        B = B + self.B_bias
        C = C + self.C_bias
        
        # Apply RoPE if angles present
        if angles is not None:
            B, C = self._apply_rotary(B, C, angles)
        
        # SSD
        y = self._ssd_with_rope(x, dt, A, B, C, trap)
        
        # D skip connection
        y = y + x * self.D.unsqueeze(-1)
        
        # Output
        y = rearrange(y, "b l h p -> b l (h p)")
        y = self.norm(y, z)
        y = self.out_proj(y)
        
        return y

    def _apply_rotary(self, B: Tensor, C: Tensor, angles: Tensor) -> tuple[Tensor, Tensor]:
        """Apply RoPE to B and C"""
        d_state = B.shape[-1]
        num_angles = angles.shape[-1]
        half_dim = min(num_angles, d_state // 2)
        
        # Apply to first half_dim pairs
        if half_dim > 0:
            cos = torch.cos(angles[..., :half_dim])
            sin = torch.sin(angles[..., :half_dim])
            
            # B
            B_rot = B[..., :2*half_dim]
            B_rot = rearrange(B_rot, "... (p d) -> ... p d", p=2)
            B_rot = torch.stack([
                B_rot[..., 0] * cos - B_rot[..., 1] * sin,
                B_rot[..., 0] * sin + B_rot[..., 1] * cos
            ], dim=-2)
            B_rot = rearrange(B_rot, "... p d -> ... (p d)")
            B = torch.cat([B_rot, B[..., 2*half_dim:]], dim=-1)
            
            # C
            C_rot = C[..., :2*half_dim]
            C_rot = rearrange(C_rot, "... (p d) -> ... p d", p=2)
            C_rot = torch.stack([
                C_rot[..., 0] * cos - C_rot[..., 1] * sin,
                C_rot[..., 0] * sin + C_rot[..., 1] * cos
            ], dim=-2)
            C_rot = rearrange(C_rot, "... p d -> ... (p d)")
            C = torch.cat([C_rot, C[..., 2*half_dim:]], dim=-1)
        
        return B, C

    def _ssd_with_rope(self, x: Tensor, dt: Tensor, A: Tensor, B: Tensor, C: Tensor, trap: Tensor) -> Tensor:
        """Simplified SSD implementation"""
        batch, seqlen, nheads, headdim = x.shape
        d_state = B.shape[-1]
        
        # Apply heavy-tail activation to A
        A = -self._heavy_tail(A)
        A = torch.clamp(A, max=-1e-4)
        
        # Prepare for chunked computation
        chunk_size = self.args.chunk_size
        n_chunks = (seqlen + chunk_size - 1) // chunk_size
        padded_len = n_chunks * chunk_size
        
        if seqlen < padded_len:
            pad_size = padded_len - seqlen
            x = F.pad(x, (0, 0, 0, 0, 0, pad_size))
            dt = F.pad(dt, (0, 0, 0, pad_size))
            A = F.pad(A, (0, 0, 0, pad_size))
            B = F.pad(B, (0, 0, 0, 0, 0, pad_size))
            C = F.pad(C, (0, 0, 0, 0, 0, pad_size))
        
        # Process in chunks
        y_chunks = []
        state = torch.zeros(batch, nheads, headdim, d_state, device=x.device)
        
        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, padded_len)
            
            x_chunk = x[:, start:end]
            dt_chunk = dt[:, start:end]
            A_chunk = A[:, start:end]
            B_chunk = B[:, start:end]
            C_chunk = C[:, start:end]
            
            # Compute chunk outputs
            y_chunk, state = self._ssm_chunk(x_chunk, dt_chunk, A_chunk, B_chunk, C_chunk, state)
            y_chunks.append(y_chunk)
        
        y = torch.cat(y_chunks, dim=1)
        return y[:, :seqlen]

    def _ssm_chunk(self, x: Tensor, dt: Tensor, A: Tensor, B: Tensor, C: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        """Process one chunk of the sequence"""
        batch, seqlen, nheads, headdim = x.shape
        d_state = B.shape[-1]
        
        # Discretize
        dt = dt.unsqueeze(-1)  # (b, l, h, 1)
        dA = torch.exp(dt * A)  # (b, l, h, 1) * (b, l, h, 1) -> (b, l, h, 1)
        dB = dt * B  # (b, l, h, 1) * (b, l, h, d_state) -> (b, l, h, d_state)
        
        outputs = []
        current_state = state.clone()
        
        for t in range(seqlen):
            # Update state
            current_state = dA[:, t, :, None] * current_state + dB[:, t, :, :] * x[:, t, :, :, None]
            
            # Compute output
            y_t = torch.einsum('bhp,bhpd->bhd', C[:, t], current_state)
            outputs.append(y_t)
        
        # Stack outputs
        y = torch.stack(outputs, dim=1)
        
        return y, current_state

    @staticmethod
    def _heavy_tail(x: Tensor) -> Tensor:
        """Heavy-tail activation"""
        neg = x.clamp_max(0)
        pos = x.clamp_min(0)
        return pos + torch.reciprocal(1 - neg)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5, device: Device = None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d, device=device))

    def forward(self, x: Tensor, z: Tensor = None) -> Tensor:
        if z is not None:
            x = x * F.silu(z)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight