# это mamba3_base.py

"""
mamba3-minimal

Минимальная, однофайловая реализация модели Mamba-3 в PyTorch.

"""

import math
from dataclasses import dataclass
from typing import Iterable, NamedTuple, TypeAlias, cast

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import LongTensor, Tensor, nn
from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined

Device: TypeAlias = str | torch.device | None


@dataclass
class Mamba3Config:
    d_model: int  # размерность модели (D)
    n_layer: int = 24  # количество слоёв Mamba-3
    d_state: int = 128  # размерность состояния (N)
    d_conv: int = 4  # размер ядра свёртки
    expand: int = 2  # коэффициент расширения (E)
    headdim: int = 64  # размерность головы (P)
    chunk_size: int = 64  # размер чанка для SSD
    vocab_size: int = 50277
    pad_vocab_size_multiple: int = 16
    
    # Mamba-3 специфичные параметры
    rope_fraction: float = 0.5  # доля размерности для RoPE
    ngroups: int = 1  # количество групп для B и C
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    A_floor: float = 1e-4

    def __post_init__(self):
        self.d_inner = self.expand * self.d_model
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (
                self.pad_vocab_size_multiple
                - self.vocab_size % self.pad_vocab_size_multiple
            )
        
        # Вычисляем размерности для RoPE
        self.rotary_dim = int(self.d_state * self.rope_fraction)
        if self.rotary_dim % 2 != 0:
            self.rotary_dim -= 1
        self.num_rope_angles = self.rotary_dim // 2
        assert self.num_rope_angles > 0


class InferenceCache(NamedTuple):
    conv_state: Tensor  # (batch, d_inner + 2 * d_state, d_conv)
    ssm_state: Tensor  # (batch, nheads, headdim, d_state)
    angle_state: Tensor  # (batch, nheads, num_rope_angles)

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

        self.backbone = nn.ModuleDict(
            dict(
                embedding=nn.Embedding(args.vocab_size, args.d_model, device=device),
                layers=nn.ModuleList(
                    [
                        nn.ModuleDict(
                            dict(
                                mixer=Mamba3(args, device=device),
                                norm=RMSNorm(args.d_model, device=device),
                            )
                        )
                        for _ in range(args.n_layer)
                    ]
                ),
                norm_f=RMSNorm(args.d_model, device=device),
            )
        )
        self.lm_head = nn.Linear(
            args.d_model, args.vocab_size, bias=False, device=device
        )
        self.lm_head.weight = self.backbone.embedding.weight

    def forward(
        self, input_ids: LongTensor, h: list[InferenceCache] | list[None] | None = None
    ) -> tuple[LongTensor, list[InferenceCache]]:
        seqlen = input_ids.shape[1]

        if h is None:
            h = [None for _ in range(self.args.n_layer)]

        x = self.backbone.embedding(input_ids)
        for i, layer in enumerate(self.backbone.layers):
            y, h[i] = layer.mixer(layer.norm(x), h[i])
            x = y + x

        x = self.backbone.norm_f(x)
        logits = self.lm_head(x)
        return logits[:, :seqlen], cast(list[InferenceCache], h)


def heavy_tail_activation(x: Tensor) -> Tensor:
    """
    Heavy-tail activation для data-dependent A.
    f(x) = 1 + x        if x >= 0
         = 1 / (1 - x)  if x < 0
    """
    neg = x.clamp_max(0)
    pos = x.clamp_min(0)
    return pos + torch.reciprocal(1 - neg)


def apply_rotary_embedding(q: Tensor, k: Tensor, angles: Tensor) -> tuple[Tensor, Tensor]:
    """
    Применяет RoPE к q и k.
    Поддерживает:
      - q, k: (batch, seq_len, heads, dim) или (batch, heads, dim)
      - angles: (batch, seq_len, num_angles), (batch, num_angles) или (batch, heads, num_angles)
    """
    dim = q.shape[-1]
    num_angles = angles.shape[-1]
    half_dim = min(num_angles, dim // 2)

    # Берём первые 2*half_dim элементов
    q_rot = q[..., :2 * half_dim]
    k_rot = k[..., :2 * half_dim]

    # Разбиваем на пары
    q_rot = rearrange(q_rot, "... (p d) -> ... p d", p=2)
    k_rot = rearrange(k_rot, "... (p d) -> ... p d", p=2)

    # Вычисляем cos и sin для нужного числа углов
    cos = angles[..., :half_dim].cos()
    sin = angles[..., :half_dim].sin()

    # Добавляем размерность для heads, если её нет в angles
    if q.dim() == 4 and angles.dim() == 3:
        # q: (batch, seq_len, heads, dim), angles: (batch, seq_len, num_angles)
        cos = cos.unsqueeze(-2)  # -> (batch, seq_len, 1, half_dim)
        sin = sin.unsqueeze(-2)
    elif q.dim() == 3 and angles.dim() == 2:
        # q: (batch, heads, dim), angles: (batch, num_angles)
        cos = cos.unsqueeze(1)   # -> (batch, 1, half_dim)
        sin = sin.unsqueeze(1)
    # Если q.dim() == 3 и angles.dim() == 3 (batch, heads, num_angles) – оставляем как есть

    # Применяем поворот
    q_rot = torch.stack([
        q_rot[..., 0] * cos - q_rot[..., 1] * sin,
        q_rot[..., 0] * sin + q_rot[..., 1] * cos
    ], dim=-2)
    k_rot = torch.stack([
        k_rot[..., 0] * cos - k_rot[..., 1] * sin,
        k_rot[..., 0] * sin + k_rot[..., 1] * cos
    ], dim=-2)

    # Собираем обратно
    q_rot = rearrange(q_rot, "... p d -> ... (p d)")
    k_rot = rearrange(k_rot, "... p d -> ... (p d)")

    # Объединяем с неизменённой частью
    q = torch.cat([q_rot, q[..., 2 * half_dim:]], dim=-1)
    k = torch.cat([k_rot, k[..., 2 * half_dim:]], dim=-1)

    return q, k


def segsum(x: Tensor, device: Device = None) -> Tensor:
    """Stable segment sum calculation."""
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd_with_rope(x, A, B, C, angles, A_bias, B_bias, chunk_size, rotary_dim, device: Device = None):
    """
    SSD с опциональным RoPE и biases для Mamba-3.
    Если biases или angles равны None, соответствующие операции пропускаются.
    """
    batch, seqlen, nheads, headdim = x.shape
    d_state = B.shape[-1]
    
    # Применяем heavy-tail активацию к A
    A = -heavy_tail_activation(A)
    A = torch.clamp(A, max=-1e-4)  # A_floor
    
    # Добавляем biases, если они переданы
    if B_bias is not None:
        B = B + B_bias
    if A_bias is not None:
        C = C + A_bias
    # Применяем RoPE, если передан angles
    if angles is not None:
        B, C = apply_rotary_embedding(B, C, angles)
    
    # Rearrange в чанки
    x, A, B, C = [
        rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size) 
        for m in (x, A, B, C)
    ]
    
    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)
    
    # 1. Внутричанковые вычисления (диагональные блоки)
    L = torch.exp(segsum(A, device=device))
    Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)
    
    # 2. Состояния для каждого чанка
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)
    
    # 3. Межчанковая рекуррентность SSM
    initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0)), device=device))
    new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]
    
    # 4. Преобразование состояние -> выход
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)
    
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    
    return Y, final_state


class Mamba3(nn.Module):
    def __init__(self, args: Mamba3Config, device: Device = None):
        super().__init__()
        self.args = args
        self.device = device
        
        # Количество голов для B и C
        self.num_bc_heads = args.ngroups
        
        # Размерность проекций: [z, x, B, C, dt, A, trap, angles]
        d_in_proj = (
            2 * args.d_inner +  # z и x
            2 * args.d_state * self.num_bc_heads +  # B и C
            args.nheads +  # dt
            args.nheads +  # A
            args.nheads +  # trap
            args.num_rope_angles  # angles
        )
        self.in_proj = nn.Linear(args.d_model, d_in_proj, bias=False, device=device)

        self.A_floor = args.A_floor
        
        # dt_bias
        _dt = torch.exp(
            torch.rand(args.nheads, device=device) * (math.log(args.dt_max) - math.log(args.dt_min))
            + math.log(args.dt_min)
        )
        _dt = torch.clamp(_dt, min=args.dt_init_floor)
        _dt_bias = _dt + torch.log(-torch.expm1(-_dt))
        self.dt_bias = nn.Parameter(_dt_bias)
        self.dt_bias._no_weight_decay = True
        
        # Biases для B и C
        self.B_bias = nn.Parameter(torch.zeros((args.nheads, args.d_state), device=device))
        self.C_bias = nn.Parameter(torch.zeros((args.nheads, args.d_state), device=device))
        
        # D "skip" параметр
        self.D = nn.Parameter(torch.ones(args.nheads, device=device))
        self.D._no_weight_decay = True
        
        # Свёртка для B и C (как в Mamba-2)
        conv_dim = args.d_inner + 2 * args.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=args.d_conv,
            groups=conv_dim,
            padding=args.d_conv - 1,
            device=device,
        )
        
        # RMS Norm для B и C
        self.B_norm = RMSNorm(args.d_state, device=device)
        self.C_norm = RMSNorm(args.d_state, device=device)
        
        # Выходная нормализация и проекция
        self.norm = RMSNorm(args.d_inner, device=device)
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=False, device=device)
        
        # Для совместимости с Mamba-3 оригиналом
        self.is_mimo = False  # Упрощаем - только SISO
        self.mimo_rank = 1

    def forward(self, u: Tensor, h: InferenceCache | None = None) -> tuple[Tensor, InferenceCache]:
        """
        Прямой проход с использованием оригинального ядра mamba3_siso_combined.
        """
        batch, seqlen, _ = u.shape

        if h is not None:
            # Если передан кэш, используем пошаговый режим (step)
            return self.step(u, h)

        # 1. In-projection
        zxBCdtAtrap = self.in_proj(u)  # (batch, seqlen, d_in_proj)
        z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
            zxBCdtAtrap,
            [
                self.args.d_inner,                          # z
                self.args.d_inner,                          # x
                self.args.d_state * self.num_bc_heads,      # B
                self.args.d_state * self.num_bc_heads,      # C
                self.args.nheads,                           # dt
                self.args.nheads,                           # A
                self.args.nheads,                           # trap
                self.args.num_rope_angles,                  # angles
            ],
            dim=-1,
        )

        # 2. Преобразование размерностей для ядра
        # Ядро ожидает:
        #   Q = C: (batch, seqlen, nheads, d_state)  (но у нас headdim?)
        #   K = B: (batch, seqlen, nheads, d_state)
        #   V = x: (batch, seqlen, nheads, headdim)
        #   ADT: (batch, nheads, seqlen)   (A*dt)
        #   DT:  (batch, nheads, seqlen)   (dt)
        #   Trap: (batch, nheads, seqlen)
        #   Z: (batch, seqlen, nheads, headdim) - для gating
        #   D: (nheads,)
        #   Q_bias, K_bias: (nheads, d_state)
        #   Angles: (batch, seqlen, nheads, num_rope_angles) или (batch, seqlen, num_rope_angles)

        # Reshape
        z = rearrange(z, "b l (h p) -> b l h p", p=self.args.headdim)
        x = rearrange(x, "b l (h p) -> b l h p", p=self.args.headdim)
        B = rearrange(B, "b l (g n) -> b l g n", g=self.num_bc_heads)
        C = rearrange(C, "b l (g n) -> b l g n", g=self.num_bc_heads)

        # Расширяем B и C до nheads, если ngroups < nheads
        if self.num_bc_heads < self.args.nheads:
            B = B.expand(-1, -1, self.args.nheads, -1)
            C = C.expand(-1, -1, self.args.nheads, -1)
        else:
            B = B.squeeze(2)   # если ngroups == nheads, убираем размерность группы
            C = C.squeeze(2)

        # Нормализация B и C
        B = self.B_norm(B)
        C = self.C_norm(C)

        # Добавляем biases (с начальным значением 1)
        B = B + self.B_bias
        C = C + self.C_bias

        # Вычисляем dt, A, trap
        dt = F.softplus(dd_dt + self.dt_bias)          # (batch, seqlen, nheads)
        A = -heavy_tail_activation(dd_A)               # (batch, seqlen, nheads)
        A = torch.clamp(A, max=-self.A_floor)
        trap = torch.sigmoid(trap)                     # (batch, seqlen, nheads)

        # Перестановка для ядра: (batch, nheads, seqlen)
        dt = rearrange(dt, "b l h -> b h l")
        A = rearrange(A, "b l h -> b h l")
        trap = rearrange(trap, "b l h -> b h l")
        ADT = A * dt   # (batch, nheads, seqlen)

        # Углы (angles) для RoPE: расширяем до nheads
        angles = angles.unsqueeze(-2).expand(-1, -1, self.args.nheads, -1)  # (batch, seqlen, nheads, num_rope_angles)

        # 3. Вызов оригинального ядра SISO
        y = mamba3_siso_combined(
            Q=C,                      # (batch, seqlen, nheads, d_state)
            K=B,                      # (batch, seqlen, nheads, d_state)
            V=x,                      # (batch, seqlen, nheads, headdim)
            ADT=ADT,                  # (batch, nheads, seqlen)
            DT=dt,                    # (batch, nheads, seqlen)
            Trap=trap,                # (batch, nheads, seqlen)
            Q_bias=self.C_bias,       # (nheads, d_state)
            K_bias=self.B_bias,       # (nheads, d_state)
            Angles=angles,            # (batch, seqlen, nheads, num_rope_angles)
            D=self.D,                 # (nheads,)
            Z=z,                      # (batch, seqlen, nheads, headdim) - для gating
            chunk_size=self.args.chunk_size,
            Input_States=None,        # для инференса
            return_final_states=False,
            cu_seqlens=None,
        )
            # Если return_final_states=False, y - единственный выход
            # y имеет форму (batch, seqlen, nheads, headdim)


        # 4. Постобработка
        # y уже имеет форму (batch, seqlen, nheads, headdim) от ядра
        # Если ядро не использовало Z, оно вернёт y без gating.
        # В нашем случае мы передали Z, и ядро применило gating внутри.
        # В оригинале ядро возвращает y = y * silu(z) (если z передан) и затем применяет норму?
        # В документации: если Z передан, ядро выполняет gate и norm (если outproj_norm_weight задан).
        # Мы не используем fused norm, поэтому просто применяем свою норму после.

        # Приводим к плоскому виду
        y = rearrange(y, "b l h p -> b l (h p)")  # (batch, seqlen, d_inner)
        z = rearrange(z, "b l h p -> b l (h p)")  # (batch, seqlen, d_inner)

        # Применяем gating и норму (если ядро не сделало этого)
        y = y * F.silu(z)          # gate
        y = self.norm(y)           # RMSNorm
        y = self.out_proj(y)       # (batch, seqlen, d_model)

        # Создаём пустой кэш (для совместимости)
        h = InferenceCache(
            torch.zeros(batch, self.args.d_inner + 2 * self.args.d_state, self.args.d_conv, device=self.device),
            torch.zeros(batch, self.args.nheads, self.args.headdim, self.args.d_state, device=self.device),
            torch.zeros(batch, self.args.nheads, self.args.num_rope_angles, device=self.device),
        )

        return y, h

    def step(self, u: Tensor, h: InferenceCache) -> tuple[Tensor, InferenceCache]:
        """
        Один шаг инференса.
        
        Args:
            u: (batch, 1, d_model)
            h: состояние
            
        Returns:
            y: (batch, 1, d_model)
            h: обновленное состояние
        """
        assert u.shape[1] == 1, "Only one token per inference step"
        
        batch = u.shape[0]
        
        zxBCdt = self.in_proj(u.squeeze(1))
        z, x, B, C, dt, A, trap, angles = torch.split(
            zxBCdt,
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
        
        # Размерности
        z = rearrange(z, "b (h p) -> b h p", p=self.args.headdim)
        x = rearrange(x, "b (h p) -> b h p", p=self.args.headdim)
        B = rearrange(B, "b (g n) -> b g n", g=self.num_bc_heads)
        C = rearrange(C, "b (g n) -> b g n", g=self.num_bc_heads)
        
        if self.num_bc_heads < self.args.nheads:
            B = B.expand(-1, self.args.nheads, -1)
            C = C.expand(-1, self.args.nheads, -1)
        
        dt = F.softplus(dt + self.dt_bias)
        
        # Нормализация
        B = self.B_norm(B)
        C = self.C_norm(C)
        
        B = B + self.B_bias
        C = C + self.C_bias
        
        # RoPE для инференса
        if angles is not None:
            # Обновляем состояние углов
            h.angle_state = h.angle_state + angles  # Простое обновление
            B, C = apply_rotary_embedding(B, C, h.angle_state)
        
        # SSM шаг
        A = -heavy_tail_activation(A)
        A = torch.clamp(A, max=-1e-4)
        
        dA = torch.exp(dt * A)  # (batch, nheads)
        dBx = torch.einsum("bh, bn, bhp -> bhpn", dt, B, x)
        h.ssm_state = h.ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx
        
        y = torch.einsum("bhpn, bn -> bhp", h.ssm_state, C)
        y = y + rearrange(self.D, "h -> h 1") * x
        
        # Выход
        y = rearrange(y, "b h p -> b (h p)")
        z = rearrange(z, "b l h p -> b l (h p)")
        y = self.norm(y, z)
        y = self.out_proj(y)
        
        return y.unsqueeze(1), h


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5, device: Device = None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d, device=device))

    def forward(self, x, z=None):
        if z is not None:
            x = x * F.silu(z)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight