import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange

from .modules import Linear
from .utils import softmax


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for transformers.
    
    Applies rotation-based position encodings to query and key vectors,
    enabling the model to capture relative position information.
    
    Args:
        theta: Base for the geometric progression of frequencies (typically 10000)
        d_k: Dimension of key/query vectors per head
        max_seq_len: Maximum sequence length to precompute sin/cos for
        device: Torch device for computations
    """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        t = torch.arange(max_seq_len, device=device)
        angles = torch.einsum("i,j->ij", t, inv_freq)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        """
        Apply rotary embeddings to input tensor.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_k)
            token_positions: Position indices of shape (..., seq_len)
            
        Returns:
            Rotated tensor of same shape as input
        """
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        if x.dim() == 4:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        x1 = x_reshaped[..., 0]
        x2 = x_reshaped[..., 1]
        rotated_x_half = torch.stack([-x2, x1], dim=-1)
        rotated_x_half = rotated_x_half.flatten(start_dim=-2)
        cos_expanded = cos.repeat_interleave(2, dim=-1)
        sin_expanded = sin.repeat_interleave(2, dim=-1)
        return (x * cos_expanded + rotated_x_half * sin_expanded).to(x.dtype)


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor | None = None):
    """
    Compute scaled dot-product attention.
    
    Implements the attention mechanism: Attention(Q,K,V) = softmax(QK^T/√d_k)V
    
    Args:
        q: Query tensor of shape (..., seq_len, d_k)
        k: Key tensor of shape (..., seq_len, d_k)
        v: Value tensor of shape (..., seq_len, d_v)
        attn_mask: Boolean mask where True = attend, False = mask out.
                   If None, assumes causal (autoregressive) masking.
        
    Returns:
        Attention output of shape (..., seq_len, d_v)
    """
    d_k = q.size(-1)
    attn_scores = einsum(q, k, "... s_q d, ... s_k d -> ... s_q s_k") / math.sqrt(d_k)
    if attn_mask is None:
        seq_len = q.size(-2)
        attn_mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool))
    attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))
    attn_weights = softmax(attn_scores, dim=-1)
    output = einsum(attn_weights, v, "... s_q s_k, ... s_k d_v -> ... s_q d_v")
    return output


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention module.
    
    Splits input into multiple heads, applies scaled dot-product attention
    in parallel, then concatenates and projects the results.
    
    Args:
        d_model: Model dimension (must be divisible by num_heads)
        num_heads: Number of attention heads
        max_seq_len: Maximum sequence length (required if use_rope=True)
        rope_theta: RoPE theta parameter (required if use_rope=True)
        use_rope: Whether to use Rotary Position Embeddings
        device: Torch device for computations
        dtype: Data type for parameters
    """
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int | None = None, rope_theta: float | None = None, use_rope: bool = True, device: torch.device | None = None, dtype: torch.dtype | None = None, use_flash_attention: bool = False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.device = device
        self.dtype = dtype
        self.use_flash_attention = use_flash_attention

        self.d_k = d_model // num_heads
        self.WQ = Linear(d_model, d_model, device=device, dtype=dtype)
        self.WK = Linear(d_model, d_model, device=device, dtype=dtype)
        self.WV = Linear(d_model, d_model, device=device, dtype=dtype)
        self.WO = Linear(d_model, d_model, device=device, dtype=dtype)

        if self.use_rope:
            assert rope_theta is not None and max_seq_len is not None, "rope_theta and max_seq_len must be provided if use_rope is True"
            self.rope = RotaryPositionalEmbedding(theta=rope_theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device)
        else:
            self.rope = None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None, mask: torch.Tensor | None = None):
        """
        Apply multi-head self-attention to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            token_positions: Position indices (required if use_rope=True)
            mask: Attention mask (optional, defaults to causal mask)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        Q = self.WQ(x)
        K = self.WK(x)
        V = self.WV(x)
        Q = rearrange(Q, "b s (h d) -> b h s d", h=self.num_heads)
        K = rearrange(K, "b s (h d) -> b h s d", h=self.num_heads)
        V = rearrange(V, "b s (h d) -> b h s d", h=self.num_heads)
        if self.use_rope:
            assert token_positions is not None, "token_positions must be provided when use_rope is True"
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        if self.use_flash_attention:
            # Prefer PyTorch's native scaled_dot_product_attention, which uses Flash/SDPA kernels when available
            if mask is None:
                attn_out = F.scaled_dot_product_attention(Q, K, V, attn_mask=None, is_causal=True)
            else:
                # If a mask is provided, rely on it and disable internal causal masking
                attn_out = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, is_causal=False)
        else:
            attn_out = scaled_dot_product_attention(Q, K, V, attn_mask=mask)
        output = rearrange(attn_out, "b h s d -> b s (h d)")
        return self.WO(output)


