import torch
import torch.nn as nn
from einops import einsum


class Linear(nn.Module):
    """
    Linear transformation layer without bias.
    
    Implements y = xW^T using einsum for clarity. Weight initialization
    uses truncated normal distribution for stable training.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        device: Torch device for parameters
        dtype: Data type for parameters
    """
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        # Initialize weight using truncated normal distribution
        # σ² = 2/(din + dout), so σ = sqrt(2/(din + dout))
        std = (2 / (in_features + out_features)) ** 0.5
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x):
        """Apply linear transformation to input."""
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    """
    Token embedding layer.
    
    Maps token IDs to dense vector representations. Initialized with
    truncated normal distribution (std=1.0).
    
    Args:
        num_embeddings: Size of vocabulary
        embedding_dim: Dimension of embedding vectors
        device: Torch device for parameters
        dtype: Data type for parameters
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        # Initialize weight using truncated normal distribution
        # σ = 1
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids):
        """Look up embeddings for token IDs."""
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Normalizes input using RMS statistics and applies learned scale.
    More stable and efficient than LayerNorm for transformers.
    
    Args:
        d_model: Model dimension to normalize
        eps: Small constant for numerical stability
        device: Torch device for parameters
        dtype: Data type for parameters
    """
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor):
        """
        Apply RMS normalization.
        
        Upcasts to float32 for stability, then downcasts back.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)

        RMS = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        result = x / RMS * self.weight

        return result.to(in_dtype)


class SwiGLU(nn.Module):
    """
    SwiGLU activation function for transformer FFN.
    
    Implements: FFN(x) = W2(SiLU(W1(x)) * W3(x))
    where SiLU(x) = x * sigmoid(x)
    
    This gated linear unit variant often outperforms standard FFN.
    
    Args:
        d_model: Input/output dimension
        d_ff: Hidden dimension (defaults to ~8/3 * d_model, rounded to multiple of 64)
        device: Torch device for parameters
        dtype: Data type for parameters
    """
    def __init__(self, d_model: int, d_ff: int | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.device = device
        self.dtype = dtype

        if d_ff is None:
            # Correctly calculate d_ff and round up to the nearest multiple of 64
            d_ff_approx = int(d_model * 8 / 3)
            self.d_ff = (d_ff_approx + 63) // 64 * 64
        else:
            self.d_ff = d_ff

        self.W1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.W2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)
        self.W3 = Linear(d_model, self.d_ff, device=device, dtype=dtype)

    def silu(self, x: torch.Tensor):
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor):
        """Apply SwiGLU transformation."""
        return self.W2(self.silu(self.W1(x)) * self.W3(x))


class SiLUFFN(nn.Module):
    """
    Standard FFN with SiLU activation.
    
    Implements: FFN(x) = W2(SiLU(W1(x)))
    A simpler alternative to SwiGLU.
    
    Args:
        d_model: Input/output dimension
        d_ff: Hidden dimension (defaults to 4 * d_model)
        device: Torch device for parameters
        dtype: Data type for parameters
    """
    def __init__(self, d_model: int, d_ff: int | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.device = device
        self.dtype = dtype
        # Default to 4 * d_model
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        self.W1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.W2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)

    def silu(self, x: torch.Tensor):
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor):
        """Apply SiLU FFN transformation."""
        return self.W2(self.silu(self.W1(x)))


