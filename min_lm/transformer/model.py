import torch
import torch.nn as nn

from .modules import RMSNorm, SwiGLU, SiLUFFN, Embedding
from .attention import MultiHeadSelfAttention


class TransformerBlock(nn.Module):
    """
    A single transformer block with multi-head attention and feed-forward network.
    
    Supports both pre-norm and post-norm architectures. Default is pre-norm
    as it's more stable for deep networks.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward network hidden dimension
        max_seq_len: Maximum sequence length
        rope_theta: RoPE theta parameter
        device: Torch device for computations
        dtype: Data type for parameters
        use_rope: Whether to use Rotary Position Embeddings
        disable_rmsnorm: Disable RMSNorm (for ablation studies)
        use_swiglu: Use SwiGLU activation (vs standard SiLU FFN)
        post_norm: Use post-normalization instead of pre-normalization
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, rope_theta: float,
                 device: torch.device | None = None, dtype: torch.dtype | None = None,
                 use_rope: bool = True, disable_rmsnorm: bool = False, use_swiglu: bool = True, post_norm: bool = False,
                 use_flash_attention: bool = False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.device = device
        self.dtype = dtype
        self.post_norm = post_norm
        self.disable_rmsnorm = disable_rmsnorm

        self.attention = MultiHeadSelfAttention(
            d_model,
            num_heads,
            max_seq_len,
            rope_theta,
            use_rope=use_rope,
            device=device,
            dtype=dtype,
            use_flash_attention=use_flash_attention,
        )
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype) if use_swiglu else SiLUFFN(d_model, d_ff, device=device, dtype=dtype)
        self.norm1 = nn.Identity() if disable_rmsnorm else RMSNorm(d_model, device=device, dtype=dtype)
        self.norm2 = nn.Identity() if disable_rmsnorm else RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Forward pass through the transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            token_positions: Position indices for RoPE
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        if not self.post_norm:
            x_norm = self.norm1(x)
            x_attn = self.attention(x_norm, token_positions, mask)
            x_part1 = x + x_attn

            x_norm2 = self.norm2(x_part1)
            x_ffn = self.ffn(x_norm2)
            x_part2 = x_part1 + x_ffn
            return x_part2
        else:
            x_attn = self.attention(x, token_positions, mask)
            z = x + x_attn
            z = self.norm1(z)
            y_ffn = self.ffn(z)
            y = z + y_ffn
            y = self.norm2(y)
            return y


class TransformerLM(nn.Module):
    """
    Transformer Language Model for autoregressive text generation.
    
    A GPT-style decoder-only transformer with causal masking.
    
    Args:
        vocab_size: Size of the vocabulary
        context_length: Maximum sequence length the model can process
        d_model: Model dimension (hidden size)
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads per block
        d_ff: Feed-forward network hidden dimension
        rope_theta: RoPE theta parameter (typically 10000)
        device: Torch device for computations
        dtype: Data type for parameters (e.g., float16 for mixed precision)
        use_rope: Whether to use Rotary Position Embeddings
        disable_rmsnorm: Disable RMSNorm (for ablation studies)
        use_swiglu: Use SwiGLU activation (vs standard SiLU FFN)
        post_norm: Use post-normalization instead of pre-normalization
        use_checkpoint: Enable gradient checkpointing to save memory
    """
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float,
                 device: torch.device | None = None, dtype: torch.dtype | None = None,
                 use_rope: bool = True, disable_rmsnorm: bool = False, use_swiglu: bool = True, post_norm: bool = False,
                 use_checkpoint: bool = False, use_flash_attention: bool = False):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.use_checkpoint = use_checkpoint

        self.token_embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model,
                num_heads,
                d_ff,
                context_length,
                rope_theta,
                device=device,
                dtype=dtype,
                use_rope=use_rope,
                disable_rmsnorm=disable_rmsnorm,
                use_swiglu=use_swiglu,
                post_norm=post_norm,
                use_flash_attention=use_flash_attention,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.Identity() if disable_rmsnorm else RMSNorm(d_model, device=device, dtype=dtype)
        from .modules import Linear
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor):
        """
        Forward pass through the language model.
        
        Args:
            token_ids: Input token IDs of shape (batch_size, seq_len)
            
        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device
        token_positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(token_ids)
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                def _fn(inp: torch.Tensor) -> torch.Tensor:
                    return layer(inp, token_positions, mask=None)
                x = torch.utils.checkpoint.checkpoint(_fn, x, use_reentrant=False)
            else:
                x = layer(x, token_positions, mask=None)
        x = self.norm(x)
        return self.lm_head(x)


