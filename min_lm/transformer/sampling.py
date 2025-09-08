import torch
import torch.nn as nn

from .utils import softmax


def _sample_from_logits(logits: torch.Tensor, temperature: float = 1.0, top_p: float | None = None) -> int:
    """
    Sample a token ID from logits with temperature and nucleus sampling.
    
    Args:
        logits: Raw model outputs for vocabulary (1D tensor)
        temperature: Sampling temperature (0 = greedy, higher = more random)
        top_p: Nucleus sampling threshold (only sample from top-p probability mass)
        
    Returns:
        Sampled token ID
    """
    if temperature is not None and temperature <= 0:
        return int(torch.argmax(logits, dim=-1).item())
    scaled = logits / (temperature if temperature is not None else 1.0)
    probs = softmax(scaled, dim=-1)
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumprobs = torch.cumsum(sorted_probs, dim=-1)
        keep = (cumprobs - sorted_probs) < top_p
        if not torch.any(keep):
            keep[..., 0] = True
        filtered_probs = sorted_probs * keep
        filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
        next_idx_sorted = torch.multinomial(filtered_probs, num_samples=1)
        next_token = sorted_indices.gather(-1, next_idx_sorted).squeeze(-1)
        return int(next_token.item())
    else:
        next_token = torch.multinomial(probs, num_samples=1)
        return int(next_token.item())


def generate(
    model: nn.Module,
    prompt_tokens: list[int] | torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    temperature: float = 1.0,
    top_p: float | None = None,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """
    Generate text autoregressively from a language model.
    
    Args:
        model: Transformer language model
        prompt_tokens: Initial tokens as list or tensor
        max_new_tokens: Maximum number of tokens to generate
        eos_token_id: Stop generation if this token is produced
        temperature: Sampling temperature (0 = greedy)
        top_p: Nucleus sampling threshold
        device: Device to run generation on
        
    Returns:
        Generated token IDs including the prompt
    """
    model_device = next(model.parameters()).device
    if isinstance(device, str):
        device = torch.device(device)
    device = device or model_device

    if isinstance(prompt_tokens, list):
        x = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
    else:
        x = prompt_tokens.to(device)
        if x.dim() == 1:
            x = x.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(x)
            last_logits = logits[:, -1, :].squeeze(0)
            next_id = _sample_from_logits(last_logits, temperature=temperature, top_p=top_p)
            next_token = torch.tensor([[next_id]], device=device, dtype=torch.long)
            x = torch.cat([x, next_token], dim=1)
            if eos_token_id is not None and next_id == eos_token_id:
                break
    return x.squeeze(0) if x.size(0) == 1 else x


