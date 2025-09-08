import math
import os
from typing import Iterable, BinaryIO, IO

import numpy as np
import torch


def softmax(x: torch.Tensor, dim: int = -1):
    """
    Numerically stable softmax implementation.
    
    Args:
        x: Input tensor
        dim: Dimension along which to compute softmax
        
    Returns:
        Softmax probabilities that sum to 1 along the specified dimension
    """
    x_max = x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor):
    """
    Cross-entropy loss for language modeling.
    
    Manually computes cross-entropy with numerical stability tricks:
    - Subtracts max for numerical stability
    - Avoids computing full softmax by using log-sum-exp trick
    
    Args:
        logits: Model predictions of shape (batch, seq_len, vocab_size) or (N, vocab_size)
        targets: Target token IDs of shape (batch, seq_len) or (N,)
        
    Returns:
        Scalar loss value (mean negative log-likelihood)
    """
    if logits.dim() == 3:
        B, T, V = logits.shape
        x = logits.to(torch.float32).view(B * T, V)
        idx = targets.view(B * T)
    else:
        N, V = logits.shape
        x = logits.to(torch.float32)
        idx = targets

    x_max = x.max(dim=1, keepdim=True).values
    x_stable = x - x_max
    sum_exp = torch.sum(torch.exp(x_stable), dim=1, keepdim=True)
    logsumexp = torch.log(sum_exp)
    log_probs = x_stable - logsumexp
    nll = -log_probs.gather(1, idx.view(-1, 1)).squeeze(1)
    return nll.mean()


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Cosine learning rate schedule with linear warmup.
    
    Three phases:
    1. Linear warmup from 0 to max_lr (0 to warmup_iters)
    2. Cosine annealing from max_lr to min_lr (warmup_iters to cosine_cycle_iters)
    3. Constant min_lr (after cosine_cycle_iters)
    
    Args:
        it: Current iteration number
        max_learning_rate: Peak learning rate after warmup
        min_learning_rate: Minimum learning rate after cosine annealing
        warmup_iters: Number of warmup iterations
        cosine_cycle_iters: Total iterations for one cosine cycle
        
    Returns:
        Learning rate for current iteration
    """
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    elif it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cosine_factor = 0.5 * (1 + math.cos(progress * math.pi))
        return min_learning_rate + cosine_factor * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    Clip gradients by global L2 norm.
    
    Prevents gradient explosion by scaling down all gradients if their
    combined L2 norm exceeds max_l2_norm.
    
    Args:
        parameters: Model parameters with gradients to clip
        max_l2_norm: Maximum allowed L2 norm for the gradient vector
    """
    eps = 1e-6
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return
    grads = [p.grad.data for p in params_with_grad]
    total_norm = torch.norm(torch.stack([torch.norm(g, p=2) for g in grads]), p=2)
    clip_coef = max_l2_norm / (total_norm + eps)
    if clip_coef < 1:
        for p in params_with_grad:
            p.grad.data.mul_(clip_coef)


def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str | torch.device | None = None):
    """
    Sample a random batch of sequences for training.
    
    Args:
        x: Dataset as 1D array of token IDs
        batch_size: Number of sequences per batch
        context_length: Length of each sequence
        device: Device to place the batch tensors on
        
    Returns:
        Tuple of (inputs, targets) where:
        - inputs: Token IDs of shape (batch_size, context_length)
        - targets: Next-token IDs of shape (batch_size, context_length)
    """
    ix = torch.randint(len(x) - context_length, (batch_size,))
    inputs = torch.stack([torch.from_numpy(x[i : i + context_length].astype(np.int64)) for i in ix])
    targets = torch.stack([torch.from_numpy(x[i + 1 : i + 1 + context_length].astype(np.int64)) for i in ix])
    if device:
        inputs, targets = inputs.to(device), targets.to(device)
    return inputs, targets


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | BinaryIO | IO[bytes]):
    """
    Save model checkpoint for resuming training.
    
    Args:
        model: Model to save
        optimizer: Optimizer state to save
        iteration: Current training iteration
        out: Output path or file handle
    """
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }, out)


def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Load model checkpoint to resume training.
    
    Args:
        src: Checkpoint path or file handle
        model: Model to load state into
        optimizer: Optimizer to load state into
        
    Returns:
        Iteration number from checkpoint
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]


