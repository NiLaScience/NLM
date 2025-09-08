import math
import torch


class AdamW(torch.optim.Optimizer):
    """
    AdamW optimizer with decoupled weight decay.
    
    Implements the AdamW algorithm from 'Decoupled Weight Decay Regularization'.
    Weight decay is applied directly to parameters rather than to gradients.
    
    Args:
        params: Model parameters to optimize
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for computing running averages of gradient
                and its square (default: (0.9, 0.999))
        eps: Term added to denominator for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.0)
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """
        Perform a single optimization step.
        
        Args:
            closure: Optional closure that reevaluates the model and returns loss
            
        Returns:
            Loss value if closure is provided
        """
        loss = None if closure is None else closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                α = group["lr"]
                β1, β2 = group["betas"]
                ε = group["eps"]
                λ = group["weight_decay"]

                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                state["t"] += 1
                t = state["t"]

                g = p.grad.data
                state["m"] = β1 * state["m"] + (1 - β1) * g
                state["v"] = β2 * state["v"] + (1 - β2) * g * g
                α_t = α * math.sqrt(1 - β2 ** t) / (1 - β1 ** t)
                p.data -= α_t * state["m"] / (torch.sqrt(state["v"]) + ε)
                if λ > 0:
                    p.data -= α * λ * p.data

        return loss


