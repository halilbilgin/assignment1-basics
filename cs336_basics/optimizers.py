import math
from typing import Callable, Iterable, Optional
from jaxtyping import Float, Int
from torch import Tensor, nn
import torch


class CrossEntropy(nn.Module):
    def forward(
        self, inputs: Float[Tensor, "batch_size output_size"], targets: Int[Tensor, " batch_size"]
    ) -> Float[Tensor, ""]:
        max_value = torch.max(inputs, dim=1).values.reshape(-1, 1)
        sum_value = torch.sum(torch.exp(inputs - max_value), dim=1, keepdim=True)
        softmax_output = inputs - max_value - torch.log(sum_value)

        return -1.0 * torch.mean(torch.gather(input=softmax_output, dim=1, index=targets.reshape(-1, 1)))


class StochasticGradientDescent(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3) -> None:
        if lr <= 0:
            raise ValueError("Learning rate must be strictly positive")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        r"""Perform a single optimization step to update parameter.

        Args:
            closure (Callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or initial value.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.

        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, weight_decay: float, betas: tuple[float, float], eps: float = 1e-8, lr=1e-3) -> None:
        if lr <= 0:
            raise ValueError("Learning rate must be strictly positive")
        defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        r"""Perform a single optimization step to update parameter.

        Args:
            closure (Callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr, weight_decay, (beta0, beta1), eps = group["lr"], group["weight_decay"], group["betas"], group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 1)  # Get iteration number from the state, or initial value.
                m = state.get("m", torch.zeros_like(p))  # Get first moment vector from the state, or initial value.
                v = state.get("v", torch.zeros_like(p))  # Get second moment vector from the state, or initial value.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                state["m"] = beta0 * m + (1 - beta0) * grad
                state["v"] = beta1 * v + (1 - beta1) * (grad**2)
                state["t"] = t + 1
                adjusted_alpha = lr * math.sqrt(1 - beta1**t) / (1 - (beta0**t))
                p.data -= adjusted_alpha * state["m"] / (torch.sqrt(state["v"]) + eps)
                p.data -= lr * weight_decay * p.data

        return loss


def clip_gradients(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6):
    """Clip gradients by scaling down by max_l2_norm."""
    with torch.no_grad():
        norms = []
        for parameter in parameters:
            if parameter.grad is None:
                continue
            l2_norm = torch.sum(parameter.grad**2) ** 0.5
            norms.append(l2_norm)

        total_norm = sum([norm**2 for norm in norms]) ** 0.5
        if total_norm < max_l2_norm:
            return

        scaling = max_l2_norm / (total_norm + 1e-6)
        for parameter in parameters:
            if parameter.grad is None:
                continue
            parameter.grad *= scaling


def learning_rate_scheduler(
    it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int
):
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    if it > cosine_cycle_iters:
        return min_learning_rate
    return min_learning_rate + 1 / 2 * (
        1 + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))
    ) * (max_learning_rate - min_learning_rate)


# weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
# opt = StochasticGradientDescent([weights], lr=100)
# for t in range(100):
#     opt.zero_grad() # Reset the gradients for all learnable parameters.
#     loss = (weights**2).mean() # Compute a scalar loss value.
#     print(loss.cpu().item())
#     loss.backward() # Run backward pass, which computes gradients.
#     opt.step() # Run optimizer step.
# print(torch.sum(torch.abs(weights)))
