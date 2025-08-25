from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer
import numpy as np
import torch


class LLMInference:
    def __init__(
        self, tokenizer: Tokenizer, llm: TransformerLM, max_token_generated: int, temperature: float, top_p: float, device: torch.device | None = None
    ) -> None:
        self.tokenizer = tokenizer
        self.llm = llm
        self.max_token_generated = max_token_generated
        self.temperature = temperature
        self.top_p = top_p
        self.device = device

    def encode(self, text, max_token_generated: int | None = None, temperature: float | None = None, top_p: float | None = None) -> str:
        temperature, top_p = temperature or self.temperature, top_p or self.top_p
        max_token_generated = max_token_generated or self.max_token_generated
        tokens = self.tokenizer.encode(text)
          
        generated_text: str = ""
        
        for _ in range(max_token_generated):
            tokens_torchified = torch.from_numpy(np.asarray(tokens)).to(device=self.device).reshape(1, -1)
            # get only last token's next token prediction as we don't care the rest for this use case.
            next_token_predictions: torch.Tensor = self.llm(tokens_torchified)[:, -1:, :]

            logits = next_token_predictions / temperature
            probs = self.llm.softmax(logits)

            # do top p
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            # Keep only tokens with cumulative probability <= top_p
            cutoff = cumulative_probs > top_p
            # Set probabilities above the cutoff to zero
            sorted_probs[cutoff] = 0
            # Re-normalize
            probs = torch.zeros_like(probs)
            probs.scatter_(-1, sorted_indices, sorted_probs)
            pmf = probs / probs.sum(dim=-1, keepdim=True)
            
            cdf = torch.cumsum(pmf, dim=-1)
            uniform_samples = torch.rand(cdf.shape[0]).to(device=self.device).reshape(-1, 1, 1)
            token_predicted = torch.searchsorted(cdf, uniform_samples)[0].cpu().numpy()[0, 0].item()
            token_predicted = min(max(token_predicted, 0), probs.shape[-1] - 1)

            tokens.append(token_predicted)
            generated_text += self.tokenizer.decode([token_predicted])

            if generated_text.endswith("<|endoftext|>"):
                break

        
        return generated_text











