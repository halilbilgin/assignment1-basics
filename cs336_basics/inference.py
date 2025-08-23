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
        
        tokens_torchified = torch.from_numpy(np.asarray(self.llm(tokens))).reshape(1, -1)
        generated_text: str = ""
        
        for _ in range(max_token_generated):
            # get only last token's next token prediction as we don't care the rest for this use case.
            next_token_predictions: torch.Tensor = self.llm(tokens_torchified)[:, -1, :]

            next_token_predictions_with_scaling = next_token_predictions / temperature
            next_token_predictions_with_scaling = self.llm.softmax(next_token_predictions_with_scaling)

            # do top p
            top_p_value = torch.quantile(next_token_predictions_with_scaling, 1-top_p, dim=2)
            next_token_predictions_with_scaling[next_token_predictions_with_scaling<top_p_value] = 0
            
            pmf: torch.Tensor = next_token_predictions_with_scaling / next_token_predictions_with_scaling.sum(dim=-1)
            cdf = torch.cumsum(pmf, dim=-1)
            uniform_samples = torch.rand(cdf.shape[0]).reshape(-1, 1, 1)
            
            generated_text += self.tokenizer.decode(torch.searchsorted(cdf, uniform_samples)[0].numpy()[0, 0])

            if generated_text[-1].endswith("<|endoftext|>"):
                break

        
        return generated_text











