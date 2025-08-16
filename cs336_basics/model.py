import math
from typing import Optional
import torch
from torch import Tensor, nn
from jaxtyping import Float, Integer
import numpy as np
import matplotlib.pyplot as plt


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()

        std = math.sqrt(2 / in_features + out_features)

        self.W: Float[nn.Parameter, "out_features in_features"] = nn.Parameter(
            nn.init.trunc_normal_(torch.zeros((out_features, in_features)), mean=0, std=std, a=-3 * std, b=3 * std).to(
                device, dtype=dtype
            ),
        )

    def forward(self, x: Float[torch.Tensor, "... in_features"]) -> Float[torch.Tensor, "out_features ..."]:
        """"""
        # Wx -> (out_features, in_features)
        return torch.einsum("ij, ...j -> ...i", self.W, x)


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.W = nn.Parameter(
            nn.init.trunc_normal_(torch.zeros((num_embeddings, embedding_dim)), mean=0, std=1, a=-3, b=3).to(
                device, dtype=dtype
            ),
        )
        self.device, self.dtype = device, dtype

    def forward(self, token_ids: Integer[torch.Tensor, "..."]) -> Float[torch.Tensor, "... "]:
        one_hot_encoded_tokens = nn.functional.one_hot(token_ids, num_classes=self.num_embeddings).to(
            self.device, self.dtype
        )

        return torch.einsum("ji, ...j -> ...i", self.W, one_hot_encoded_tokens)


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device, self.dtype = device, dtype
        self.g = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        in_type = x.dtype

        denominator = torch.sqrt(1 / (self.d_model) * torch.einsum("...j->...", torch.pow(x, 2)) + self.eps)
        denominator = torch.unsqueeze(denominator, -1)

        return (x.to(torch.float32) / denominator * self.g).to(in_type)


class SiLU(nn.Module):
    """SiLU"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply silu, which can be thought of a smooth ReLU."""
        return torch.sigmoid(x) * x


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else round(8 / 3 * d_model)
        self.silu = SiLU()
        std = math.sqrt(2 / (self.d_model + self.d_ff))

        self.W1 = nn.Parameter(
            nn.init.trunc_normal_(torch.zeros((self.d_ff, self.d_model)), mean=0, std=std, a=-3 * std, b=3 * std)
        )
        self.W2 = nn.Parameter(
            nn.init.trunc_normal_(torch.zeros((self.d_model, self.d_ff)), mean=0, std=std, a=-3 * std, b=3 * std)
        )
        self.W3 = nn.Parameter(
            nn.init.trunc_normal_(torch.zeros((self.d_ff, self.d_model)), mean=0, std=std, a=-3 * std, b=3 * std)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feed forward network. Implements SwiGLU.

        W_{2}((W_{1}x)sigmoid(W_{1}*x)*W_{3})
        """
        silu_output = self.silu(torch.einsum("ji, ...i -> j...", self.W1, x))
        glu_output = silu_output * torch.einsum("ji, ...i -> j...", self.W3, x)

        return torch.einsum("ji, i...", self.W2, glu_output)


class ROPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        self.theta, self.d_k, self.max_seq_len, self.device = theta, d_k, max_seq_len, device

        all_token_positions = torch.arange(start=0, end=self.max_seq_len, step=1).unsqueeze(1)  # (max_seq_len, 1)
        all_ks = torch.arange(start=0, end=d_k // 2, step=1).unsqueeze(0)  # (1, d/2)

        theta_ik = all_token_positions / self.theta ** (2 * all_ks / d_k)
        cos_results = torch.cos(theta_ik)  # (max_seq_len, d/2)

        sin_results = torch.sin(theta_ik)  # (max_seq_len, d/2)

        self.sin_results: torch.Tensor
        self.cos_results: torch.Tensor
        self.register_buffer("sin_results", sin_results, persistent=False)
        self.register_buffer("cos_results", cos_results, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x: (..., seq_len, d_k)
        token_positions: (..., seq_len)
        """
        # (..., 12, 64)
        # (64, 64, 12)
        cos = self.cos_results[token_positions]  # (..., seq_len, d_k/2)
        sin = self.sin_results[token_positions]  # (..., seq_len, d_k/2)

        # x(i) := (m)
        # W_q(i) := (d_k, m)
        # q(i) = W_q*x(i) := (d_k, 1)
        # R(i) := (d_k, d_k) (block diagonal)
        # R(i)(k) := (2,2)

        # R(i)(k) q(k) = [  R(ik)11*q11+ R(ik)12*q21
        #                   R(ik)21*q11+ R(ik)22*q21
        #  ]

        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        x_even_new = x_even * cos - x_odd * sin
        x_odd_new = x_even * sin + x_odd * cos
        return torch.stack([x_even_new, x_odd_new], dim=-1).flatten(-2)


class SoftMax(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_value = torch.max(x)
        return torch.exp(x - max_value) / torch.sum(torch.exp(x - max_value), dim=self.dim, keepdim=True)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int) -> None:
        super().__init__()
        self.d_k = d_k

    def forward(
        self,
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys d_k"],
        V: Float[Tensor, " ... values d_v"],
        mask: Float[Tensor, " ... queries keys"] | None = None,
    ) -> Float[Tensor, "... values d_v"]:
        nominator = torch.einsum("...qd, ...kd -> ...qk", Q, K)
        denominator = math.sqrt(self.d_k)
        if mask is not None:
            nominator[~mask] = float("-inf")

        attention_values = SoftMax(dim=-1)(nominator / denominator)

        return torch.einsum("...qk, ...kd -> ...qd", attention_values, V)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, positional_embedding_layer: nn.Module | None = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // num_heads
        self.d_v = self.d_model // num_heads
        self.positional_embedding_layer = positional_embedding_layer

        self.q_proj_weight = nn.Parameter(torch.nn.init.xavier_normal_(torch.zeros(num_heads * self.d_k, self.d_model)))
        self.k_proj_weight = nn.Parameter(torch.nn.init.xavier_normal_(torch.zeros(num_heads * self.d_k, self.d_model)))
        self.v_proj_weight = nn.Parameter(torch.nn.init.xavier_normal_(torch.zeros(num_heads * self.d_v, self.d_model)))
        self.o_proj_weight = nn.Parameter(torch.nn.init.xavier_normal_(torch.zeros(self.d_model, num_heads * self.d_v)))
        self.attention_layer = ScaledDotProductAttention(self.d_k)

    def forward(
        self, x: Float[Tensor, "... seq_len d_model"], token_positions: Float[Tensor, "... seq_len"] | None = None
    ) -> Float[Tensor, "... seq_len d_model"]:
        sequence_length = x.shape[-2]

        Q = torch.einsum("hkd, ...sd -> ...hsk", self.q_proj_weight.reshape(self.num_heads, self.d_k, self.d_model), x)

        K = torch.einsum("hkd, ...sd -> ...hsk", self.k_proj_weight.reshape(self.num_heads, self.d_k, self.d_model), x)

        if self.positional_embedding_layer is not None and token_positions is not None:
            token_positions = torch.unsqueeze(token_positions, -2)
            Q = self.positional_embedding_layer(Q, token_positions=token_positions)
            K = self.positional_embedding_layer(K, token_positions=token_positions)
        elif any([self.positional_embedding_layer is not None, token_positions is not None]):
            raise ValueError("Either provide both positional embedding layer and token positions or none.")

        V = torch.einsum("hkd, ...sd -> ...hsk", self.v_proj_weight.reshape(self.num_heads, self.d_k, self.d_model), x)

        causal_mask = torch.tril(
            torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=x.device)
        ).expand(*Q.shape[:-2], sequence_length, sequence_length)
        # Compute attention
        attn_output = self.attention_layer(Q, K, V, mask=causal_mask)
        attn_output = attn_output.transpose(-3, -2).reshape(*x.shape[:-2], sequence_length, -1)
        # Attention(Q,K,V) = softmax(Q^TK/c)
        # MultiHead = Concat(head_1, head_2, head_n)
        # head_i = Attention(W_Qix, W_Kix, W_Vix)
        # MultiHeadSelfAttention(x) = W_O x MultiHead(WQx, WKx, WVx)

        return torch.einsum("dv, ...sv -> ...sd", self.o_proj_weight, attn_output)


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.rms_norm_first = RMSNorm(d_model, device=device)
        self.rms_norm_second = RMSNorm(d_model, device=device)
        self.rope = ROPE(rope_theta, self.d_model // num_heads, max_seq_len=max_seq_len, device=device)
        self.multi_head_self_attention = MultiHeadSelfAttention(
            d_model, num_heads, positional_embedding_layer=self.rope
        )

        self.d_ff = d_ff
        self.feed_forward = FFN(self.d_model, self.d_ff)

    def forward(self, x: Float[Tensor, "... seq_len d_model"]) -> Float[Tensor, "... seq_len d_model"]:
        token_positions = torch.range(0, x.shape[-2] - 1, dtype=torch.int).broadcast_to(*x.shape[:-1])
        first_layer = x + self.multi_head_self_attention(self.rms_norm_first(x), token_positions=token_positions)

        return first_layer + self.feed_forward(self.rms_norm_second(first_layer))


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.token_embedding_layer = Embedding(vocab_size, d_model, dtype=torch.float, device=device)
        self.transformer_blocks = [
            Transformer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                rope_theta=rope_theta,
                max_seq_len=context_length,
                device=device,
            )
            for _ in range(num_layers)
        ]
        self.rms_norm_final = RMSNorm(d_model, device=device)
        self.final_linear = torch.nn.Linear(d_model, vocab_size, bias=False, device=device)

        self.softmax = SoftMax(dim=-1)

    def forward(self, x: Float[Tensor, "... seq_len d_model"]) -> Float[Tensor, "... seq_len vocab_size"]:
        result = self.token_embedding_layer(x)

        for transformer_block in self.transformer_blocks:
            result = transformer_block(result)

        return self.final_linear(self.rms_norm_final(result))
