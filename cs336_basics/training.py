import os
import pickle
import typing
import torch
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train_tokenizer import run_bpe, train_tokenizer
from cs336_basics.optimizers import AdamW

def _save_checkpoint(f: typing.BinaryIO | typing.IO[bytes], data: dict) -> None:
    f.write(pickle.dumps(data))


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    output_dictionary = {
        "model_state": model.state_dict(),
        "iteration": iteration,
        "optimizer_state": optimizer.state_dict(),
    }

    if isinstance(out, str) or isinstance(out, os.PathLike):
        with open(out, "wb") as f:
            _save_checkpoint(f, output_dictionary)
    else:
        _save_checkpoint(out, output_dictionary)


def _load_checkpoint(f: typing.BinaryIO | typing.IO[bytes]):
    return pickle.loads(f.read())


def load_checkpoint(src, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    if isinstance(src, str) or isinstance(src, os.PathLike):
        with open(src, "rb") as f:
            output_dictionary = _load_checkpoint(f)
    else:
        output_dictionary = _load_checkpoint(src)

    model.load_state_dict(output_dictionary["model_state"])
    optimizer.load_state_dict(output_dictionary["optimizer_state"])
    return output_dictionary["iteration"]


def train(
    input_path: str,
    vocab_size: int,
    context_length: int,
    d_model: int,
    d_ff: int,
    rope_theta: float,
    num_layers: int,
    num_heads: int,
    min_learning_rate: int,
    max_learning_rate: int,
    warmup_iters: int,
    cosine_cycle_iters: int,
    adamw_betas: tuple[float, float],
    adamw_weight_decay: float,
    : float,

    output_path: str,
):
    tokenizer_path = os.path.join(output_path, "tokenizer")
    os.makedirs(tokenizer_path)
    
    tokenizer = train_tokenizer(
        input_path=input_path, output_path=tokenizer_path, vocabulary_size=vocab_size, special_tokens=["<|endoftext|>"]
    )

    optimizer = AdamW(betas=)
    

if __name__ == "__main__":
    pass
