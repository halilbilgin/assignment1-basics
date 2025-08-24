import os
import pickle
import tempfile
import typing
import click
import torch
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train_tokenizer import run_bpe, train_tokenizer
from cs336_basics.optimizers import AdamW, CrossEntropy, learning_rate_scheduler
from cs336_basics.model import TransformerLM
from cs336_basics.data_loader import load_dataset, get_batch
import numpy as np
import tqdm


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

@click.command()
@click.option("--input-path", type=str, required=True, help="Path to input text file.")
@click.option("--vocab-size", type=int, required=True, help="Vocabulary size for tokenizer.")
@click.option("--context-length", type=int, required=True, help="Context length for transformer.")
@click.option("--d-model", type=int, required=True, help="Transformer model dimension.")
@click.option("--d-ff", type=int, required=True, help="Feedforward dimension.")
@click.option("--rope-theta", type=float, default=10000.0, help="RoPE theta value.")
@click.option("--num-layers", type=int, required=True, help="Number of transformer layers.")
@click.option("--num-heads", type=int, required=True, help="Number of attention heads.")
@click.option("--min-learning-rate", type=float, required=True, help="Minimum learning rate.")
@click.option("--max-learning-rate", type=float, required=True, help="Maximum learning rate.")
@click.option("--warmup-iters", type=int, required=True, help="Number of warmup iterations.")
@click.option("--iters", type=int, required=True, help="Cosine cycle iterations.")
@click.option("--batch-size", type=int, required=True, help="Batch size for training.")
@click.option("--adamw-betas", type=(float, float), default=(0.9, 0.999), help="AdamW betas.")
@click.option("--adamw-weight-decay", type=float, default=0.01, help="AdamW weight decay.")
@click.option("--output-path", type=str, required=True, help="Output directory for checkpoints and tokenizer.")
@click.option("--device", type=str, default="cpu", help="Device to use (cpu or cuda).")
@click.option("--seed", type=int, default=42, help="Seed")
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
    iters: int,
    batch_size: int,
    adamw_betas: tuple[float, float],
    adamw_weight_decay: float,
    output_path: str,
    device: str,
    seed: int
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if warmup_iters >= iters:
        raise ValueError("Iters must be greater than warmup iters.")

    tokenizer_path = os.path.join(output_path, "tokenizer")
    os.makedirs(tokenizer_path, exist_ok=True)
    checkpoint_path = os.path.join(output_path, "checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)

    tokenizer = train_tokenizer(
        input_path=input_path, output_path=tokenizer_path, vocabulary_size=vocab_size, special_tokens=["<|endoftext|>"]
    )
    _, tokenized_data_file = tempfile.mkstemp()
    with open(tokenized_data_file, "wb+") as f_write, open(input_path) as f_read:
        tokens = tokenizer.encode(f_read.read())
        np.save(f_write, np.asarray(tokens))

    dataset = load_dataset(tokenized_data_file)

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        num_heads=num_heads,
        d_model=d_model,
        d_ff=d_ff,
        rope_theta=rope_theta,
        device=torch.device(device)
    )
    loss_function = CrossEntropy()
    optimizer = AdamW(params=model.parameters(), weight_decay=adamw_weight_decay, betas=adamw_betas)

    with tqdm.tqdm(total=iters, unit=" iter", mininterval=1) as tepoch:
        for iteration in range(iters):
            tepoch.update(1)
            # Update learning rate of the optimizer using the scheduler:
            lr = learning_rate_scheduler(
                it=iteration,
                max_learning_rate=max_learning_rate,
                cosine_cycle_iters=iters,
                warmup_iters=warmup_iters,
                min_learning_rate=min_learning_rate,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            
            batch = get_batch(dataset, batch_size, context_length, device)
            optimizer.zero_grad()
            training_loss = loss_function(model(batch[0]).reshape(-1, vocab_size), batch[1].reshape(-1))

            training_loss.backward()
            optimizer.step()

            tepoch.set_postfix({'loss': training_loss})

            if iteration % 1000 == 0:
                save_checkpoint(model, optimizer, iteration, os.path.join(checkpoint_path, f"iter{iteration}.ckp"))

    save_checkpoint(model, optimizer, iters, os.path.join(checkpoint_path, "final.ckp"))
    

if __name__ == "__main__":
    train()
