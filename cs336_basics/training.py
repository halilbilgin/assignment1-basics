import os
import pickle
import typing
import click
import torch
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train_tokenizer import train_tokenizer
from cs336_basics.optimizers import AdamW, CrossEntropy, learning_rate_scheduler
from cs336_basics.model import TransformerLM, TransformerLMConfig
from cs336_basics.data_loader import load_dataset, get_batch
import numpy as np
import tqdm
from numpy import typing as npt

def _save_checkpoint(f: typing.BinaryIO | typing.IO[bytes], data: dict) -> None:
    f.write(pickle.dumps(data))


def save_checkpoint(
    model: TransformerLM,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    output_dictionary = {
        "model_state": model.state_dict(),
        "iteration": iteration,
        "optimizer_state": optimizer.state_dict(),
        "model_config": model.model_config.model_dump() if hasattr(model, "model_config") else {}
    }
    

    if isinstance(out, str) or isinstance(out, os.PathLike):
        with open(out, "wb") as f:
            _save_checkpoint(f, output_dictionary)
    else:
        _save_checkpoint(out, output_dictionary)


def _load_checkpoint(f: typing.BinaryIO | typing.IO[bytes]):
    return pickle.loads(f.read())


def load_checkpoint(src, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None):
    if isinstance(src, str) or isinstance(src, os.PathLike):
        with open(src, "rb") as f:
            output_dictionary = _load_checkpoint(f)
    else:
        output_dictionary = _load_checkpoint(src)

    model.load_state_dict(output_dictionary["model_state"])
    if optimizer:
        optimizer.load_state_dict(output_dictionary["optimizer_state"])
    return output_dictionary["iteration"]

def load_model(src, device: torch.device|None = None):
    if isinstance(src, str) or isinstance(src, os.PathLike):
        with open(src, "rb") as f:
            output_dictionary = _load_checkpoint(f)
    else:
        output_dictionary = _load_checkpoint(src)
    
    return TransformerLM(TransformerLMConfig.model_validate(output_dictionary["model_config"]), device)
    


def compute_loss(loss_function: torch.nn.Module, model: torch.nn.Module, dataset: npt.NDArray, vocab_size: int, batch_size: int, context_length: int, device: torch.device):
    # cross entropy ?
    # sum()
    num_batches = dataset.shape[0]//(context_length*batch_size)-1
    validation_loss = torch.Tensor([0])
    for i in range(num_batches):
        # context_length * batch_size * 2
        batch_indices = (
            np.tile(np.arange(start=0, stop=context_length), reps=(batch_size, 2))
        )
        batch_indices[:, context_length:] += i*batch_size*context_length

        # batch size 4, m=3
        # [
        #   [0,1,2,3,4,5],
        #   [0,1,2,3,4,5],
        #   [0,1,2,3,4,5],
        #   [0,1,2,3,4,5]
        # ]

        batch_np = dataset[batch_indices.flatten()].reshape(batch_size, 2, context_length)
        result = torch.from_numpy(batch_np).to(device)
        batch = result[:, 0, :], result[:, 1, :]
        validation_loss += loss_function(model(batch[0]).reshape(-1, vocab_size), batch[1].reshape(-1))

    return validation_loss / num_batches

@click.command()
@click.option("--input-path", type=str, required=True, help="Path to input text file.")
@click.option("--input-validation-path", type=str, required=True, help="Path to validation input text file.")
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
@click.option("--pretrained-tokenizer-path", type=str, default=None, help="pretrained tokenizer path if exists.")
@click.option("--pretrained-checkpoint-path", type=str, default=None, help="checkpoint path")
@click.option("--tokenized-training-data-path", type=str, default=None, help="pretokenized training dataset path")
@click.option("--tokenized-validation-data-path", type=str, default=None, help="pretokenized validation dataset path.")
@click.option("--first-n-tokens", type=int, default=None, help="First n tokens from dataset to use during training for debugging purposes.")
def train(
    input_path: str,
    input_validation_path: str,
    vocab_size: int,
    context_length: int,
    d_model: int,
    d_ff: int,
    rope_theta: float,
    num_layers: int,
    num_heads: int,
    min_learning_rate: float,
    max_learning_rate: float,
    warmup_iters: int,
    iters: int,
    batch_size: int,
    adamw_betas: tuple[float, float],
    adamw_weight_decay: float,
    output_path: str,
    device: str,
    seed: int,
    pretrained_tokenizer_path: str | None,
    pretrained_checkpoint_path: str | None,
    tokenized_training_data_path: str | None,
    tokenized_validation_data_path: str | None,
    first_n_tokens: int | None
):
    if (tokenized_training_data_path or tokenized_validation_data_path) and not pretrained_tokenizer_path:
        raise ValueError("Must provide the pretrained tokenizer if tokenized data is provided.")

    torch.manual_seed(seed)
    np.random.seed(seed)

    if warmup_iters >= iters:
        raise ValueError("Iters must be greater than warmup iters.")

    tokenizer_path = os.path.join(output_path, "tokenizer")
    os.makedirs(tokenizer_path, exist_ok=True)
    checkpoint_path = os.path.join(output_path, "checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)

    if pretrained_tokenizer_path:
        tokenizer = Tokenizer.load_from_path(pretrained_tokenizer_path)
    else:
        tokenizer = train_tokenizer(
            input_path=input_path, output_path=tokenizer_path, vocabulary_size=vocab_size, special_tokens=["<|endoftext|>"]
        )
    
    if tokenized_training_data_path is None:
        tokenized_training_data_path = os.path.join(output_path, "tokenized_training_data.npt")
        
        with open(tokenized_training_data_path, "wb+") as f_write, open(input_path) as f_read:
            training_dataset_tokens = tokenizer.encode(f_read.read())
            np.save(f_write, np.asarray(training_dataset_tokens))

    if tokenized_validation_data_path is None:
        tokenized_validation_data_path = os.path.join(output_path, "tokenized_validation_data.npt")
        
        with open(tokenized_validation_data_path, "wb+") as f_write, open(input_validation_path) as f_read:
            validation_dataset_tokens = tokenizer.encode(f_read.read())
            np.save(f_write, np.asarray(validation_dataset_tokens))

    training_dataset = load_dataset(tokenized_training_data_path)[:first_n_tokens]
    validation_dataset = load_dataset(tokenized_validation_data_path)

    if pretrained_checkpoint_path:
        model = load_model(pretrained_checkpoint_path, device=torch.device(device))

        optimizer = AdamW(params=model.parameters(), weight_decay=adamw_weight_decay, betas=adamw_betas)

        current_iteration = load_checkpoint(pretrained_checkpoint_path, model, optimizer)
    else:
        current_iteration = 0
        model_config = TransformerLMConfig(
            vocab_size=vocab_size,
            context_length=context_length,
            num_layers=num_layers,
            num_heads=num_heads,
            d_model=d_model,
            d_ff=d_ff,
            rope_theta=rope_theta
        )
        model = TransformerLM(model_config, device=torch.device(device))
        optimizer = AdamW(params=model.parameters(), weight_decay=adamw_weight_decay, betas=adamw_betas)


    loss_function = CrossEntropy()

    
    val_loss = torch.Tensor([0.0])

    with tqdm.tqdm(total=iters, unit=" iter", mininterval=1) as tepoch:
        for iteration in range(current_iteration, iters):
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
            
            batch = get_batch(training_dataset, batch_size, context_length, device)
            optimizer.zero_grad()
            training_loss = loss_function(model(batch[0]).reshape(-1, vocab_size), batch[1].reshape(-1))

            training_loss.backward()
            optimizer.step()


            if iteration % 1000 == 0:
                with torch.no_grad():
                    val_loss = compute_loss(loss_function, model, validation_dataset, vocab_size=vocab_size, batch_size=batch_size, context_length=context_length, device=torch.device(device))
                save_checkpoint(model, optimizer, iteration, os.path.join(checkpoint_path, f"iter{iteration}.ckp"))

            tepoch.set_postfix({'loss': training_loss.item(), 'validation_loss': val_loss.item()})


    save_checkpoint(model, optimizer, iters, os.path.join(checkpoint_path, "final.ckp"))
    

if __name__ == "__main__":
    train()
