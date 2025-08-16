from numpy import typing as npt
import torch
import numpy as np


def load_dataset(input_path: str) -> npt.NDArray:
    return np.load(input_path, mmap_mode='r')


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    
    # len(dataset) - context_length*2
    random_start_indices = np.random.randint(0, len(dataset)-context_length, size=(batch_size, 1)) # (batch_size, 1)
    
    # context_length * batch_size * 2
    batch_indices = (
        np.tile(np.arange(start=0, stop=context_length), reps=(batch_size, 2))
    )
    batch_indices[:, context_length:] += 1
    batch_indices += random_start_indices # (batch_size, context_length*2)
    # batch size 4, m=3
    # [
    #   [0,1,2,3,4,5],
    #   [0,1,2,3,4,5],
    #   [0,1,2,3,4,5],
    #   [0,1,2,3,4,5]
    # ]

    batch_np = dataset[batch_indices.flatten()].reshape(batch_size, 2, context_length)
    result = torch.from_numpy(batch_np).to(device)

    return result[:, 0, :], result[:, 1, :]


