from numpy import typing as npt
import torch


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    
    dataset_torch = torch.from_numpy(dataset).to(device)
    # len(dataset) - context_length*2
    random_start_indices = torch.randint(0, len(dataset)-context_length, size=(batch_size, 1)) # (batch_size, 1)
    
    # context_length * batch_size * 2
    batch_indices = torch.arange(start=0, end=context_length).repeat(batch_size, 2) # (batch_size, context_length*2)
    batch_indices[:, context_length:] += 1
    batch_indices += random_start_indices # (batch_size, context_length*2)
    # batch size 4, m=3
    # [
    #   [0,1,2,3,4,5],
    #   [0,1,2,3,4,5],
    #   [0,1,2,3,4,5],
    #   [0,1,2,3,4,5]
    # ]

    batch = dataset_torch[batch_indices.flatten()].reshape(batch_size, 2, context_length)
    return batch[:, 0, :], batch[:, 1, :]


