from typing import List, Tuple
import torch
from types import SimpleNamespace
import math


def pack(tensors: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Size]]:
    """Packs a list of tensors into one buffer for sending to other workers"""
    buffer = torch.cat([t.view(-1) for t in tensors])  # copies
    shapes = [tensor.shape for tensor in tensors]
    return buffer, shapes


def unpack(buffer: torch.Tensor, shapes: List[torch.Size]) -> List[torch.Tensor]:
    """Provides pointers to tensors of original `shapes` in a flat-packed buffer."""
    idx = 0
    entries = []
    for tensor_shape in shapes:
        end = idx + tensor_shape.numel()
        entries.append(buffer[idx:end].view(size=tensor_shape))
        idx = end

    return entries


def params_in_optimizer(optimizer: torch.optim.Optimizer) -> List[torch.Tensor]:
    params = []
    for group in optimizer.param_groups:
        params.extend(group["params"])
    return params


def is_distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()  # type: ignore


def flatten(tensors: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    out = []
    for list in tensors:
        out.extend(list)
    return out


def allreduce_average(data, *args, **kwargs):
    """All-reduce average if torch.distributed is available, otherwise do nothing"""
    if is_distributed():
        data.div_(torch.distributed.get_world_size())  # type: ignore
        return torch.distributed.all_reduce(data, *args, **kwargs)  # type: ignore
    else:
        return SimpleNamespace(wait=lambda: None)
    
def find_factors(n : int) -> Tuple[int, int]:
    
    print(f'FINDINF FACTORS OF {n}')
    factors = set()
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            factors.add(i)
            # factors.add(n // i)
    
    factors = sorted(factors)
    print(f'len(factors) = {len(factors)}, factors = {factors}')
    mid_factor = factors[-2]
    print(f'factor1 = {mid_factor}, factor2 = {n // mid_factor}')
    return mid_factor, n // mid_factor

def min_distance_factors(n: int) -> Tuple[int, int]:
    i = int(n**0.5)
    while n % i != 0:
        i -= 1
        
    return i, n // i

        