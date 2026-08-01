import os

import torch
import torch.distributed as dist


__all__ = [
    "average_gradients",
    "broadcast_params",
    "cleanup_dist",
    "get_device",
    "init_dist",
    "is_distributed",
    "reduce_tensor",
]


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def init_dist(backend="nccl", master_ip="127.0.0.1", port=29500):
    """Initialize torchrun when requested, or select one local CUDA device.

    A normal ``python script.py`` invocation uses a single GPU and does not
    create a process group. ``torchrun`` provides WORLD_SIZE/RANK/LOCAL_RANK,
    in which case the process group is initialized normally.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "DominoSearch requires a CUDA GPU. In Colab, select "
            "Runtime > Change runtime type > GPU."
        )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        os.environ.setdefault("MASTER_ADDR", master_ip)
        os.environ.setdefault("MASTER_PORT", str(port))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
    else:
        rank = 0
        world_size = 1
        torch.cuda.set_device(0)

    return rank, world_size


def get_device():
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required.")
    return torch.device("cuda", torch.cuda.current_device())


def reduce_tensor(tensor, average=True):
    reduced = tensor.detach().clone()
    if is_distributed():
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        if average:
            reduced.div_(dist.get_world_size())
    return reduced


def average_gradients(model):
    if not is_distributed():
        return
    world_size = dist.get_world_size()
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(world_size)


def broadcast_params(model):
    if not is_distributed():
        return
    for value in model.state_dict().values():
        dist.broadcast(value, src=0)


def cleanup_dist():
    if is_distributed():
        dist.destroy_process_group()
