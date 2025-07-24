import json
import os
import random
from collections import OrderedDict
from itertools import repeat
from pathlib import Path
from typing import Any, Iterator, Literal, Union

import numpy as np
import torch
from torch.utils.data import DataLoader


def ensure_dir(dirname: Union[str, Path]) -> None:
    """Create directory if it doesn't exist."""
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=True)


def read_json(fname: Union[str, Path]) -> OrderedDict:
    """Read JSON file and return as OrderedDict."""
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content: Any, fname: Union[str, Path]) -> None:
    """Write content to JSON file."""
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader: DataLoader) -> Iterator[Any]:
    """Wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(n_gpu_use: int) -> tuple[torch.device, list[int]]:
    """
    Setup GPU device if available. Get gpu device indices which are used for DataParallel.

    Args:
        n_gpu_use: Number of GPUs to use

    Returns:
        Tuple of (device, list of GPU indices)
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print(
            "Warning: There's no GPU available on this machine, "
            "training will be performed on CPU."
        )
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            f"Warning: The number of GPU's configured to use is {n_gpu_use}, "
            f"but only {n_gpu} are available on this machine."
        )
        n_gpu_use = n_gpu
    device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def set_worker_seed(worker_id: int) -> None:
    """
    Set seed for each dataloader worker.

    For more info, see https://pytorch.org/docs/stable/notes/randomness.html

    Args:
        worker_id (int): id of the worker.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_random_seed(seed: int) -> None:
    """
    Set random seed for model training or inference.

    Args:
        seed (int): defines which seed to use.
    """
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def create_saved_dirs(
    config, mode: Literal["train", "test"] = "train"
) -> dict[str, str]:
    from datetime import datetime

    if mode == "train":
        save_dir = Path(config.trainer.save_dir)
    else:
        save_dir = Path(config.inferencer.save_dir)

    name = config.name
    run_id = datetime.now().strftime(r"%m%d_%H%M%S")

    _save_dir = save_dir / "models" / name / run_id
    _save_dir.mkdir(parents=True, exist_ok=True)

    _log_dir = save_dir / "logs" / name / run_id
    _log_dir.mkdir(parents=True, exist_ok=True)

    return {"save_dir": str(_save_dir), "log_dir": str(_log_dir)}
