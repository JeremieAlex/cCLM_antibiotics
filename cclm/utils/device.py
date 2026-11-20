import logging
import random
from typing import List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def get_device(name: str = "auto") -> torch.device:
    """``auto`` picks CUDA, then MPS, then CPU."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed: {seed}")


def pad_sequences(seqs: List[torch.Tensor]) -> torch.Tensor:
    """Right-pad 1-D token tensors with 0 into a (B, T) tensor."""
    out = torch.zeros(len(seqs), max(s.size(0) for s in seqs), dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : s.size(0)] = s
    return out
