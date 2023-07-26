from typing import List, Optional
from torch import ByteTensor
import torch


def pad(seq_batch: List[List[object]],
        pad_token=0,
        min_len: Optional[int] = None) -> tuple[List[List[object]], ByteTensor]:
    """Pads a batch of sequences with pad_token to the same length.
    
    Args:
        seq_batch: A batch of sequences to pad
        pad_token: A token to pad with
        min_len: The minimum length to pad to. If None, the maximum length in
    
    Returns:
        A tuple of (padded_seq_batch, mask)
        padded_seq_batch: A padded batch of sequences
        mask: A mask of the padded batch
    """
    max_len = max(len(seq) for seq in seq_batch)
    if min_len is not None:
        max_len = max(max_len, min_len)

    batch_size = len(seq_batch)
    mask = torch.ones(batch_size, max_len).byte()

    padded = []
    for i, seq in enumerate(seq_batch):
        padding = max_len - len(seq)
        padded.append(seq + [pad_token] * padding)
        if padding > 0:
            mask[i, -padding:] = 0
    
    return padded, mask