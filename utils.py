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

def mask_renormalize(probs : torch.FloatTensor, mask: torch.ByteTensor) -> torch.FloatTensor:
    """Renormalizes probs with a mask so that the unmasked entries sum to 1.
    
    Args:
        probs (torch.FloatTensor): batch of probability distributions with shape 
          (batch_dim1, batch_dim2, ..., num_elements).
        mask (torch.ByteTensor): masks out elements if the value is 0.

    Returns:
        renormalized_probs (torch.FloatTensor): the tensor of same shape as probs.
          Each batch row (last dim) sums to 1, where masked entries have value 0.
          If all entries in a row are masked, the batch row sums to 0.
    """
    # Set masked entries to 0
    masked_probs = probs * mask.float()
    # Renormalize the unmasked entries
    renormalized_probs = masked_probs / (masked_probs.sum(dim=-1, keepdim=True) + 1e-8)
    return renormalized_probs
