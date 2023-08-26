import math
import os
import pickle
import torch
import numpy as np
from typing import List, Optional
from torch import ByteTensor

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

def as_batches(parallel_data: List[List[object]],
               batch_size: int, sequence_length: int) -> List[List[List[object]]]:
    """Iterable of batches of sequences of consecutive data of sequence_length.

    A single pass through this iterable will include all starting positions in
    each of the parallel sequences in data exactly once.
    
    Args:
        paralell_data (List[List[object]]): parallel sequences of consecutive
            timesteps of data. Resulting batches will only include consecutive 
            subsequences within a single parallel sequence of data.
        batch_size (int): size of the batches. Last batch may contain fewer than
            batch_size sequences.
        sequence_length (int): length of the sequences in each batch.

    Yields:
        List[List[object]]: the outer list is length of batch_size, the inner list
        are all length sequence_length. Inner lists are all consecutive subsequences.
    """
    positions = []
    for i, seq in enumerate(parallel_data):
        positions.extend([(i, start_pos) for start_pos in range(len(seq) - sequence_length)])

    # Shuffle the positions
    np.random.shuffle(positions)

    # Yield batches of positions of size batch_size * sequence_length
    for i in range(math.ceil(len(positions) / batch_size)):
        batch = [
            parallel_data[index][start:start+sequence_length]
                for index, start in positions[i*batch_size:(i+1)*batch_size]
        ]
        # (batch_size, sequence_length)
        yield batch

def save_pickle(content, pickle_file_path):
    """Saves content to pickle_file_path.
    
    Args:
        content (object): object to save.
        pickle_file_path (str): path to save the content to.
    
    Raises:
        ValueError: if the file already exists.
    
    Returns:
        None
    """

    if os.path.exists(pickle_file_path):
        raise ValueError(f"File already exists: {pickle_file_path}")
    
    print("Saving pickle file: ", pickle_file_path)
    with open(pickle_file_path,"wb") as f:
        pickle.dump(content, f)

def load_pickle(pickle_file_path):
    """Loads a pickle file.
    
    Args:
        pickle_file_path (str): path to the pickle file.
    
    Raises:
        ValueError: if the file does not exist.
    
    Returns:
        object: the object loaded from the pickle file.
    """
    if os.path.exists(pickle_file_path):
        print("Loading pickle file: ", pickle_file_path)
        with open(pickle_file_path, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError(f"File does not exist: {pickle_file_path}")