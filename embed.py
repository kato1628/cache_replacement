import math
from typing import List
import numpy as np
import torch
from torch import nn
from abc import ABCMeta

def generate_embedder(embedder_config: dict) -> nn.Module:
    """Generates an embedder from a configuration dictionary.
    
    Args:
        embedder_config (dict): a dictionary containing the configuration for
            the embedder.
    
    Returns:
        nn.Module: an embedder.
    """
    if embedder_config["type"] == "dynamic_vocab":
        return DynamicVocabEmbedder(embedder_config["embedding_dim"],
                                    embedder_config["max_vocab_size"])
    elif embedder_config["type"] == "logarithmic":
        return LogarithmicEmbedder(embedder_config["embedding_dim"],
                                   embedder_config["max_size"],
                                   embedder_config["max_vocab_size"])
    elif embedder_config["type"] == "positional":
        return PositionalEmbedder(embedder_config["embedding_dim"])
    else:
        raise ValueError(f"Unknown embedder type: {embedder_config['type']}")

class Embedder(nn.Module):
    """Embeds a batch of objects into a vector space.
    
    Subclasses of Embedder should implement the forward method.
    """

    __metaclass__ = ABCMeta

    def __init__(self, embedding_dim: int) -> None:
        """Sets the embedding dimension.
        
        Args:
            embedding_dim (int): the dimension of the embedding vector.
        """
        super(Embedder, self).__init__()
        self._embedding_dim = embedding_dim

    @property
    def embedding_dim(self) -> int:
        """Returns the embedding dimension."""
        return self._embedding_dim

class DynamicVocabEmbedder(Embedder):
    """Dynamically constructs a vocabulary, assigning embeddings to new inputs.
    
    After max_vocab_size unique inputs have been seen, all new inputs will be
    assigned to a UNK embedding.
    """

    def __init__(self, embedding_dim: int, max_vocab_size: int) -> None:
        super().__init__(embedding_dim)

        self._max_vocab_size = max_vocab_size
        self._input_to_index = {}
        self._vocab_size = 1 # 0 is reserved for UNK

        # Overwrite the default embedding weights with xavier uniform
        weights = torch.zeros(max_vocab_size, embedding_dim)
        nn.init.xavier_uniform_(weights)
        self._embedding = nn.Embedding(max_vocab_size, embedding_dim, _weight=weights)

    def forward(self, inputs: list[object]) -> torch.Tensor:
        """Embeds a batch of inputs.
        
        Args:
            inputs (list[object]): a list of inputs to embed.
        
        Returns:
            torch.FloatTensor: the embeddings of the inputs.
        """
        def input_to_index(input):
            if (input not in self._input_to_index and self._vocab_size < self._max_vocab_size):

                self._input_to_index[input] = self._vocab_size
                self._vocab_size += 1

            # Return the index of the input, or 0 (UNK) if it is not in the vocabluary
            return self._input_to_index.get(input, 0)

        indices = torch.tensor([input_to_index(input) for input in inputs]).long()

        return self._embedding(indices)

class PositionalEmbedder(Embedder):
    """Takes position index as input and outputs a simple fixed embedding."""

    def forward(self, position_indices):
        """Returns a fixed embedding for each position index.
        
        Embeds each position index into Vaswani et al.'s transformer positional embedding space.
          embed_{2i}(pos) = sin(pos / 10000^(2i/embed_dim))
          embed_{2i+1}(pos) = cos(pos / 10000^(2i/embed_dim))

        Args:
            position_indices (list[int]): a batch of position indices.

        Returns:
            embeddings (torch.FloatTensor): the embeddings of the position indices with shape
            (batch_size, embedding_dim).
        """
        batch_size = len(position_indices)

        # i's in above formula
        embed_indices = torch.arange(self.embedding_dim).expand(batch_size, -1).float()
        position_tensor = torch.tensor(position_indices).unsqueeze(-1).float()

        embedding = position_tensor / 10000. ** (2 * embed_indices / self.embedding_dim)
        embedding = torch.where(embed_indices % 2 == 0,
                                torch.sin(embedding),
                                torch.cos(embedding))

        return embedding

class LogarithmicEmbedder(Embedder):
    """ Embeds a batch of object sizes into a vector space using a logarithmic scale. """

    def __init__(self, embedding_dim: int, max_size: int, max_vocab_size: int) -> None:
        super().__init__(embedding_dim)

        # Calculate logarithmic scale boundaries
        log_boundaries = np.logspace(0, np.log10(max_size), num=max_vocab_size)

        # Calculate embedding indices based on the logarithmic scale
        self._log_to_index = {}
        for i, boundary in enumerate(log_boundaries):
            self._log_to_index[boundary] = i

        # Create embedding matrix
        self._embedding_matrix = torch.nn.Embedding(max_vocab_size, embedding_dim)

    def forward(self, inputs: List[float]) -> torch.FloatTensor:
        """Embeds a batch of inputs using the logarithmic scale.
        
        Args:
            inputs (list[float]): a list of inputs to embed.
        
        Returns:
            torch.FloatTensor: the embeddings of the inputs.
        """
        indices = []
        for size in inputs:
            # Find the index that satisfies the condition
            index = next(idx for boundary, idx in self._log_to_index.items() if boundary > size)
            
            indices.append(index)
        indices = torch.tensor(indices, dtype=torch.long)

        embeddings = self._embedding_matrix(indices)
        return embeddings