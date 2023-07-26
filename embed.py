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
            if (input not in self._input_to_index and
                self._vocab_size < self._max_vocab_size):

                self._input_to_index[input] = self._vocab_size
                self._vocab_size += 1

            # Return the index of the input, or 0 (UNK) if it is not in the vocabluary
            return self._input_to_index.get(input, 0)

        indices = torch.tensor([input_to_index(input) for input in inputs]).long()

        return self._embedding(indices)