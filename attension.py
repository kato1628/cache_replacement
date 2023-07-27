import abc
import torch
import utils
from torch import nn
from torch.nn import functional as F

class Attention(nn.Module):
    """attends over memory cells to produce a context vector.
    
    Given:
        - memory keys k_1, ..., k_n (n = num_cells)
        - memory values v_1, ..., v_n
        - query q

    Computes:
        - attention weights a_i = softmax(score(q, k)_i)
        - context = sum_i a_i v_i
    """

    __metaclass__ = abc.ABCMeta

    def forward(self, memory_keys: torch.FloatTensor,
                memory_values: torch.FloatTensor,
                queries: torch.FloatTensor,
                mask: torch.ByteTensor) -> torch.FloatTensor:
        """Computes the context vector.

        Args:
            memory_keys (torch.FloatTensor): the memory keys with the shape of 
              (batch_size, num_cells, key_dim).
            memory_values (torch.FloatTensor): the memory values with the shape
              of (batch_size, num_cells, value_dim).
            queries (torch.FloatTensor): the queries with the shape of 
              (batch_size, query_dim).
            mask (torch.ByteTensor): masks out elements if the value is 0 with
                the shape of (batch_size, num_cells).

        Returns:
            torch.FloatTensor: the context vector with the shape of (batch_size, value_dim).
        """
        if mask is None:
            mask = torch.ones(memory_keys.shape[0], memory_keys.shape[1])
        masks = mask.unsqueeze(1)

        # Compute the attention weights
        # (batch_size, 1, num_cells)
        attention_weights = F.softmax(self._score(queries, memory_keys), dim=-1)
        masked_attention_weights = utils.mask_renormalize(attention_weights, masks)

        # Compute the context vector
        # (batch_size, 1, value_dim)
        context = torch.bmm(masked_attention_weights, memory_values) # batch matrix multiplication
        return masked_attention_weights.squeeze(1), context.squeeze(1)
    
    @abc.abstractmethod
    def _score(self, queries: torch.FloatTensor, memory_keys: torch.FloatTensor) -> torch.FloatTensor:
        """Computes the score between the queries and memory keys.

        Args:
            queries (torch.FloatTensor): the queries with the shape of 
              (batch_size, query_dim).
            memory_keys (torch.FloatTensor): the memory keys with the shape of 
              (batch_size, num_cells, key_dim).

        Returns:
            torch.FloatTensor: the score between the queries and memory keys
              with the shape of (batch_size, 1, num_cells).
        """
        raise NotImplementedError

class GeneralAttention(Attention):
    """Score(q, k) = q.T W k. W is learned. (Luong et al., 2015)"""

    def __init__(self, query_dim: int,
                 memory_key_dim: int,
                 weight_initializer: callable = nn.init.xavier_uniform):
        """Constructs.
        
        Args:
            query_dim (int): the dimension of the queries.
            memory_key_dim (int): the dimension of the memory keys.
            weight_initializer (callable): the initializer for the weight matrix.
              Default is xavier uniform.
        """
        super().__init__()

        # Initialize the weight matrix
        # (query_dim, memory_key_dim)
        w = torch.zeros(query_dim, memory_key_dim)
        weight_initializer(w)
        self._w = nn.Parameter(w)

    def _score(self, queries: torch.FloatTensor, memory_keys: torch.FloatTensor) -> torch.FloatTensor:
        """Computes the score between the queries and memory keys.

        Args:
            queries (torch.FloatTensor): the queries with the shape of 
              (batch_size, query_dim).
            memory_keys (torch.FloatTensor): the memory keys with the shape of 
              (batch_size, num_cells, key_dim).

        Returns:
            torch.FloatTensor: the score between the queries and memory keys
              with the shape of (batch_size, 1, num_cells).
        """
        
        # add a extra dimension to queries
        # (batch_size, query_dim) to (batch_size, 1, query_dim)
        queries = queries.unsqueeze(1)

        # swap the last two dimensions of memory_keys
        # (batch_size, num_cells, key_dim) to (batch_size, key_dim, num_cells)
        memory_keys = memory_keys.transpose(1, 2)

        # (batch_size, 1, query_dim) * (batch_size, key_dim, num_cells)
        #   -> (batch_size, 1, num_cells)
        transformed_queries = torch.matmul(queries, self._w) # matrix multiplication
        
        # (batch_size, 1, num_cells) * (batch_size, num_cells, key_dim)
        #   -> (batch_size, 1, key_dim)
        scores = torch.bmm(transformed_queries, memory_keys)

        return scores