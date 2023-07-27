import collections
import torch
import torch.nn as nn
import numpy as np
from itertools import chain

from typing import List, Optional
from attension import GeneralAttention, MultiQueryAttention
from cache import CacheState
from embed import generate_embedder
from loss_function import ReuseDistanceLoss
from utils import pad

class CachePolicyModel(nn.Module):

    @classmethod
    def from_config(self, config) -> 'CachePolicyModel':
        # Generate the embedders
        obj_id_embedder = generate_embedder(config["obj_id_embedder"])
        obj_size_embedder = generate_embedder(config["obj_size_embedder"])

        if config["cache_lines_embedder"] == "obj_id_embedder":
            cache_lines_embedder = obj_id_embedder
        else:
            cache_lines_embedder = generate_embedder(config["cache_lines_embedder"])
        
        cache_history_embedder = generate_embedder(config["cache_history_embedder"])

        # Generate loss function
        loss_function = ReuseDistanceLoss()

        return self(obj_id_embedder,
                    obj_size_embedder,
                    cache_lines_embedder,
                    cache_history_embedder,
                    loss_function,
                    config["lstm_hidden_size"],
                    config["max_attention_history"])

    def __init__(self,
                 obj_id_embedder: nn.Embedding,
                 obj_size_embedder: nn.Embedding,
                 cache_lines_embedder: nn.Embedding,
                 cache_history_embedder: nn.Embedding,
                 loss_function: nn.Module,
                 lstm_hidden_size: int,
                 max_attention_history: int):
        super(CachePolicyModel, self).__init__()

        # Embedding layers
        self._obj_id_embedder = obj_id_embedder
        self._obj_size_embedder = obj_size_embedder
        self._cache_lines_embedder = cache_lines_embedder
        self._cache_history_embedder = cache_history_embedder

        # LSTM layer
        self._lstm_cell = nn.LSTMCell(
            input_size=obj_id_embedder.embedding_dim + obj_size_embedder.embedding_dim,
            hidden_size=lstm_hidden_size)
        
        # Attention layer
        query_dim = cache_lines_embedder.embedding_dim
        self._history_attention = MultiQueryAttention(
            GeneralAttention(query_dim,
                             lstm_hidden_size))
        
        # Linear layer
        # (lstm_hidden_size + cache_history_embedder.embedding_dim) -> 1
        self._cache_line_scorer = nn.Linear(
            in_features=lstm_hidden_size + self._cache_history_embedder.embedding_dim,
            out_features=1)
        # (lstm_hidden_size + cache_history_embedder.embedding_dim) -> 1
        self._reuse_distance_estimator = nn.Linear(
            in_features=lstm_hidden_size + self._cache_history_embedder.embedding_dim,
            out_features=1)
        
        # Needs to be capped to prevent memory explosion
        self._max_attention_history = max_attention_history

        # Loss function
        self._loss_function = loss_function

    def forward(self, cache_states: List[CacheState],
                prev_hidden_state: Optional[object] = None) -> torch.Tensor:
        """Computes cache line to evict
        
            Each cache line in the cache state is scored by the model.
            Higher scores indicate that the cache line should be evicted.
            
        Args:
            cache_states (List[CacheState]): batch of cache states to process
                and whose cache lines to choose from.
            prev_hidden_state (Optional[object]): the result from the previous
                call to this function on the previous cache states. Use None
                only for the first call.
            inference (bool): set to be True at inference time, when the outputs
                are not used for training. If True, the hidden state will not be
                updated, be detached from the computation graph to prevent
                memory explosion."""
        batch_size = len(cache_states)

        if prev_hidden_state is None:
            hidden_state, hidden_state_history, cache_states_history \
                = (self._initial_hidden_state(batch_size))
        else:
            hidden_state, hidden_state_history, cache_states_history \
                = prev_hidden_state
        
        # Extract the cache access, cache lines, and cache history from the cache states
        cache_access, cache_lines, cache_history = zip(*cache_states)

        # Embed the obj_id and obj_size
        # (batch_size, embedding_dim)
        obj_id_embedding = self._obj_id_embedder([access.obj_id for access in cache_access])
        obj_size_embedding = self._obj_size_embedder([access.obj_size for access in cache_access])
        
        # Cache lines are padded to the same length for embedding layer
        cache_lines, mask = pad(cache_lines, pad_token=-1, min_len=1)
        cache_lines = np.array(cache_lines)
        num_cache_lines = cache_lines.shape[1]

        # Flatten cache_lines into a single list
        cache_lines = chain.from_iterable(cache_lines)

        # Embed the cache lines
        # (batch_size, num_cache_lines, embedding_dim)
        cache_lines_embedding = self._cache_lines_embedder(cache_lines).view(
            batch_size,
            num_cache_lines,
            -1)

        cache_history_embedding = self._cache_history_embedder(cache_history)

        return

    def _initial_hidden_state(self, batch_size: int) -> tuple[tuple[torch.FloatTensor,torch.FloatTensor],
                                                              collections.deque[torch.FloatTensor],
                                                              collections.deque[List[CacheState]]]:
        """Returns the initial hidden state, used when no hidden state is provided.
        
        Args:
            batch_size (int): the batch size of the hidden state to return.
            
        Returns:
            initial_hidden_state (tuple[torch.FloatTensor, torch.FloatTensor]) tuple of initial
                cell state and initial LSTM hidden state.
            hidden_state_history (collections.deque[torch.FloatTensor]): the list of past hidden
                states.
            cache_states_history (collections.deque[List[CacheState]]): the list of past cache
                states.
        """
        initial_cell_state = torch.zeros(batch_size, self._lstm_cell.hidden_size)
        initial_hidden_state = torch.zeros(batch_size, self._lstm_cell.hidden_size)
        initial_hidden_state_history = collections.deque([],
                                                         maxlen=self._max_attention_history)
        initial_cache_states_history = collections.deque([],
                                                        maxlen=self._max_attention_history)
        
        return ((initial_cell_state, initial_hidden_state),
                initial_hidden_state_history,
                initial_cache_states_history)