import torch
import torch.nn as nn
import numpy as np
from itertools import chain

from typing import List
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

    def forward(self, cache_states: List[CacheState]) -> torch.Tensor:
        batch_size = len(cache_states)

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
