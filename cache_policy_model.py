from typing import List
import torch
import torch.nn as nn

from cache import CacheState
from embed import generate_embedder

class CachePolicyModel(nn.Module):

    @classmethod
    def from_config(self, config) -> 'CachePolicyModel':
        obj_id_embedder = generate_embedder(config["obj_id_embedder"])
        obj_size_embedder = generate_embedder(config["obj_size_embedder"])
        cache_lines_embedder = generate_embedder(config["cache_lines_embedder"])
        cache_history_embedder = generate_embedder(config["cache_history_embedder"])

        return self(obj_id_embedder,
                    obj_size_embedder,
                    cache_lines_embedder,
                    cache_history_embedder,
                    config["num_heads"],
                    config["num_layers"])

    def __init__(self,
                 obj_id_embedder,
                 obj_size_embedder,
                 cache_lines_embedder,
                 cache_history_embedder,
                 num_heads,
                 num_layers):
        super(CachePolicyModel, self).__init__()

        self._obj_id_embedder = obj_id_embedder
        self._obj_size_embedder = obj_size_embedder
        self._cache_lines_embedder = cache_lines_embedder
        self._cache_history_embedder = cache_history_embedder

        d_model = obj_id_embedder.embedding_dim + obj_size_embedder.embedding_dim + cache_lines_embedder.embedding_dim + cache_history_embedder.embedding_dim
        self.transformer = nn.Transformer(d_model=d_model,
                                          nhead=num_heads,
                                          num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers,
                                          dim_feedforward=2048,
                                          dropout=0.1)

        self.linear = nn.Linear(d_model, cache_lines_embedder.embedding_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, cache_states: List[CacheState]) -> torch.Tensor:
        # Extract the cache access, cache lines, and cache history from the cache states
        cache_access, cache_lines, cache_history = zip(*cache_states)

        # Embed the object indices in cache_lines and access_history
        obj_id_embedding = self._obj_id_embedder([access.obj_id for access in cache_access])
        obj_size_embedding = self._obj_size_embedder([access.obj_size for access in cache_access])
        cache_lines_embedding = self._cache_lines_embedder(cache_lines)
        cache_history_embedding = self._cache_history_embedder(cache_history)

        # Combine current_access, access_history, and embedded_cache_lines
        combined_input = torch.cat((obj_id_embedding,
                                    obj_size_embedding,
                                    cache_lines_embedding,
                                    cache_history_embedding),
                                    dim=2)
        
        # Perform the Transformer computation
        output = self.transformer(combined_input)

        # Apply linear layer to get logits
        logits = self.linear(output)

        # Apply softmax to get the probability distribution
        probabilities = self.softmax(logits, dim=1)

        return probabilities
