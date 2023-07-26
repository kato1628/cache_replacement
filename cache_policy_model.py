from typing import List
import torch
import torch.nn as nn

from cache import CacheState
from embed import generate_embedder

class CachePolicyModel(nn.Module):

    @classmethod
    def from_config(self, config) -> 'CachePolicyModel':
        cache_access_embedder = generate_embedder(config["cache_access_embedder"])
        cache_lines_embedder = generate_embedder(config["cache_lines_embedder"])
        cache_history_embedder = generate_embedder(config["cache_history_embedder"])

        return self(cache_access_embedder,
                    cache_lines_embedder,
                    cache_history_embedder,
                    config["num_heads"],
                    config["num_layers"])

    def __init__(self,
                 cache_access_embedder,
                 cache_lines_embedder,
                 cache_history_embedder,
                 num_heads,
                 num_layers):
        super(CachePolicyModel, self).__init__()

        self._cache_access_embedder = cache_access_embedder
        self._cache_lines_embedder = cache_lines_embedder
        self._cache_history_embedder = cache_history_embedder

        d_model = cache_access_embedder.embedding_dim + cache_lines_embedder.embedding_dim + cache_history_embedder.embedding_dim
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
        cache_access_embedding = self._cache_access_embedder([access.obj_id for access in cache_access])
        cache_lines_embedding = self._cache_lines_embedder(cache_lines)
        cache_history_embedding = self._cache_history_embedder(cache_history)

        # Combine current_access, access_history, and embedded_cache_lines
        combined_input = torch.cat((cache_access_embedding,
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
