from __future__ import annotations # to avoid circular import
import six
import abc
from typing import TYPE_CHECKING, Dict, List, Optional, TypedDict

import torch
from cache_policy_model import CachePolicyModel
from wiki_trace import WikiTrace
from configuration import config

# to avoid circular import
if TYPE_CHECKING:
    from cache import CacheState

class CacheLineScores(TypedDict):
    obj_id: int
    score: int

class CacheLineScorer(six.with_metaclass(abc.ABCMeta, object)):
  """Scores cache lines based on how evictable each line is."""

  @abc.abstractmethod
  def __call__(self, cache_state: CacheState, access_times: Dict[int, int]):
    """Scores all the cache lines in the given cache state.

    Args:
      cache_state (CacheState): the current cache state.
      access_times (Dict[int, int]): maps each cache line (int) to its access time (int).

    Returns:
      scores (dict{int: int}): maps each cache line (int) to its score (int).
        Lower score indicates more evictable.
    """
    raise NotImplementedError()

class BeladyScorer(CacheLineScorer):

    def __init__(self, trace: WikiTrace) -> None:
        super().__init__()
        self._trace = trace

    def __call__(self, cache_state: CacheState, access_times: Dict[int, int]) -> CacheLineScores:
        """ Returns cache line scores for each object
        
        The score will be lower if the reuse distance is longer,
        and be higher if the distance is shorter.
        """
        del access_times
        scores = {line: -self._trace.next_access_time(line) for line in cache_state.cache_lines}

        return scores

class LRUScorer(CacheLineScorer):
    """Scores cache lines based on how recently each line was accessed.
    
    Specifically, the scores are just th access time.
    """
    def __call__(self, cache_state: CacheState, access_times: Dict[int, int]) -> CacheLineScores:
        """ Returns cache line scores for each object
        
        The score will be lower if the object was accessed more recently,
        and be higher if the object was accessed less recently.
        """
        scores = {line: access_times[line] for line in cache_state.cache_lines}

        return scores

class LearnedScorer(CacheLineScorer):
    """Cache line scorer that uses a learned model under the hood.""" 
    
    def __init__(self, scoring_model) -> None:
      """ Constructs a LearnedScorer from a given model.
      
        Args:
          scoring_model: the learned model to use for scoring.
          self._scoring_model = scoring_model
      """
      self._scoring_model = scoring_model
      self._hidden_state = None
    
    def __call__(self, cache_state: CacheState, access_times: Dict[int, int]) -> CacheLineScores:
      """ Returns cache line scores for each object
        
        The score will be lower if the object was accessed more recently,
        and be higher if the object was accessed less recently.
        """
      del access_times

      scores, _, self._hidden_state = self._scoring_model(
                                    [cache_state],
                                    self._hidden_state,
                                    inference=True)
      return {line: -scores[0, i].item() for i, line in enumerate(cache_state.cache_lines)}      

    @classmethod
    def from_model_checkpoint(self, config: Dict, model_checkpoint: Optional[str] = None) -> CacheLineScorer:
      """Creates scorer from a model loaded from the given checkpoint and config.
      
      Arg:
        config: the config to use for the model.
        model_checkpoint (str | None): path to acheckpoint for the model. Model uses default random
          initialization if no checkpoint is provided.
      
      Returns:
        scorer(CacheLineScorer) : the scorer using the given model.
      """ 
      device = "cpu"
      if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        device = "cuda:0"
      
      scoring_model = CachePolicyModel.from_config(config).to(torch.device(device))

      if model_checkpoint:
         with open(model_checkpoint, "rb") as f:
            scoring_model.load_state_dict(torch.load(f, map_location=device))

      return self(scoring_model)
        


class EvictionPolicy(six.with_metaclass(abc.ABCMeta, object)):
  """Policy for determining what cache line to evict."""

  @abc.abstractmethod
  def __call__(self, cache_state: CacheState, access_times: Dict[int, int]) -> tuple[tuple[List[Optional[int]]], CacheLineScores]:
    """Chooses a cache line to evict.

    Args:
      cache_state (CacheState): the cache state to choose an eviction for.
      access_times (Dict[int, int]): maps each cache line (int) to its access time (int).

    Returns:
      lines_to_evict (List[Optional[int]]: the cache lines the policy has chosen to evict.
        Return [] if there are no cache lines.
      scores (dict{int: int}): maps each cache line (int) to its score (int).
        Lower score means the policy prefers to evict the cache line more.
    """
    raise NotImplementedError()

class GreedyEvictionPolicy(EvictionPolicy):

    def __init__(self, scorer: CacheLineScorer) -> None:
        super().__init__()
        self._scorer = scorer
        pass

    def __call__(self, cache_state: CacheState, access_times: Dict[int, int]) -> tuple[List[Optional[int]], CacheLineScores]:
        scores = self._scorer(cache_state, access_times)
        lines_to_evict = sorted(scores.keys(), key=lambda line: scores[line])

        return lines_to_evict, scores

def generate_eviction_policy(scorer_type: str, 
                             trace: WikiTrace,
                             learned_policy=None,
                             model_checkpoint: str=None) -> EvictionPolicy:
    if scorer_type == "belady":
        return GreedyEvictionPolicy(BeladyScorer(trace))
    elif scorer_type == "lru":
        return GreedyEvictionPolicy(LRUScorer())
    elif scorer_type == "learned":
        if learned_policy is not None:
            scorer = LearnedScorer(learned_policy)
        elif model_checkpoint is not None:
            scorer = LearnedScorer.from_model_checkpoint(config["model"], model_checkpoint)
        else:
            raise ValueError("Must provide either a learned policy or a model checkpoint")
        return GreedyEvictionPolicy(scorer)
    else:
        raise ValueError("Unknown scorer: {}".format(scorer_type))