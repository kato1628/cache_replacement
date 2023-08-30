from __future__ import annotations
import numpy as np # to avoid circular import
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
    def from_model_checkpoint(self, config: Dict, model_checkpoint: Optional[str] = None, eval: bool = False) -> CacheLineScorer:
      """Creates scorer from a model loaded from the given checkpoint and config.
      
      Arg:
        config: the config to use for the model.
        model_checkpoint (str | None): path to acheckpoint for the model. Model uses default random
          initialization if no checkpoint is provided.
        eval (bool): whether to put the model in eval mode.
      
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
      else:
          print("WARNING: No model checkpoint provided, using default random initialization")

      if eval:
          print("Putting model in eval mode")
          scoring_model.eval()

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

class RandomPolicy(EvictionPolicy):
    """Randomly chooses a cache line to evict."""

    def __init__(self, seed: int = 0) -> None:
      super().__init__()
      self._random = np.random.RandomState(seed)

    def __call__(self, cache_state: CacheState, access_times: Dict[int, int]) -> tuple[List[Optional[int]], CacheLineScores]:
      del access_times

      scores = {line: self._random.random() for line in cache_state.cache_lines}
      if not scores:
         return None, scores 

      lines_to_evict = sorted(scores.keys(), key=lambda line: scores[line]) 

      return lines_to_evict, scores


class MixturePolicy(EvictionPolicy):
  """Foollows different policies at each timestep.
  
  Given
    - N eviction policies: pi_1, ..., pi_N and
    - N probabilities(weights): p_1, ..., p_N with sum(p_i) = 1)
    
  At each timestep:
    Acts according to policy pi_i with probability p_i.
  """

  def __init__(self,policies: List[EvictionPolicy],
               weights: Optional[List[float]] = None,
               seed: int = 0, scoring_policy_index: Optional[int] = None) -> None:
      """Constructs a mixture policy from a list of policies and weights.
      
      Args:
        policies: the list of policies to use, pi_1, ..., pi_N.
        weights: the weights to use for each policy. If None, all policies are weighted equally.
        seed: the random seed to use for sampling.
        scoring_policy_index: policy always returns the scores according to policies[scoring_policy_index]
           , even if that policy was not used to choose a cache line to evict. If None, uses the same policy
           for choosing a line to evict and scores.
      """
      super().__init__()
      
      if weights is None:
          weights = [1.0 / len(policies)] * len(policies)
      
      if len(policies) != len(weights):
          raise ValueError(f"Need the same number of weights ({len(weights)}) as policies ({len(policies)})")
      
      if not np.isclose(sum(weights), 1.0):
          raise ValueError(f"Weights must sum to 1, but sum(weights) = {sum(weights)}")
      
      if scoring_policy_index is not None and not (0 <= scoring_policy_index < len(policies)):
          raise ValueError(f"scoring_policy_index must be in [0, {len(policies)}), but got {scoring_policy_index}")
      
      self._policies = policies
      self._weights = weights
      self._random = np.random.RandomState(seed)
      self._scoring_policy_index = scoring_policy_index
  
  def __call__(self, cache_state: CacheState, access_times: Dict[int, int]) -> tuple[List[Optional[int]], CacheLineScores]:
      policy = self._random.choice(self._policies, p=self._weights)
      lines_to_evict, scores = policy(cache_state, access_times)
 
      # Over write scores if we are using a different policy for scoring
      if self._scoring_policy_index is not None:
          scoring_policy = self._policies[self._scoring_policy_index]
          _, scores = scoring_policy(cache_state, access_times)
      
      return lines_to_evict, scores

def generate_eviction_policy(scorer_type: str, 
                             trace: WikiTrace,
                             learned_policy=None,
                             model_checkpoint: Optional[str]=None,
                             model_prob: Optional[List[float]]=None,
                             scoring_policy_index: Optional[int]=None) -> EvictionPolicy:
    """Generates an eviction policy from the given scorer type.
    
    Args:
      scorer_type: the type of scorer to use. One of "belady", "lru", "learned", "mixture".
      trace: the trace to use for the belady scorer.
      learned_policy: the learned policy to use for the learned scorer.
      model_checkpoint: the checkpoint to load the learned policy from.
      model_prob: the probability to use the learned policy for the mixture scorer.
      scoring_policy_index: the index of the policy to use for scoring. Only used for the mixture scorer.

    Returns:
      An eviction policy.
    """
    if scorer_type == "belady":
        return GreedyEvictionPolicy(BeladyScorer(trace))
    elif scorer_type == "lru":
        return GreedyEvictionPolicy(LRUScorer())
    elif scorer_type == "random":
        return RandomPolicy()
    elif scorer_type == "learned":
        return generate_eviction_policy_from_learned_model(learned_policy, model_checkpoint, eval=True)
    elif scorer_type == "mixture":
        oracle_policy = GreedyEvictionPolicy(BeladyScorer(trace))
        learned_policy = generate_eviction_policy_from_learned_model(learned_policy, model_checkpoint)
        return MixturePolicy([oracle_policy, learned_policy],
                              [1-model_prob, model_prob],
                              scoring_policy_index=scoring_policy_index)
    else:
        raise ValueError("Unknown scorer: {}".format(scorer_type))
    
def generate_eviction_policy_from_learned_model(learned_policy,
                                                model_checkpoint: Optional[str]=None,
                                                eval: bool = False) -> EvictionPolicy:
    """Generates an eviction policy from a given learned policy.
    
    Args:
      learned_policy: the learned policy to use.
      model_checkpoint: the checkpoint to load the learned policy from.
      eval: whether to put the model in eval mode.
    
    Returns:
      An eviction policy.
    """
    
    if learned_policy is not None:
        if eval:
            print("Putting model in eval mode")
            learned_policy.eval()
        scorer = LearnedScorer(learned_policy)
    elif model_checkpoint is not None:
        scorer = LearnedScorer.from_model_checkpoint(config["model"], model_checkpoint, eval=eval)
    else:
        raise ValueError("Must provide either a learned policy or a model checkpoint")
    
    return GreedyEvictionPolicy(scorer)