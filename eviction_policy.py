import six
import abc
from typing import Dict, List, Optional, TypedDict
from wiki_trace import WikiTrace

class CacheLineScores(TypedDict):
    obj_id: int
    score: int

class CacheLineScorer(six.with_metaclass(abc.ABCMeta, object)):
  """Scores cache lines based on how evictable each line is."""

  @abc.abstractmethod
  def __call__(self, cache_lines: Dict[int, int]):
    """Scores all the cache lines in the cache_access.

    Args:
      cache_access (CacheAccess): the cache access whose lines to score.

    Returns:
      scores (dict{int: int}): maps each cache line (int) to its score (int).
        Lower score indicates more evictable.
    """
    raise NotImplementedError()

class BeladyScorer(CacheLineScorer):

    def __init__(self, trace: WikiTrace) -> None:
        super().__init__()
        self._trace = trace

    def __call__(self, cache_lines: Dict[int, int]) -> CacheLineScores:
        """ Returns cache line scores for each object
        
        The score will be lower if the reuse distance is longer,
        and be higher if the distance is shorter.
        """
        scores = {line: -self._trace.next_access_time(line) for line in cache_lines.keys()}

        return scores

class EvictionPolicy(six.with_metaclass(abc.ABCMeta, object)):
  """Policy for determining what cache line to evict."""

  @abc.abstractmethod
  def __call__(self, cache_lines: Dict[int, int]) -> tuple[tuple[List[Optional[int]]], CacheLineScores]:
    """Chooses a cache line to evict.

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

    def __call__(self, cache_lines: Dict[int, int]) -> tuple[List[Optional[int]], CacheLineScores]:
        scores = self._scorer(cache_lines)
        lines_to_evict = sorted(scores.keys(), key=lambda line: scores[line])

        return lines_to_evict, scores
