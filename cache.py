from collections import namedtuple, deque
from eviction_policy import EvictionPolicy

class CacheAccess(namedtuple(
    "CacheAccess",
    ("time", "obj_id", "obj_size", "obj_type"))):
    """ A sigle access to a cache

    Consists of:
      - time (int): the time a access arrives
      - obj_id (int): the requested object id
      - obj_size (int): the size of the requested object
      - obj_type (int): the object type
    """
    pass

class CacheState(namedtuple(
    "CacheState",
    ("cache_access", "cache_lines", "cache_history"))):
    """ A cache state

    Consists of:
      - cache_access (CacheAccess): a single access to a cache
      - cache_lines (Dict[int, int]): maps of object id to its size
      - cache_history (List[CacheAccess]): a history of cache accesses
    """
    pass

class CacheDecision(namedtuple(
    "CacheDecision",
    ("evict", "cache_line_scores"))):
    """ Information about which cache line was evicted for a Cache Access.

    Consists of:
      - evict (bool): True if a cache line was evicted.
      - cache_line_scores (Dict): maps a cache line (int) to its score (int) as
      determined by an EvictionPolicy. Lower score indicates more evictable.
    """

class Cache(object):
    """ A cache object """
    def __init__(self, capacity: int, eviction_policy: EvictionPolicy,
                access_history_len=10000):
        """Constructs.

        Args:
          - capacity: Number of bytes to store in cache
          - eviction_policy: A policy to evict a cache line
          - access_history_len: The length of history to keep
        """
        self._capacity = capacity
        # (Dict[int, int]) Dictionary for cache lines,
        # including objects as a key and object byte size as a value
        # e.x. {1: 128, 2:256, 3: 512}
        self._cache_lines = {} 
        self._eviction_policy = eviction_policy
        self._cache_history = deque(maxlen=access_history_len)
        self._hit_rate_statistic = BernoulliProcessStatistic()

    def read(self, access: CacheAccess) -> tuple[CacheState, CacheDecision]:
        """Constructs.

        Args:
          obj_key: An identifier for an object
          obj_size: Requested Object size

        Reterns:
          cache_state (CacheState)
          cache_decision (CacheDecision)
        """

        # store the access to cache history
        self._cache_history.append(access)

        # create cache state
        cache_state = self._get_cache_state(access)

        # compute scores and identify the line to evict by policy
        lines_to_evict, scores = self._eviction_policy(self._cache_lines)

        # The case of cache hit
        hit = access.obj_id in self._cache_lines
        # Record cache hit/miss
        self._hit_rate_statistic.trial(hit)
        if hit:
            return cache_state, CacheDecision(False, scores)

        # The case of cache miss
        next_cache_size = self._current_cache_used() + access.obj_size

        # Check whether cache line should be evicted
        evict = next_cache_size < self._capacity
        
        # Evict cache lines until the current cache size is less than capacity
        while next_cache_size > self._capacity:
            line_to_evict = lines_to_evict.pop(0)
            next_cache_size -= self._cache_lines[line_to_evict]
            del self._cache_lines[line_to_evict]
        
        # store the object to cache line
        self._cache_lines[access.obj_id] = access.obj_size

        return cache_state, CacheDecision(evict, scores)

    def _current_cache_used(self) -> int:
        """ Calculate the current used size

        Return:
          - size (int): total size of cache line objects
        """
        return sum([v for v in self._cache_lines.values()])

    def _get_cache_state(self, access: CacheAccess) -> CacheState:
        """ Build cache state

        Args:
          - access (CacheAccess)

        Returns:
          - cache_state (CacheState)
        """
        return CacheState(
            access,
            list(self._cache_lines.keys()),
            self._cache_history
        )

    @property
    def hit_rate_statistic(self):
        """Returns the hit_rate_statistic provided to the constructor.

        Returns:
          - BernoulliProcessStatistic
        """
        return self._hit_rate_statistic
    
class BernoulliProcessStatistic(object):

    def __init__(self) -> None:
        self.reset()

    def trial(self, success) -> None:
        self._trials += 1
        if success:
            self._successes += 1

    @property
    def num_trials(self) -> int:
        return self._trials

    @property
    def num_successes(self) -> int:
        return self._successes
    
    def success_rate(self) -> float:
        if self.num_trials == 0:
            raise ValueError("Success rate is undefined when num_trials is 0.")
        return self.num_successes / self.num_trials

    def reset(self) -> None:
        self._successes = 0
        self._trials = 0