import tqdm
from typing import Dict, List
from wiki_trace import WikiTrace
from cache import Cache, CacheAccess, EvictionEntry
from eviction_policy import generate_eviction_policy


def train_data_generator(config: Dict) -> tuple[List[EvictionEntry], List[float]]:
    """
    Generates training data from the trace file.

    Args:
        config: A dictionary containing the following keys:
            filepath (str): the path to the trace file.
            window_size (int): the maximum number of accesses to look ahead.
            capacity (int): the cache capacity (byte).
            access_history_len (int): the length of the access history.
            max_examples (int): the maximum number of examples to generate.
    
    Yields:
        A tuple of (train_data, cache_hit_rates), where train_data is a list of
        EvictionEntry objects and cache_hit_rates is a list of cache hit rates
        for each example.
    """
    with WikiTrace(config["filepath"], max_look_ahead=config["window_size"]) as trace:
        eviction_policy = generate_eviction_policy(trace, config["scorer_type"])
        cache = Cache(config["capacity"], eviction_policy, config["access_history_len"])

        desc = "Generating training data..."
        with tqdm.tqdm(desc=desc) as pbar:
            while not trace.done():
                train_data = []
                cache_hit_rates = []

                # Reset the cache hit rate statistic
                cache.hit_rate_statistic.reset()

                while len(train_data) <= config["max_examples"] and not trace.done():
                    time, obj_id, obj_size, obj_type = trace.next()
                    access = CacheAccess(time, obj_id, obj_size, obj_type)
                    eviction_entry = cache.read(access)
                    train_data.append(eviction_entry)
                    cache_hit_rates.append(cache.hit_rate_statistic.success_rate())
                    pbar.update(1)

                yield train_data, cache_hit_rates