import tqdm
from typing import List
from wiki_trace import WikiTrace
from cache import Cache, CacheAccess, EvictionEntry
from eviction_policy import BeladyScorer, GreedyEvictionPolicy


def train_data_generator(config):
    """
    Generates training data from the trace file.
    """
    with WikiTrace(config["filepath"], max_look_ahead=config["window_size"]) as trace:
        scorer = BeladyScorer(trace)
        eviction_policy = GreedyEvictionPolicy(scorer)
        cache = Cache(config["capacity"], eviction_policy, config["access_history_len"])

        desc = "Generating training data..."
        with tqdm.tqdm(desc=desc) as pbar:
            while not trace.done():
                train_data = []
                cache.hit_rate_statistic.reset()

                while len(train_data) <= config["max_examples"] and not trace.done():
                    time, obj_id, obj_size, obj_type = trace.next()
                    access = CacheAccess(time, obj_id, obj_size, obj_type)
                    eviction_entry = cache.read(access)
                    train_data.append(eviction_entry)
                    pbar.update(1)

                print("Cache hit rate:", cache.hit_rate_statistic.success_rate())
                yield train_data