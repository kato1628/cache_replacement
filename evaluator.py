import tqdm
from cache import Cache, CacheAccess
from eviction_policy import generate_eviction_policy
from wiki_trace import WikiTrace

def measure_cache_hit_rate(policy_model, config, max_examples):
    with WikiTrace(config["filepath"], max_look_ahead=config["window_size"]) as trace:
        eviction_policy = generate_eviction_policy(
                            "learned",
                            None,
                            policy_model)
        cache = Cache(config["capacity"], eviction_policy, config["access_history_len"])

        desc = "Evaluating the model..."
        step = 0
        with tqdm.tqdm(desc=desc) as pbar:
            while not trace.done():
                cache_hit_rates = []

                # Reset the cache hit rate statistic
                cache.hit_rate_statistic.reset()

                while step <= max_examples and not trace.done():
                    time, obj_id, obj_size, obj_type = trace.next()
                    access = CacheAccess(time, obj_id, obj_size, obj_type)
                    _ = cache.read(access)
                    cache_hit_rates.append(cache.hit_rate_statistic.success_rate())
                    step += 1
                    pbar.update(1)

                yield cache_hit_rates