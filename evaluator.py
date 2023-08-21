import tqdm
from cache import Cache, CacheAccess
from eviction_policy import generate_eviction_policy
from wiki_trace import WikiTrace

def cache_hit_rate_evaluator(config, policy_model, model_checkpoint, max_examples=5000):
    """
    Measures the cache hit rate of the given policy model.
    Args:
        config: A dictionary containing the following keys:
            filepath (str): the path to the trace file.
            window_size (int): the maximum number of accesses to look ahead.
            capacity (int): the cache capacity (byte).
            access_history_len (int): the length of the access history.
        policy_model: The policy model to evaluate.
        model_checkpoint: The path to the model checkpoint file.
        max_examples: The maximum number of examples to evaluate.
    
    Yields:
        A list of cache hit rates for each example.
    """
    with WikiTrace(config["filepath"], max_look_ahead=config["window_size"]) as trace:
        eviction_policy = generate_eviction_policy(
                            config["scorer_type"],
                            None,
                            policy_model,
                            model_checkpoint,
                            1.0) # Always follow the model predictions
        cache = Cache(config["capacity"],
                      eviction_policy,
                      config["access_history_len"])

        desc = "Evaluating the model..."
        with tqdm.tqdm(desc=desc) as pbar:
            while not trace.done():
                cache_hit_rates = []

                # Reset the cache hit rate statistic
                cache.hit_rate_statistic.reset()

                while len(cache_hit_rates) <= max_examples and not trace.done():
                    time, obj_id, obj_size, obj_type = trace.next()
                    access = CacheAccess(time, obj_id, obj_size, obj_type)
                    cache.read(access)
                    cache_hit_rates.append(cache.hit_rate_statistic.success_rate())
                    pbar.update(1)

                yield cache_hit_rates