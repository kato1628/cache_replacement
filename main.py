import tqdm
from cache import Cache, CacheAccess
from eviction_policy import BeladyScorer, GreedyEvictionPolicy
from wiki_trace import WikiTrace

FILE_DIR = './dataset/'
FILENAME = 'wiki2018_dev.tr'
WINDOW_SIZE = 100000
CAPACITY = 1000000000
ACCESS_HISTORY_LEN = 10000

with WikiTrace(f"{FILE_DIR}{FILENAME}", max_look_ahead=WINDOW_SIZE) as trace:

    scorer = BeladyScorer(trace)
    eviction_policy = GreedyEvictionPolicy(scorer)
    cache = Cache(CAPACITY, eviction_policy, ACCESS_HISTORY_LEN)

    train_data = []

    with tqdm.tqdm(total=10000) as pbar:
        while not trace.done():
            time, obj_id, obj_size, obj_type = trace.next()
            access = CacheAccess(time, obj_id, obj_size, obj_type)
            cache_state, cache_decision = cache.read(access)
            train_data.append((cache_state, cache_decision))
            pbar.update(1)

    print("Cache hit rate:", cache.hit_rate_statistic.success_rate())