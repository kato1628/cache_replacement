from cache import Cache, CacheAccess
from eviction_policy import BeladyScorer, GreedyEvictionPolicy
from wiki_trace import WikiTrace

FILE_DIR = './dataset/'
FILENAME = 'wiki2018_dev.tr'
WINDOW_SIZE = 100000
CAPACITY = 100000000

with WikiTrace(f"{FILE_DIR}{FILENAME}", max_look_ahead=WINDOW_SIZE) as trace:

    scorer = BeladyScorer(trace)
    eviction_policy = GreedyEvictionPolicy(scorer)
    cache = Cache(CAPACITY, eviction_policy)

    train_data = []

    while not trace.done():
        time, obj_id, obj_size, obj_type = trace.next()
        access = CacheAccess(time=time, obj_id=obj_id, obj_size=obj_size, obj_type=obj_type)
        cache_state, cache_decision = cache.read(access)
        train_data.append((cache_state, cache_decision))

    print(train_data[0])