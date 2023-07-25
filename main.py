import tqdm
from cache import Cache, CacheAccess
from eviction_policy import BeladyScorer, GreedyEvictionPolicy
from wiki_trace import WikiTrace

# Dataset configuration
FILE_DIR = './dataset/'
FILENAME = 'wiki2018_dev.tr'
TRACE_FILE_PATH = f'{FILE_DIR}{FILENAME}'

# Cache configuration
WINDOW_SIZE = 100000
CAPACITY = 1000000000
ACCESS_HISTORY_LEN = 10000

# Training configuration
MAX_EXAMPLES = 5000

def generate_training_data(trace_file_path: str = TRACE_FILE_PATH,
                           window_size: int = WINDOW_SIZE,
                           capacity: int = CAPACITY,
                           access_history_len: int = ACCESS_HISTORY_LEN,
                           max_examples: int = MAX_EXAMPLES):
    """
    Generates training data from the trace file.
    """
    with WikiTrace(trace_file_path, max_look_ahead=window_size) as trace:
        scorer = BeladyScorer(trace)
        eviction_policy = GreedyEvictionPolicy(scorer)
        cache = Cache(capacity, eviction_policy, access_history_len)

        with tqdm.tqdm(total=100000) as pbar:
            while not trace.done():
                train_data = []
                cache.hit_rate_statistic.reset()

                while len(train_data) <= max_examples and not trace.done():
                    time, obj_id, obj_size, obj_type = trace.next()
                    access = CacheAccess(time, obj_id, obj_size, obj_type)
                    cache_state, cache_decision = cache.read(access)
                    train_data.append((cache_state, cache_decision))
                    pbar.update(1)

                print("Cache hit rate:", cache.hit_rate_statistic.success_rate())
                yield train_data


train_data_generator = generate_training_data()
for train_data in train_data_generator:
    print("Generating training data...")