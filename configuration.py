config = {
    "dataset": {
        "filepath": "./dataset/wiki2018_dev.tr",
        "window_size": 100000,
        "capacity": 1000000000,
        "access_history_len": 10000,
        "max_examples": 5000
    },
    "model": {
        "max_cache_access_vocab": 100000,
        "cache_access_embedding_dim": 32,
        "max_cache_lines_vocab": 100000,
        "cache_lines_embedding_dim": 128,
        "max_cache_history_vocab": 100000,
        "cache_history_embedding_dim": 128,
        "num_heads": 8,
        "num_layers": 6,
    },
    "training": {
        "learning_rate": 0.001,
        "batch_size": 32,
    }
}