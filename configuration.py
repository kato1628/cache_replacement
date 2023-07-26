config = {
    "dataset": {
        "filepath": "./dataset/wiki2018_dev.tr",
        "window_size": 100000,
        "capacity": 1000000000,
        "access_history_len": 10000,
        "max_examples": 500
    },
    "model": {
        "cache_access_embedder": {
            "type": "dynamic_vocab",
            "max_vocab_size": 100000,
            "embedding_dim": 32,
        },
        "cache_lines_embedder": {
            "type": "dynamic_vocab",
            "max_vocab_size": 100000,
            "embedding_dim": 128,
        },
        "cache_history_embedder": {
            "type": "dynamic_vocab",
            "max_vocab_size": 100000,
            "embedding_dim": 128,
        },
        "num_heads": 8,
        "num_layers": 6,
    },
    "training": {
        "learning_rate": 0.001,
        "batch_size": 32,
    }
}