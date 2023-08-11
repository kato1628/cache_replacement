config = {
    "experiment": {
        "base_dir": "./experiments",
        "name": "test",
    },
    "dataset": {
        "filepath": "./dataset/wiki2018_test.tr",
        "window_size": 100000,
        "capacity": 1000000000,
        "access_history_len": 20,
        "max_examples": 2500,
        "scorer_type": "belady"
    },
    "model": {
        "obj_id_embedder": {
            "type": "dynamic_vocab",
            "max_vocab_size": 10000,
            "embedding_dim": 64,
        },
        "obj_size_embedder": {
            "type": "dynamic_vocab",
            "max_vocab_size": 5000,
            "embedding_dim": 64,
        },
        "cache_lines_embedder": "obj_id_embedder",
        "positional_embedder": {
            "type": "positional",
            "embedding_dim": 128,
        },
        "lstm_hidden_size": 128,
        "max_attention_history": 50,
    },
    "training": {
        "learning_rate": 0.001,
        # "batch_size": 32,
        "batch_size": 16,
        # "sequence_length": 80,
        "sequence_length": 20,
        "update_frequency": 15,
        "collection_multiplier": 5,
        "total_steps": 10000,
    }
}