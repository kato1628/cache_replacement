config = {
    "experiment": {
        "base_dir": "./experiments",
        "name": "test",
    },
    "dataset": {
        "filepath": "./dataset/wiki2018_dev.tr",
        "window_size": 100000,
        # "capacity": 2000000000,
        "capacity": 500000000,
        "access_history_len": 20,
        "scorer_type": "belady"
    },
    "model": {
        "obj_id_embedder": {
            "type": "dynamic_vocab",
            # "max_vocab_size": 10000,
            "max_vocab_size": 5000,
            # "embedding_dim": 64,
            "embedding_dim": 32,
        },
        "obj_size_embedder": {
            "type": "dynamic_vocab",
            "max_vocab_size": 5000,
            # "embedding_dim": 64,
            "embedding_dim": 32,
        },
        "cache_lines_embedder": "obj_id_embedder",
        "positional_embedder": {
            "type": "positional",
            # "embedding_dim": 128,
            "embedding_dim": 64,
        },
        # "lstm_hidden_size": 128,
        "lstm_hidden_size": 64,
        "max_attention_history": 50,
    },
    "training": {
        "learning_rate": 0.001,
        # "batch_size": 32,
        "batch_size": 16,
        # "sequence_length": 80,
        "sequence_length": 40,
        # "update_frequency": 10000,
        "update_frequency": 10,
        # "collection_multiplier": 5,
        "collection_multiplier": 5,
        # "total_steps": 10000,
        "total_steps": 2000,
        # "save_frequency": 20000,
        "save_frequency": 20,
        # "evaluation_frequency": 400,
        "evaluation_frequency": 10,
        # "evaluation_size": 30000,
        "evaluation_size": 10000,
        "checkpoint_dir": "./result/checkpoints",
    }
}