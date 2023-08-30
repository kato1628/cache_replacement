import os
import tqdm
import pprint
import concurrent
import numpy as np
from typing import Dict, List, Tuple
from matplotlib import pyplot as plt
from wiki_trace import WikiTrace
from cache import Cache, CacheAccess
from common.utils import create_directory
from utils import load_pickle, save_pickle
from eviction_policy import generate_eviction_policy
from evaluation_configuration import eval_config
from concurrent.futures import ProcessPoolExecutor
pp = pprint.PrettyPrinter(indent=4)

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
                            trace,
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

def measure_chr(eval_config: Dict, save_path: str = None) -> List[float]:
    """
    Measure the cache hit rate by the given evaluation configuration.
    The result is saved to the given path.
    
    Args:
        eval_config (Dict): The evaluation configuration.
        save_path (str, optional): The path to save the result. Defaults to None.
    
    Returns:
        List[float]: A list of cache hit rates.
    """
    print(f"Measure cache hit rate by {eval_config['scorer_type']}")
    evaluator = cache_hit_rate_evaluator(eval_config,
                                         None, None,
                                         max_examples=5000)

    if os.path.exists(save_path):
        print(f"Load cache hit rate from {save_path}")
        return load_pickle(save_path)

    hit_rates = []
    step = 0
    for chr in evaluator:
        step += 1
        print(f"step {step}: {np.mean(chr)}")
        hit_rates.append(np.mean(chr))

    print(f"Average: {np.mean(hit_rates)}")

    if save_path:
        save_pickle(hit_rates, save_path)
    
    return hit_rates

def measure_chr_by_checkpoints(eval_config: Dict, checkpoints: List[str], save_path: str = None) -> Dict[str, List[float]]:
    """Measure the cache hit rate by checkpoints. The result is saved to the given path.
    
    Args:
        eval_config (Dict): The evaluation configuration.
        checkpoints (List[str]): The list of checkpoints to evaluate.
        save_path (str, optional): The path to save the result. Defaults to None.
    
    Returns:
        Dict[str, List[float]]: A dictionary mapping from checkpoint to a list of cache hit rates.
    """
    map_checkpoint_to_chr = {}
    for checkpoint in checkpoints:
        print(f"Checkpoint: {checkpoint}")
        if not os.path.exists(checkpoint):
            print(f"Checkpoint {checkpoint} does not exist.")
            continue

        evaluator = cache_hit_rate_evaluator(eval_config,
                                            None, checkpoint,
                                            max_examples=5000)
        step = 0
        map_checkpoint_to_chr[checkpoint] = []
        for hit_rates in evaluator:
            step += 1
            print(f"step {step}: {np.mean(hit_rates)}")
            map_checkpoint_to_chr[checkpoint].append(np.mean(hit_rates))
        print(f"Average: {np.mean(map_checkpoint_to_chr[checkpoint])}")

    for checkpoint, hit_rates in map_checkpoint_to_chr.items():
        print(f"{checkpoint}: {hit_rates}")
    
    if save_path:
        save_pickle(map_checkpoint_to_chr, save_path)

    return map_checkpoint_to_chr

def evaluate_checkpoint(checkpoint, eval_config: Dict) -> Tuple[str, List[float]]:
    """Evaluate the given checkpoint.
    
    Args:
        checkpoint (str): The checkpoint to evaluate.
        eval_config (Dict): The evaluation configuration.
    
    Returns:
        Tuple[str, List[float]]: A tuple of checkpoint and a list of cache hit rates.
    """
    print(f"Checkpoint: {checkpoint}")
    evaluator = cache_hit_rate_evaluator(eval_config, None, checkpoint, max_examples=5000)
    result = [np.mean(rates) for rates in evaluator]
    return checkpoint, result

def measure_chr_by_checkpoints_with_multi_process(eval_config: Dict, checkpoints: List[str], save_path: str = None) -> Dict[str, List[float]]:
    """Measure the cache hit rate by checkpoints with multi-process. The result is saved to the given path.
    
    Args:
        eval_config (Dict): The evaluation configuration.
        checkpoints (List[str]): The list of checkpoints to evaluate.
        save_path (str, optional): The path to save the result. Defaults to None.
    
    Returns:
        Dict[str, List[float]]: A dictionary mapping from checkpoint to a list of cache hit rates.
    """

    print("Measuring cache hit rate by checkpoints with multi-process...")
    map_checkpoint_to_chr = {}
    
    with ProcessPoolExecutor(max_workers=len(checkpoints)) as executor:
        futures = {}
        for checkpoint in checkpoints:
            if not os.path.exists(checkpoint):
                print(f"Checkpoint {checkpoint} does not exist.")
                continue
            # map future to checkpoint
            futures[executor.submit(evaluate_checkpoint, checkpoint, eval_config)] = checkpoint

        # futures = {executor.submit(evaluate_checkpoint, checkpoint, eval_config): checkpoint for checkpoint in checkpoints}
        
        for future in concurrent.futures.as_completed(futures):
            checkpoint, hit_rates = future.result()
            print(f"{checkpoint}: {hit_rates}")
            map_checkpoint_to_chr[checkpoint] = hit_rates

    if save_path:
        save_pickle(map_checkpoint_to_chr, save_path)

    return map_checkpoint_to_chr

def evaluate(experiment_id: str, multi_process: bool = False, benchmarking: bool = True, show_result: bool = True, save_graph: bool = True):
    """Evaluate the given experiment.
    
    Args:
        experiment_id (str): The experiment id.
        multi_process (bool, optional): Whether to use multi-process. Defaults to False.
        benchmarking (bool, optional): Whether to do benchmarking. Defaults to True.
        show_result (bool, optional): Whether to show the result. Defaults to True.
    """
    checkpoint_path_prefix = os.path.join("./result/checkpoints", experiment_id)
    config_path = os.path.join(checkpoint_path_prefix, "config.pkl")

    # load config
    config = load_pickle(config_path)
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(config)

    eval_filename = eval_config['filepath'].split("/")[2].split(".")[0]
    eval_result_path = os.path.join(checkpoint_path_prefix, f"result_{eval_filename}.pkl")
    
    # collect checkpoints
    start = config["training"]["save_frequency"]
    end = config["training"]["total_steps"] + 1
    step = start
    checkpoints_paths = [os.path.join(checkpoint_path_prefix, f"model_{x}.ckpt") for x in range(start, end, step)]

    # Save evaluation config
    eval_config_path = os.path.join(checkpoint_path_prefix, "eval_config.pkl")
    save_pickle(eval_config, eval_config_path, overwrite=True)

    # measure cache hit rate by checkpoints
    if os.path.exists(eval_result_path):
        print(f"experiment id {experiment_id} has been evaluated.")
    else:
        if multi_process:
            measure_chr_by_checkpoints_with_multi_process(eval_config, checkpoints_paths, eval_result_path)
        else:
            measure_chr_by_checkpoints(eval_config, checkpoints_paths, eval_result_path)

    if benchmarking:
        # create directory for saving cache hit rate
        save_path_prefix = os.path.join("./result/cache_hit_rates",
                                        eval_config["filepath"].split("/")[2].split(".")[0])
        if not os.path.exists(save_path_prefix):
            create_directory(save_path_prefix)

        benchmarks = ["lru", "belady", "random"]

        for benchmark in benchmarks:
            eval_config["scorer_type"] = benchmark
            benchmark_save_path = os.path.join(save_path_prefix, f"{benchmark}_result.pkl")
            if os.path.exists(benchmark_save_path):
                print(f"{benchmark} has been evaluated.")
            else:
                measure_chr(eval_config, benchmark_save_path)
    
    if show_result:
        show_graph(experiment_id, show_benchmark=benchmarking, save_graph=save_graph, show_params=True)


def plot_hit_rates(map_label_to_hit_rates: Dict[str, List[float]], save_path: str = None, params: bool = False):
    """Plot the cache hit rates by different policies.
    
    Args:
        map_label_to_hit_rates (Dict[str, List[float]]): A dictionary mapping from policy name to a list of cache hit rates.
    
    Returns:
        None
    """
    trace_boundaries = [5000*i for i in range(1, 11)]
    for label, hit_rates in map_label_to_hit_rates.items():
        plt.plot(trace_boundaries, hit_rates, label=label)

    plt.axis([4500, 55000, 0, 0.5])
    plt.xlabel('Visited trace')
    plt.ylabel('Cache Hit Rate')
    plt.title('Visited trace vs. Cache Hit Rate')
    # plt.plot([5000, 15000], [0.3, 0.3], 'g--', linewidth=1.2)
    # plt.plot([15000, 15000], [0, 0.3], 'g--', linewidth=1.2)
    plt.legend(loc="best")
    plt.grid(linestyle='-', axis='y')

    if params:
        fig = plt.gcf()  # Get the current figure
        fig.text(0.1, 0.01, params, fontsize=10, ha='left', va='top', wrap=False)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

def show_graph(experiment_id: str, show_benchmark: bool = True, save_graph: bool = True, show_params: bool = False):
    """Show the result of the given experiment.
    
    Args:
        experiment_id (str): The experiment id.
        show_benchmark (bool, optional): Whether to show the benchmark result. Defaults to True.
        save_graph (bool, optional): Whether to save the graph. Defaults to True.
    """
    checkpoint_path_prefix = os.path.join('./result/checkpoints', experiment_id)
    config_path = os.path.join(checkpoint_path_prefix, "config.pkl")
    eval_config_path = os.path.join(checkpoint_path_prefix, "eval_config.pkl")

    # print config
    config = load_pickle(config_path)
    # pp.pprint(config)

    # print eval config
    eval_config = load_pickle(eval_config_path)
    eval_filename = eval_config['filepath'].split("/")[2].split(".")[0]
    # pp.pprint(eval_config)
    
    result_path = os.path.join(checkpoint_path_prefix, f"result_{eval_filename}.pkl")

    # load learned policy result
    result = load_pickle(result_path)

    if show_benchmark:
        # load lru result
        benchmark_result_prefix = os.path.join("./result/cache_hit_rates", eval_filename)
        benchmarks = [("lru", "LRU"), ("belady", "Belady"), ("random", "Random")]

        # load lru result
        for benchmark, label in benchmarks:
            benchmark_result_path = os.path.join(benchmark_result_prefix, f"{benchmark}_result.pkl")
            result[label] = load_pickle(benchmark_result_path)
    
    if save_graph:
        graph_path_prefix = os.path.join("./result/outputs", eval_filename)
        if not os.path.exists(graph_path_prefix):
            create_directory(graph_path_prefix)
        graph_path = os.path.join(graph_path_prefix, f"{experiment_id}.png")
    else:
        graph_path = None

    if show_params:
        params = build_param_description(config)
    else:
        params = None
        
    # plot hit rates
    plot_hit_rates(result, graph_path, params)

def build_param_description(config: Dict) -> str:
    params = f"window_size: {config['dataset']['window_size']}, \n" \
                    f"capacity: {config['dataset']['capacity']}, \n" \
                    f"scorer_type: {config['dataset']['scorer_type']}, \n" \
                    f"obj_id_embedder: {config['model']['obj_id_embedder']['type']} \n" \
                    f"  max_vocab_size: {config['model']['obj_id_embedder']['max_vocab_size']} " \
                    f"  embedding_dim: {config['model']['obj_id_embedder']['embedding_dim']}, \n" \
                    f"obj_size_embedder: {config['model']['obj_size_embedder']['type']} \n" \
                    f"  embedding_dim: {config['model']['obj_size_embedder']['embedding_dim']},  " \
                    f"  max_size: {config['model']['obj_size_embedder']['max_size']},  " \
                    f"  max_vocab_size: {config['model']['obj_size_embedder']['max_vocab_size']},  \n" \
                    f"cache_lines_embedder: {config['model']['cache_lines_embedder']},  \n" \
                    f"positional_embedder: {config['model']['positional_embedder']['type']} " \
                    f"  embedding_dim: {config['model']['positional_embedder']['embedding_dim']},  \n" \
                    f"lstm_hidden_size: {config['model']['lstm_hidden_size']},  \n" \
                    f"max_attention_history: {config['model']['max_attention_history']},  \n" \
                    f"dagger_schedule:  \n" \
                    f"  num_steps: {config['dagger_schedule']['num_steps']}, " \
                    f"  update_frequency: {config['dagger_schedule']['update_frequency']}, \n" \
                    f"training: \n" \
                    f"  learning_rate: {config['training']['learning_rate']}, " \
                    f"  batch_size: {config['training']['batch_size']}, " \
                    f"  sequence_length: {config['training']['sequence_length']}, \n" \
                    f"  collection_multiplier: {config['training']['collection_multiplier']}, " \
                    f"  total_steps: {config['training']['total_steps']}, " \
                    f"  save_frequency: {config['training']['save_frequency']}, \n" \
    
    return params

if __name__ == "__main__":
    print("Enter experiment id:")
    experiment_id = input()
    evaluate(experiment_id, multi_process=True, benchmarking=True, show_result=True, save_graph=True)
    # show_graph(experiment_id, show_benchmark=True, save_graph=True, show_params=True)