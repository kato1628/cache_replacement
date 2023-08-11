import torch.optim as optim
import tqdm
from common.utils import create_experiment_directory
from cache_tensorboard import log_hit_rates
from utils import as_batches
from cache_policy_model import CachePolicyModel
from configuration import config
from generator import train_data_generator

def main():
    # Create experiment directory
    # experiment_dir = os.path.join(config["experiment"]["base_dir"],
    #                               config["experiment"]["name"])
    # create_experiment_directory(experiment_dir, overwrite=True)

    # Create tensorboard writer
    # tb_writer = create_tb_writer(experiment_dir)

    # Create training datasets generator
    training_datasets = train_data_generator(config["dataset"])

    # Initialize the model and optimizer
    model = CachePolicyModel.from_config(config["model"])
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    # Initialize the step counter
    step = 0
    get_step = lambda: step
    total_steps = config["training"]["total_steps"]

    with tqdm.tqdm(total=total_steps) as pbar:
        # Train the model
        for dataset, cache_hit_rates in training_datasets:

            # Log the hit rates
            # log_hit_rates("cache_hit_rates/train_belady_policy", cache_hit_rates, get_step())

            print("Training...")
            batch_size = config["training"]["batch_size"]
            sequence_length = config["training"]["sequence_length"]
            warmup_period = sequence_length // 2

            # Generate batches from dataset
            for batch_num, batch in enumerate(as_batches([dataset], batch_size, sequence_length)):
                optimizer.zero_grad()
                loss = model.loss(batch, warmup_period)
                loss.backward()
                optimizer.step()
                pbar.update(1)
                step += 1
                print(loss)

                # Break if the step counter exceeds the total number of steps
                if step >= total_steps:
                    break

                # Break out of inner loop to get next dataset
                if batch_num >= config["training"]["update_frequency"]:
                    break
            
if __name__ == "__main__":
    main()