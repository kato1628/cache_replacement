import io
import os
import numpy as np
import torch
import torch.optim as optim
import tqdm
from cache_tensorboard import log_hit_rates
from evaluator import measure_cache_hit_rate
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

    update_frequency = config["training"]["update_frequency"]
    batch_size = config["training"]["batch_size"]
    collection_multiplier = config["training"]["collection_multiplier"]
    max_examples = (update_frequency * batch_size * collection_multiplier)

    # Create training datasets generator
    training_datasets = train_data_generator(config["dataset"], max_examples)

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

                # Save model
                if step % config["training"]["save_frequency"] == 0 and step != 0:
                    save_path = os.path.join(config["training"]["checkpoint_dir"], f"model_{step}.ckpt")
                    with open(save_path, "wb") as save_file:
                        checkpoint_buffer = io.BytesIO()
                        torch.save(model.state_dict(), checkpoint_buffer)
                        print(f"Saving model at step {step}")
                        save_file.write(checkpoint_buffer.getvalue())

                # Evaluate model
                if step % config["training"]["evaluation_frequency"] == 0 and step != 0:
                    hit_rates = next(measure_cache_hit_rate(model,
                                                            config["dataset"],
                                                            config["training"]["evaluation_size"]))
                    print(f"Hit rates: {np.mean(hit_rates)}, step: {step}")
                    log_hit_rates("cache_hit_rates/train", hit_rates, get_step())

                # Break if the step counter exceeds the total number of steps
                if step >= total_steps:
                    break

                # Break out of inner loop to get next dataset
                if batch_num >= config["training"]["update_frequency"]:
                    break
            
if __name__ == "__main__":
    main()