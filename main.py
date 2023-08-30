import datetime
import io
import os
import numpy as np
import torch
import tensorflow as tf
import torch.optim as optim
import tqdm
from cache_tensorboard import log_hit_rates, log_scalar
from common.utils import create_directory
from evaluator import cache_hit_rate_evaluator
from utils import as_batches, save_pickle
from cache_policy_model import CachePolicyModel
from configuration import config
from generator import train_data_generator
from baselines.common import schedules

def schedule_from_config(config):
    """Create a schedule from a configuration dictionary."""
    if config["type"] == "linear":
        return schedules.LinearSchedule(config["num_steps"], config["final"], config["initial"])
    elif config["type"] == "constant":
        return schedules.ConstantSchedule(config["value"])
    else:
        raise ValueError(f"Unknown schedule type: {config['type']}")

def main():
    # Create a datetime string for saving models
    experiment_id =  datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    print("Experiment ID:", experiment_id)

    # Create experiment directory
    experiment_dir = os.path.join(config["experiment"]["base_dir"], 'tensorboard', experiment_id)
    create_directory(experiment_dir, overwrite=True)

    # Create tensorboard writer
    tb_writer = tf.summary.create_file_writer(experiment_dir)

    update_frequency = config["dagger_schedule"]["update_frequency"]
    batch_size = config["training"]["batch_size"]
    collection_multiplier = config["training"]["collection_multiplier"]
    max_examples = (update_frequency * batch_size * collection_multiplier)

    # Create dagger schedule
    dagger_schedule = schedule_from_config(config["dagger_schedule"])

    # Process everything on GPU if available
    device = torch.device("cpu")
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        device = torch.device("cuda:0")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    #     torch.set_default_device(device)

    print("Device:", device)

    # Initialize the model and optimizer
    model = CachePolicyModel.from_config(config["model"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    # Initialize the step counter
    step = 0
    get_step = lambda: step
    total_steps = config["training"]["total_steps"]
    
    # Create the checkpoint directory
    checkpoint_dir = os.path.join(config["training"]["checkpoint_dir"], experiment_id)
    create_directory(checkpoint_dir)

    # Save the configuration
    config_save_path = os.path.join(checkpoint_dir, "config.pkl")
    save_pickle(config, config_save_path)

    with tqdm.tqdm(total=total_steps) as pbar:
        # Create training datasets generator
        training_datasets = train_data_generator(config["dataset"],
                                                dagger_schedule,
                                                get_step,
                                                model,
                                                max_examples)
        # Train the model
        for dataset, cache_hit_rates in training_datasets:

            # Log the hit rates
            # log_hit_rates("cache_hit_rates/train_belady_policy", cache_hit_rates, get_step())

            print("Training...")
            sequence_length = config["training"]["sequence_length"]
            warmup_period = sequence_length // 2

            # Generate batches from dataset
            for batch_num, batch in enumerate(as_batches([dataset], batch_size, sequence_length)):
                optimizer.zero_grad(set_to_none=True)
                loss = model.loss(batch, warmup_period)
                loss.backward()
                optimizer.step()
                pbar.update(1)
                step += 1

                # log the loss
                if step % config["training"]["log_loss_frequency"] == 0 and step != 0:
                    loss_cpu = loss.cpu()
                    log_scalar(tb_writer, "loss/reuse_distance", loss_cpu.detach().numpy(), step)
                    print(f"Step: {step}, loss: {loss_cpu.detach().numpy()}")

                # Save model
                if step % config["training"]["save_frequency"] == 0 and step != 0:
                    save_path = os.path.join(checkpoint_dir, f"model_{step}.ckpt")
                    with open(save_path, "wb") as save_file:
                        checkpoint_buffer = io.BytesIO()
                        torch.save(model.state_dict(), checkpoint_buffer)
                        print(f"Saving model at step {step}")
                        save_file.write(checkpoint_buffer.getvalue())

                # Evaluate model
                # if step % config["training"]["evaluation_frequency"] == 0 and step != 0:
                #     hit_rates = next(cache_hit_rate_evaluator(config["dataset"],
                #                                             model,
                #                                             None,
                #                                             config["training"]["evaluation_size"]))
                #     print(f"Hit rates: {np.mean(hit_rates)}, step: {step}")
                #     log_hit_rates("cache_hit_rates/train", hit_rates, get_step())

                # Break if the step counter exceeds the total number of steps
                if step >= total_steps:
                    return 

                # Break out of inner loop to get next dataset
                if batch_num >= config["dagger_schedule"]["update_frequency"]:
                    break
            
if __name__ == "__main__":
    main()