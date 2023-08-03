import torch.optim as optim
from utils import as_batches
from cache_policy_model import CachePolicyModel
from configuration import config
from generator import train_data_generator

# Create training datasets generator
training_datasets = train_data_generator(config["dataset"])

# Initialize the model and optimizer
model = CachePolicyModel.from_config(config["model"])
optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

# Train the model
for dataset in training_datasets:

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
        print(loss)

    # print("Generating training data...")
    # for cache_state, cache_decision in dataset:
    #     print("Training...")
    #     optimizer.zero_grad()
        # output = model(cache_state.cache_access,
        #                cache_state.cache_lines,
        #                cache_state.cache_history)
        # loss = model.loss(output, cache_decision)
        # loss.backward()
        # optimizer.step()